"""Async fallback router across providers.

Tries models in priority order (strongest first) until one answers. Handles:
- Multi-key round-robin load balancing per provider
- Key rotation on rate-limit / auth failures
- Provider cooldowns with automatic recovery
- <think> block stripping from reasoning models
- Tool calling support via route_chat()

Adapted from the reference project (genai-project-idea). Extended with
route_chat() for LangChain tool-calling integration.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from .config import settings
from .models import MODELS, ModelInfo
from .providers import BASE_URLS, KEY_ENV_VARS, NO_PENALTY_PROVIDERS, all_keys

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
RATE_LIMITED = "rate_limited"
AUTH_FAILED = "auth_failed"
BAD_REQUEST = "bad_request"
SERVER_ERROR = "server_error"
UNKNOWN = "unknown"

# Failures about the KEY, not the provider. Retrying on a different key
# before cooling the provider makes sense for these.
KEY_SPECIFIC_FAILURES = frozenset({RATE_LIMITED, AUTH_FAILED})


def classify_error(exc: Exception) -> str:
    """Map a provider exception to a failure kind."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None)

    if status == 429:
        return RATE_LIMITED
    if status in (401, 403):
        return AUTH_FAILED
    if status == 400:
        return BAD_REQUEST
    if status and 500 <= int(status) < 600:
        return SERVER_ERROR

    text = str(exc).lower()
    if ("429" in text or "rate limit" in text or "ratelimit" in text
            or "quota" in text or "too many requests" in text
            or "try again in" in text or "retry after" in text):
        return RATE_LIMITED
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text \
            or "invalid api key" in text or "authentication" in text:
        return AUTH_FAILED
    if "timeout" in text or "timed out" in text or "connection" in text:
        return SERVER_ERROR
    if "context length" in text or "too long" in text or "maximum context" in text:
        return BAD_REQUEST
    return UNKNOWN


def parse_retry_after(exc: Exception) -> float | None:
    """Seconds to wait, if the provider said so."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    match = re.search(r"try again in ([\d.]+)\s*(ms|s|m)\b", str(exc), re.I)
    if match:
        value, unit = float(match.group(1)), match.group(2).lower()
        return value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
    return None


# ---------------------------------------------------------------------------
# <think> block stripping
# ---------------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.I)
_ORPHAN_THINK = re.compile(r"^\s*<think>.*", re.DOTALL | re.I)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models leak into content."""
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think>" in cleaned.lower():
        cleaned = _ORPHAN_THINK.sub("", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Provider health
# ---------------------------------------------------------------------------
@dataclass
class _Cooldown:
    until: float
    reason: str


class ProviderHealth:
    """Remembers which providers are temporarily unusable (in-memory only)."""

    def __init__(self):
        self._cooldowns: dict[str, _Cooldown] = {}

    def is_available(self, provider: str) -> bool:
        cooldown = self._cooldowns.get(provider)
        if not cooldown:
            return True
        if cooldown.until <= time.time():
            del self._cooldowns[provider]
            return True
        return False

    def seconds_remaining(self, provider: str) -> float:
        cooldown = self._cooldowns.get(provider)
        return max(0.0, cooldown.until - time.time()) if cooldown else 0.0

    def penalise(self, provider: str, kind: str, retry_after: float | None = None) -> float:
        """Put a provider on cooldown for a failure of `kind`. Returns seconds."""
        if kind == RATE_LIMITED:
            seconds = retry_after if retry_after else settings.ROUTER_RATE_LIMIT_COOLDOWN_S
        elif kind == AUTH_FAILED:
            seconds = settings.ROUTER_AUTH_FAILURE_COOLDOWN_S
        elif kind == SERVER_ERROR:
            seconds = settings.ROUTER_SERVER_ERROR_COOLDOWN_S
        else:
            return 0.0

        seconds = float(seconds)
        self._cooldowns[provider] = _Cooldown(time.time() + seconds, kind)
        return seconds

    def clear(self, provider: str | None = None) -> None:
        if provider:
            self._cooldowns.pop(provider, None)
        else:
            self._cooldowns.clear()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@dataclass
class _Attempt:
    model: ModelInfo
    kind: str
    detail: str
    elapsed: float
    key_index: int = 0
    key_count: int = 1


class RoutedLLMBackend:
    """Tries models in priority order (strongest first) until one answers."""

    def __init__(self):
        self._keys = all_keys()
        self._index = {provider: 0 for provider in KEY_ENV_VARS}
        self.health = ProviderHealth()
        self.last_attempts: list[_Attempt] = []

    # -- catalogue ---------------------------------------------------------
    def available_models(self) -> list[ModelInfo]:
        """Only models whose provider has a key configured."""
        return [m for m in MODELS if self._keys.get(m.provider)]

    def configured_providers(self) -> list[str]:
        return sorted(p for p, keys in self._keys.items() if keys)

    def chain(self) -> list[ModelInfo]:
        """Models to attempt, in priority order (strongest first)."""
        available = self.available_models()
        return sorted(available, key=lambda m: (m.priority, m.name))

    # -- main entry points -------------------------------------------------
    async def route(
        self,
        messages: list[dict],
        temperature: float = settings.TEMPERATURE,
        max_tokens: int = settings.MAX_TOKENS,
    ) -> str:
        """Text-only routing. Returns stripped content string."""
        msg = await self._route_internal(messages, temperature, max_tokens)
        return strip_reasoning(msg.content or "")

    async def route_chat(
        self,
        messages: list[dict],
        temperature: float = settings.TEMPERATURE,
        max_tokens: int = settings.MAX_TOKENS,
        tools: list[dict] | None = None,
    ):
        """Tool-calling routing. Returns the full OpenAI message object.

        Used by the LangChain wrapper to extract both content and tool_calls.
        """
        return await self._route_internal(messages, temperature, max_tokens, tools=tools)

    # -- core loop (shared by route and route_chat) ------------------------
    async def _route_internal(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
    ):
        """Attempt models in order. Returns the raw OpenAI message object."""
        order = self.chain()

        if not order:
            raise RuntimeError(
                "No API keys configured. Add comma-separated keys to your .env "
                "for at least one provider (GEMINI_API_KEYS, GROQ_API_KEYS, etc)."
            )

        self.last_attempts = []
        started = time.time()
        attempted = 0

        for candidate in order:
            if attempted >= settings.ROUTER_MAX_ATTEMPTS:
                break

            if not self.health.is_available(candidate.provider):
                remaining = self.health.seconds_remaining(candidate.provider)
                log.debug("Skipped %s (%0.fs cooldown)", candidate.provider, remaining)
                continue

            key_count = self._key_pass_count(candidate.provider)
            for key_pass in range(key_count):
                if attempted >= settings.ROUTER_MAX_ATTEMPTS:
                    break

                attempted += 1
                key_index = self._current_key_index(candidate.provider)
                attempt_started = time.time()

                try:
                    msg = await self._call_raw(
                        candidate, messages, temperature, max_tokens, tools)
                    self._advance_after_success(candidate.provider)
                    elapsed = time.time() - started
                    log.info(
                        "Router success: %s (%s) in %.1fs [key %d/%d, attempt %d]",
                        candidate.name, candidate.provider,
                        elapsed, key_index + 1, key_count, attempted
                    )
                    return msg

                except Exception as exc:
                    kind = classify_error(exc)
                    rotated = self._rotate_if_key_specific(
                        candidate.provider, kind, key_pass, key_count)
                    seconds = 0.0 if rotated else self.health.penalise(
                        candidate.provider, kind, parse_retry_after(exc))
                    self.last_attempts.append(_Attempt(
                        candidate, kind, str(exc)[:200],
                        time.time() - attempt_started, key_index, key_count))

                    note = f"{candidate.name}: {kind}"
                    if key_count > 1:
                        note += f" [key {key_index + 1}/{key_count}]"
                    if rotated:
                        note += " — retrying on next key"
                    elif seconds:
                        note += f" (cooldown {seconds:.0f}s)"
                    log.warning("Router: %s", note)

                    if not rotated:
                        break  # out of keys, or not a key problem

        raise RuntimeError(self._failure_summary())

    # -- internals ---------------------------------------------------------
    def _client(self, provider: str) -> AsyncOpenAI:
        keys = self._keys.get(provider) or []
        if not keys:
            raise RuntimeError(f"No API key configured for {provider}")
        return AsyncOpenAI(
            api_key=keys[self._index[provider] % len(keys)],
            base_url=BASE_URLS[provider],
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

    async def _call_raw(
        self,
        model: ModelInfo,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
    ):
        """Call a model and return the full OpenAI message object."""
        client = self._client(model.provider)
        kwargs = dict(
            model=model.id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=settings.TOP_P,
            stream=False,
        )
        if model.provider not in NO_PENALTY_PROVIDERS:
            kwargs["presence_penalty"] = 0.0
            kwargs["frequency_penalty"] = 0.0
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # Validate: must have either content or tool_calls
        content = strip_reasoning(msg.content or "")
        has_content = bool(content.strip())
        has_tools = hasattr(msg, "tool_calls") and msg.tool_calls
        if not has_content and not has_tools:
            raise RuntimeError("Model returned empty response (no content or tool calls)")

        return msg

    # -- key pool ----------------------------------------------------------
    def _advance_after_success(self, provider: str) -> None:
        """Round-robin: move to next key after success."""
        if settings.ROUTER_ROUND_ROBIN_KEYS:
            self._rotate_key(provider)

    def _key_pass_count(self, provider: str) -> int:
        """How many key passes to attempt per model."""
        if not settings.ROUTER_ROTATE_KEYS:
            return 1
        return max(1, len(self._keys.get(provider) or []))

    def _current_key_index(self, provider: str) -> int:
        keys = self._keys.get(provider) or []
        return self._index.get(provider, 0) % len(keys) if keys else 0

    def _rotate_key(self, provider: str) -> None:
        keys = self._keys.get(provider) or []
        if len(keys) > 1:
            self._index[provider] = (self._index[provider] + 1) % len(keys)

    def _rotate_if_key_specific(self, provider: str, kind: str,
                                key_pass: int, key_count: int) -> bool:
        """Advance to next key on key-specific failures, if untried keys remain."""
        if kind not in KEY_SPECIFIC_FAILURES or key_pass >= key_count - 1:
            return False
        self._rotate_key(provider)
        return True

    def _failure_summary(self) -> str:
        if not self.last_attempts:
            available = self.configured_providers()
            if not available:
                return ("No API keys configured. Copy .env.example to .env "
                        "and add at least one key.")
            return "Every provider is rate limited. Try again shortly."

        lines = ["Every provider tried failed for this request.", ""]
        for attempt in self.last_attempts:
            where = attempt.model.provider
            if attempt.key_count > 1:
                where += f" key {attempt.key_index + 1}/{attempt.key_count}"
            lines.append(f"  - {attempt.model.name} ({where}): "
                         f"{attempt.kind} — {attempt.detail}")
        return "\n".join(lines)
