"""Provider endpoints and key handling — the single source of truth.

Every provider here speaks the OpenAI chat-completions API, so one client class
covers all of them and only the base URL and key pool differ.

Ported from the reference project (genai-project-idea) — verified working 2026-09-11.
"""

import os


BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEYS",
    "groq": "GROQ_API_KEYS",
    "mistral": "MISTRAL_API_KEYS",
    "nvidia": "NVIDIA_API_KEYS",
    "openrouter": "OPENROUTER_API_KEYS",
}

# Endpoints that reject OpenAI's presence_penalty / frequency_penalty.
NO_PENALTY_PROVIDERS = frozenset({"gemini", "mistral"})


def read_keys(provider: str) -> list[str]:
    """Keys for one provider, from its comma-separated env var."""
    env_var = KEY_ENV_VARS.get(provider)
    if not env_var:
        return []
    return [k.strip() for k in os.getenv(env_var, "").split(",") if k.strip()]


def all_keys() -> dict[str, list[str]]:
    return {provider: read_keys(provider) for provider in KEY_ENV_VARS}
