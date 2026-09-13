"""Curated model registry — ordered by STRENGTH × SPEED combined.

Models are ranked so the STRONGEST + FASTEST models are tried first, then
strong but slower, then balanced, then lightweight. This optimizes for both
response quality AND latency.

Speed tiers (based on provider infra + measured latency):
  Groq:        Fastest inference (hardware-optimized, sub-second, ~30 RPM)
  Gemini:      Fast (generous free tier, good latency)
  Mistral:     Moderate (0.41s measured for Small, ~1B tokens/month free)
  NVIDIA NIM:  Moderate (credit-based, larger models slower)
  OpenRouter:  Slowest (queued, variable, limited daily cap on free)

ALL models verified with live inference calls — 2026-09-11.
Ported from the reference project (genai-project-idea).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """One selectable model. `provider` routes to a base URL and key pool."""
    name: str
    id: str
    provider: str
    priority: int = 100
    strength: int = 3

    @property
    def key(self) -> tuple[str, str]:
        """Stable identity. Ids can repeat across providers, so pair them."""
        return (self.provider, self.id)


# ======================================================================
# RANKING: Combined strength + speed score.
#   Strong + Fast  → tried first (best user experience)
#   Strong + Slow  → tried next (quality fallback)
#   Weak   + Fast  → lightweight fallback
#   Weak   + Slow  → last resort
# ======================================================================
_MODEL_DICTS = [
    # ══════════════════════════════════════════════════════════════
    # GROUP A — Strong + Fast (strength 4-5, Groq/Gemini speed)
    # Best of both worlds: great reasoning AND low latency.
    # ══════════════════════════════════════════════════════════════

    # Groq GPT-OSS 120B: strong (4) + fastest provider. Best first pick.
    {"name": "Groq GPT-OSS 120B",
     "id": "openai/gpt-oss-120b", "provider": "groq", "strength": 4, "priority": 10},

    # Groq Qwen 3.8 27B: strong (4) + fastest provider.
    {"name": "Groq Qwen 3.8 27B",
     "id": "qwen/qwen3.8-27b", "provider": "groq", "strength": 4, "priority": 15},

    # Gemini 3.8 Flash: strong (4) + fast Gemini infra. Newest Flash.
    {"name": "Gemini 3.8 Flash",
     "id": "gemini-3.8-flash", "provider": "gemini", "strength": 4, "priority": 20},

    # Gemini 3.7 Flash: strong (4) + fast.
    {"name": "Gemini 3.7 Flash",
     "id": "gemini-3.7-flash", "provider": "gemini", "strength": 4, "priority": 25},

    # Gemini 3.5 Flash: strong (4) + fast, most battle-tested.
    {"name": "Gemini 3.5 Flash",
     "id": "gemini-3.5-flash", "provider": "gemini", "strength": 4, "priority": 30},

    # ══════════════════════════════════════════════════════════════
    # GROUP B — Strong + Moderate (strength 4-5, Mistral/NVIDIA speed)
    # Great reasoning, slightly higher latency.
    # ══════════════════════════════════════════════════════════════

    # Mistral Magistral Medium: strong reasoning (4) + Mistral speed.
    {"name": "Mistral Magistral Medium",
     "id": "magistral-medium-latest", "provider": "mistral", "strength": 4, "priority": 40},

    # Mistral Medium: strong general (4) + Mistral speed.
    {"name": "Mistral Medium",
     "id": "mistral-medium-latest", "provider": "mistral", "strength": 4, "priority": 45},

    # NVIDIA Nemotron 3 Super 120B: strongest (5), moderate NIM speed.
    {"name": "NVIDIA Nemotron 3 Super 120B",
     "id": "nvidia/nemotron-3-super-120b-a12b", "provider": "nvidia", "strength": 5, "priority": 50},

    # NVIDIA DeepSeek V4 Flash: strong (4), NIM speed.
    {"name": "NVIDIA DeepSeek V4 Flash",
     "id": "deepseek-ai/deepseek-v4-flash-0731", "provider": "nvidia", "strength": 4, "priority": 55},

    # NVIDIA Mistral-Nemotron: strong (4), NIM speed.
    {"name": "NVIDIA Mistral-Nemotron",
     "id": "mistralai/mistral-nemotron", "provider": "nvidia", "strength": 4, "priority": 58},

    # Gemini 3.1 Pro: strongest (5) but heavily rate-limited (~5 RPM).
    # Placed here rather than first because it exhausts quickly.
    {"name": "Gemini 3.1 Pro",
     "id": "gemini-3.1-pro-preview", "provider": "gemini", "strength": 5, "priority": 60},

    # ══════════════════════════════════════════════════════════════
    # GROUP C — Balanced + Fast (strength 3, Groq/Gemini speed)
    # Decent quality, excellent speed. Good mid-chain options.
    # ══════════════════════════════════════════════════════════════

    # Groq Compound Beta: balanced (3) + Groq speed.
    {"name": "Groq Compound Beta",
     "id": "compound-beta", "provider": "groq", "strength": 3, "priority": 70},

    # Groq Qwen 3.6 27B: balanced (3) + Groq speed.
    {"name": "Groq Qwen 3.6 27B",
     "id": "qwen/qwen3.6-27b", "provider": "groq", "strength": 3, "priority": 72},

    # Gemini 3.6 Flash: balanced (3) + Gemini speed.
    {"name": "Gemini 3.6 Flash",
     "id": "gemini-3.6-flash", "provider": "gemini", "strength": 3, "priority": 75},

    # ══════════════════════════════════════════════════════════════
    # GROUP D — Balanced + Moderate (strength 3, Mistral/NVIDIA speed)
    # ══════════════════════════════════════════════════════════════

    # Mistral Magistral Small: reasoning (3) + Mistral speed.
    {"name": "Mistral Magistral Small",
     "id": "magistral-small-latest", "provider": "mistral", "strength": 3, "priority": 80},

    # Mistral Small: balanced (3), fastest Mistral (0.41s measured).
    {"name": "Mistral Small",
     "id": "mistral-small-latest", "provider": "mistral", "strength": 3, "priority": 82},

    # Mistral Ministral 14B: balanced (3) + Mistral speed.
    {"name": "Mistral Ministral 14B",
     "id": "ministral-14b-latest", "provider": "mistral", "strength": 3, "priority": 85},

    # NVIDIA Nemotron 3.5 Lightning 30B: balanced (3), NIM speed.
    {"name": "NVIDIA Nemotron 3.5 Lightning 30B",
     "id": "nvidia/nemotron-3.5-lightning-30b-a3b", "provider": "nvidia", "strength": 3, "priority": 88},

    # NVIDIA Nemotron Nano Omni 30B: reasoning-capable (3), NIM speed.
    {"name": "NVIDIA Nemotron Nano Omni 30B",
     "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "provider": "nvidia", "strength": 3, "priority": 90},

    # NVIDIA Muse Glimmer 30B: Meta creative model (3), NIM speed.
    {"name": "NVIDIA Muse Glimmer 30B",
     "id": "meta/muse-glimmer-30b", "provider": "nvidia", "strength": 3, "priority": 92},

    # ══════════════════════════════════════════════════════════════
    # GROUP E — Strong but Slow (strength 5, OpenRouter queued)
    # Great quality but OpenRouter free tier is queued and variable.
    # ══════════════════════════════════════════════════════════════

    # OR Nemotron 3 Super 120B (free): strongest (5), slow/queued.
    {"name": "OR Nemotron 3 Super 120B (free)",
     "id": "nvidia/nemotron-3-super-120b-a12b:free", "provider": "openrouter", "strength": 5, "priority": 100},

    # OR Gemma 4 31B (free): strong (5), slow/queued.
    {"name": "OR Gemma 4 31B (free)",
     "id": "google/gemma-4-31b-it:free", "provider": "openrouter", "strength": 5, "priority": 102},

    # OR Nemotron Nano Omni (free): balanced (3), slow/queued.
    {"name": "OR Nemotron Nano Omni (free)",
     "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "provider": "openrouter", "strength": 3, "priority": 105},

    # OR Poolside Laguna S (free): balanced (3), slow/queued.
    {"name": "OR Poolside Laguna S (free)",
     "id": "poolside/laguna-s-2.1:free", "provider": "openrouter", "strength": 3, "priority": 108},

    # ══════════════════════════════════════════════════════════════
    # GROUP F — Lightweight + Fast (strength 2, Groq/Gemini speed)
    # Quick fallbacks when stronger models are rate-limited.
    # ══════════════════════════════════════════════════════════════

    # Gemini 3.5 Flash-Lite: light (2) + very fast Gemini.
    {"name": "Gemini 3.5 Flash-Lite",
     "id": "gemini-3.5-flash-lite", "provider": "gemini", "strength": 2, "priority": 115},

    # Gemini 3.1 Flash-Lite: light (2) + fast.
    {"name": "Gemini 3.1 Flash-Lite",
     "id": "gemini-3.1-flash-lite", "provider": "gemini", "strength": 2, "priority": 117},

    # Groq GPT-OSS 20B: light (2) + fastest.
    {"name": "Groq GPT-OSS 20B",
     "id": "openai/gpt-oss-20b", "provider": "groq", "strength": 2, "priority": 120},

    # Groq Compound Beta Mini: light (2) + fastest.
    {"name": "Groq Compound Beta Mini",
     "id": "compound-beta-mini", "provider": "groq", "strength": 2, "priority": 122},

    # Mistral Ministral 8B: light (2) + Mistral speed.
    {"name": "Mistral Ministral 8B",
     "id": "ministral-8b-latest", "provider": "mistral", "strength": 2, "priority": 125},

    # ══════════════════════════════════════════════════════════════
    # GROUP G — Lightweight + Slow (strength 1-2, OpenRouter/last)
    # Absolute last resort.
    # ══════════════════════════════════════════════════════════════

    # OR Nemotron 3.5 Lightning (free): light (2), queued.
    {"name": "OR Nemotron 3.5 Lightning (free)",
     "id": "nvidia/nemotron-3.5-lightning:free", "provider": "openrouter", "strength": 2, "priority": 135},

    # OR Poolside Laguna XS (free): light (2), queued.
    {"name": "OR Poolside Laguna XS (free)",
     "id": "poolside/laguna-xs-2.1:free", "provider": "openrouter", "strength": 2, "priority": 138},

    # Mistral Ministral 3B: ultra-light (1), last-resort.
    {"name": "Mistral Ministral 3B",
     "id": "ministral-3b-latest", "provider": "mistral", "strength": 1, "priority": 145},

    # Groq Allam 2 7B: ultra-light (1), last-resort.
    {"name": "Groq Allam 2 7B",
     "id": "allam-2-7b", "provider": "groq", "strength": 1, "priority": 150},
]

MODELS: list[ModelInfo] = [
    ModelInfo(name=d["name"], id=d["id"], provider=d["provider"],
             priority=d.get("priority", 100), strength=d.get("strength", 3))
    for d in _MODEL_DICTS
]
