"""
Centralized LLM router configuration.

Adapted from reference project's config.py — stripped to LLM-only settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


class Settings:
    """LLM and router settings from environment variables."""

    # ── GCP (used by tools, not the router — kept here for single source of truth) ──
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "adk-mini-project")
    GCP_DATASET_ID: str = os.getenv("GCP_DATASET_ID", "ipl_stats")

    # ── LLM Defaults ──
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.3      # Lower for deterministic SQL/code generation
    TOP_P: float = 0.95
    LLM_TIMEOUT_SECONDS: int = 120

    # ── Router Behaviour ──
    ROUTER_MAX_ATTEMPTS: int = 5
    ROUTER_ROTATE_KEYS: bool = True        # Failover: retry same model on next key
    ROUTER_ROUND_ROBIN_KEYS: bool = True   # Load balance: advance key after success
    ROUTER_RATE_LIMIT_COOLDOWN_S: int = 45
    ROUTER_SERVER_ERROR_COOLDOWN_S: int = 20
    ROUTER_AUTH_FAILURE_COOLDOWN_S: int = 3600


settings = Settings()
