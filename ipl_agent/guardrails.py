"""
Guardrails module for Stumped AI — input validation, topic gating, and SQL safety.
All guardrails are pure Python (zero cost, no paid services).
"""
import re


# ============================================================================
# 1. INPUT SANITIZATION
# ============================================================================

MAX_INPUT_LENGTH = 500  # characters

# Prompt injection patterns — blocks manipulation attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+(a|an)\s+",
    r"forget\s+(everything|all|your)",
    r"disregard\s+(your|all|previous)",
    r"new\s+instructions?\s*:",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(prompt|instructions|rules)",
    r"output\s+(the|your)\s+(system|initial)\s+",
    r"pretend\s+(you|to)\s+(are|be)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"developer\s+mode",
    r"override\s+(your|all|the)\s+(rules|instructions|prompt)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_input(user_input: str) -> tuple[str, str | None]:
    """
    Validates and sanitizes user input.
    Returns: (sanitized_input, error_message_or_None)
    """
    cleaned = user_input.strip()

    if not cleaned:
        return "", "Please enter a question."

    if len(cleaned) > MAX_INPUT_LENGTH:
        return "", (
            f"🚫 Query too long ({len(cleaned)} chars). "
            f"Please keep it under {MAX_INPUT_LENGTH} characters."
        )

    if _INJECTION_RE.search(cleaned):
        return "", "🛡️ That query pattern is not allowed. Please ask a genuine IPL cricket question."

    return cleaned, None


# ============================================================================
# 2. TOPIC VALIDATION (Cricket/IPL only)
# ============================================================================
#
# DESIGN NOTES (keyword selection):
#
# These keywords are an ALLOWLIST — if any keyword matches, the query proceeds
# directly without an LLM topic check. If NONE match, we ask the LLM: "Is this
# about cricket? YES/NO" (1 cheap call).
#
# INCLUDED: Only cricket-specific terms that are unlikely to appear in
#           non-cricket queries. Even if a non-cricket query slips through,
#           the Data Agent will fail to find relevant SQL (no real damage).
#
# EXCLUDED: Generic words like "top", "most", "best", "table", "average",
#           "record", "hit", "compare", "trend" — these match too many
#           non-cricket queries and would bypass the topic gate.
#
# EXCLUDED: Player names — impossible to list all cricketers. The LLM
#           topic classifier handles this perfectly:
#           "Tell me about Kapil Dev" → LLM: YES (cricketer) → allowed
#           "Tell me about Narendra Modi" → LLM: NO → blocked
#

CRICKET_KEYWORDS = frozenset({
    # Core sport terms
    "ipl", "cricket", "t20", "twenty20", "cricinfo",

    # Roles
    "batsman", "batter", "bowler", "fielder", "keeper", "wicketkeeper",
    "allrounder", "all-rounder", "opener", "spinner", "pacer", "fast bowler",

    # IPL team abbreviations (unambiguous)
    "csk", "rcb", "kkr", "srh", "pbks", "lsg",

    # IPL team names (specific enough)
    "mumbai indians", "chennai super kings", "royal challengers",
    "kolkata knight riders", "sunrisers", "rajasthan royals",
    "delhi capitals", "punjab kings", "gujarat titans", "lucknow super giants",

    # Stats & metrics (cricket-specific only)
    "runs", "wickets", "overs", "balls", "sixes", "fours",
    "boundary", "boundaries", "strike rate", "economy rate", "maiden",
    "extras", "wides", "no ball", "no-ball", "dot ball", "dot balls",
    "run rate", "net run rate", "nrr",

    # Phases of play
    "powerplay", "death overs", "middle overs",

    # Match events (cricket-specific)
    "innings", "super over", "toss", "run chase",

    # Awards (IPL-specific)
    "orange cap", "purple cap", "man of the match",

    # Dismissals (unambiguously cricket)
    "bowled", "stumped", "lbw", "run out", "hit wicket", "caught and bowled",

    # Milestones (cricket-specific)
    "century", "half century", "hat trick", "hat-trick",

    # Scorecard
    "scorecard", "scoreboard",

    # IPL-specific events
    "playoff", "qualifier", "eliminator",

    # Visualization intent (user wants data charted — indicates data use)
    "plot", "chart", "graph", "visualize",

    # Partnership
    "partnership",
})


def is_cricket_related(query: str) -> bool:
    """
    Fast keyword check — returns True if query likely relates to cricket/IPL.
    If False, the caller should invoke the LLM topic classifier (Tier 2).
    """
    q = query.lower()
    return any(kw in q for kw in CRICKET_KEYWORDS)


# ============================================================================
# 3. SQL SAFETY VALIDATION
# ============================================================================

FORBIDDEN_SQL_RE = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|INSERT|UPDATE|ALTER|CREATE|MERGE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

ALLOWED_TABLES = frozenset({
    "matches", "mom", "players", "totals", "deliveries", "wickets",
    "reviews", "replacements", "overs", "batting_scorecard",
    "bowling_scorecard", "partnerships",
})


def validate_sql(sql: str) -> str | None:
    """
    Returns error message if SQL is unsafe, None if OK.
    Blocks write/DDL operations and queries to unauthorized tables.
    """
    # Block write / DDL operations
    if FORBIDDEN_SQL_RE.search(sql):
        return "BLOCKED: Only SELECT queries are allowed. Write operations are forbidden."

    # Ensure only allowed tables are referenced
    table_refs = re.findall(r"`[^`]*\.([^`]+)`", sql)
    for ref in table_refs:
        table_name = ref.split(".")[-1] if "." in ref else ref
        if table_name not in ALLOWED_TABLES:
            return f"BLOCKED: Table '{table_name}' is not in the allowed dataset."

    return None
