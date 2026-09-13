# 🏏 Stumped AI — IPL Analytics GenAI Agent

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://stumped-ai.onrender.com/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-orange)
![BigQuery](https://img.shields.io/badge/Google_Cloud-BigQuery-4285F4)
![Multi-LLM](https://img.shields.io/badge/LLM-Multi--Provider_Router-9cf)

---

## Overview

**Stumped AI** is a multi-agent, LangGraph-powered conversational assistant that answers complex questions about the Indian Premier League (IPL). Rather than relying on hallucinated LLM knowledge, the system operates on a **Tool-Augmented Generation** architecture — it writes and executes real SQL against a Google Cloud BigQuery warehouse, generates interactive Plotly visualizations from the actual query results, and falls back to live web search (DuckDuckGo) for trivia that lives outside the database.

### Why This Exists

Traditional chatbots either hallucinate statistics or require rigid, pre-built dashboards. Stumped AI bridges the gap: users ask questions in plain English, and a deterministic graph of specialised agents collaborates to fetch, analyze, and visualize the answer — with full transparency into every SQL query and Python snippet executed.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Multi-Provider LLM Router](#multi-provider-llm-router)
- [Agent Deep Dive](#agent-deep-dive)
  - [Router (Intent Classification)](#1-router-intent-classification)
  - [Data Agent (BigQuery SQL Expert)](#2-data-agent-bigquery-sql-expert)
  - [Plot Agent (Visualization Scientist)](#3-plot-agent-visualization-scientist)
  - [Trivia Agent (Web Search)](#4-trivia-agent-web-search)
- [Tool Definitions](#tool-definitions)
- [LangGraph Workflow (State Machine)](#langgraph-workflow-state-machine)
- [Data Pipeline](#data-pipeline)
- [Streamlit Frontend](#streamlit-frontend)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Deployment](#deployment)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                        │
│  (app.py — Chat UI, SQL/Code expanders, DataFrame, Plotly HTML)  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ run_agent(prompt, history)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH STATE MACHINE                        │
│                     (ipl_agent/agent.py)                          │
│                                                                  │
│   START ──▶ route_query ──┬──▶ data_node ──▶ check_plot_needed   │
│                           │                   │            │     │
│                           │              plot_node        END    │
│                           │                   │                  │
│                           │                  END                 │
│                           │                                      │
│                           └──▶ trivia_node ──▶ END               │
└──────────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌────────────┐
        │ BigQuery │ │  Plotly   │ │ DuckDuckGo │
        │  (GCP)   │ │(Sandboxed)│ │ Web Search │
        └──────────┘ └──────────┘ └────────────┘
```

The system is composed of **three specialised sub-agents** orchestrated by a **deterministic LangGraph state machine**. Each agent is a ReAct loop (Reason → Act → Observe) built with `create_react_agent`, constrained to its own tool subset.

---

## Multi-Provider LLM Router

Instead of relying on a single LLM provider, Stumped AI uses a custom **Multi-Provider Router** (`RoutedChatModel`) that wraps 5 providers and 33 models behind a unified LangChain-compatible `BaseChatModel` interface.

```
                    ┌─────────────────┐
                    │  RoutedChatModel │
                    │  (LangChain)     │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  Groq (Fast) │  │ Gemini (Free)│  │ Mistral      │  ...
   │  3 keys      │  │  3 keys      │  │  3 keys      │
   └──────────────┘  └──────────────┘  └──────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **5 Providers** | Groq, Google Gemini, Mistral, NVIDIA NIM, OpenRouter |
| **33-Model Chain** | Models sorted by priority score (Reasoning × Speed). Strongest & fastest tried first. |
| **Auto-Failover** | If a model rate-limits or errors, the router instantly tries the next model/provider. |
| **Round-Robin Keys** | Multiple API keys per provider are rotated to distribute rate limits. |
| **Cooldown Tracking** | Rate-limited or errored models are temporarily suspended with configurable cooldown periods. |
| **LangChain Bridge** | `RoutedChatModel` extends `BaseChatModel` — fully compatible with `create_react_agent` and tool calling. |

### Architecture

```
ipl_agent/llm/
├── config.py              # Router settings (timeouts, cooldowns, defaults)
├── providers.py           # Provider endpoints + key reading from .env
├── models.py              # 33-model registry with priority scores
├── router.py              # Async failover engine (rate-limit aware)
└── langchain_wrapper.py   # RoutedChatModel — LangChain BaseChatModel bridge
```

---

## Agent Deep Dive

### 1. Router (Intent Classification)

**File**: `ipl_agent/agent.py` — `route_query()` function

The Router is not a full agent — it is a deterministic node in the graph that classifies every incoming user message into one of two intents: **`DATA`** or **`TRIVIA`**.

#### How It Works

```
User Message
     │
     ▼
┌────────────────────────┐
│  Keyword Scan (Fast)   │  ← Checks for "runs", "wickets", "plot", "chart", etc.
│  ~30 data keywords     │
└────────┬───────────────┘
         │ Match found? → Route: DATA
         │ No match?
         ▼
┌────────────────────────┐
│  LLM Fallback (Slow)  │  ← Sends a classification prompt to the LLM
│  "Output DATA or       │
│   TRIVIA strictly"     │
└────────┬───────────────┘
         │ Response contains "DATA"? → Route: DATA
         │ Otherwise → Route: TRIVIA
         │ Exception? → Fallback: DATA
         ▼
      Route Set
```

**Design Rationale**: The two-tier approach (keywords first, LLM fallback) minimises latency for the ~80% of queries that contain obvious statistical keywords, while still handling ambiguous edge cases via the LLM.

---

### 2. Data Agent (BigQuery SQL Expert)

**File**: `ipl_agent/agent.py` — `data_agent` (ReAct agent)  
**Tools**: `get_schema_info`, `execute_bq_query`, `resolve_player_name`

The Data Agent is the core analytical engine. It receives a natural language question and autonomously:

1. **Inspects table schemas** — calls `get_schema_info` to learn column names, types, and descriptions before writing any SQL.
2. **Resolves player names** — calls `resolve_player_name` to convert full names (e.g. "Virat Kohli") to the abbreviated database format (e.g. "V Kohli").
3. **Writes & executes SQL** — generates BigQuery-compliant SQL and runs it via `execute_bq_query`.
4. **Summarises results** — provides a concise text summary of the findings (raw data is rendered separately by the UI).

#### Execution Flow

```
User: "Top 5 run scorers in IPL 2024"
     │
     ▼
┌─────────────────────────────┐
│  1. get_schema_info          │  ← Fetches schema for batting_scorecard, matches
│     ["batting_scorecard",    │
│      "matches"]              │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  2. execute_bq_query         │  ← Writes and runs SQL with JOIN on match_id
│     SELECT batter,           │     to filter by season = '2024'
│     SUM(runs) ...            │
│     ORDER BY ... LIMIT 5     │
└──────────────┬──────────────┘
               │ Returns: CSV path + data preview
               ▼
┌─────────────────────────────┐
│  3. Text Summary             │  ← "The top 5 run scorers in 2024 were..."
└─────────────────────────────┘
```

#### Critical Rules Enforced via System Prompt

| # | Rule | Purpose |
|---|------|---------|
| 1 | Always call `get_schema_info` first | Prevents hallucinated column names |
| 2 | Call `resolve_player_name` for player queries | Handles abbreviated name format in DB |
| 3 | JOIN with `matches` for temporal queries | Scorecard tables lack a `season` column |
| 4 | Use `(SUM(runs)/SUM(balls))*100` for strike rate | Prevents incorrect `AVG(strike_rate)` |
| 5 | Filter by `batting_team` for team stats | Prevents pulling opponent batters |
| 6 | One tool call per LLM turn | Prevents parallel tool-call errors |
| 7 | Never generate plot code | Separation of concerns — Plot Agent's job |
| 8 | Auto-append `LIMIT 1000` | Safety guardrail for runaway queries |
| 9 | Season format awareness | Handles split-year seasons like `'2020/21'` |
| 10 | Use `bowling_scorecard` for wickets | `wickets` table includes run-outs, inflating counts |

---

### 3. Plot Agent (Visualization Scientist)

**File**: `ipl_agent/agent.py` — `plot_agent` (ReAct agent)  
**Tools**: `execute_plot_code` (Sandboxed)

The Plot Agent is invoked **after** the Data Agent whenever a CSV file was generated. It receives:
- The **CSV file path** from the Data Agent's query results
- The **SQL query** that generated the data (for schema context)
- The **original user question** (to determine chart type and title)

#### Execution Flow

```
Data Agent Output (CSV + SQL + Answer)
     │
     ▼
┌──────────────────────────────────┐
│  Context Trimming                │  ← Only user query + CSV path + SQL context
│  (Strips full message history    │     are passed to avoid context pollution
│   to prevent confusion)         │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  LLM writes Python code:        │
│  - pd.read_csv(r'<csv_path>')   │  ← Reads actual data, no hallucination
│  - plotly.express / graph_objects│
│  - fig.write_html(__OUTPUT_PATH__)│  ← Pre-injected variable
│  - template='plotly_dark'        │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  execute_plot_code tool          │  ← Sandboxed exec() with import whitelist
│  - Only pandas/plotly/numpy/math │
│  - os/subprocess/sys BLOCKED     │
│  - exec/eval/compile REMOVED     │
└──────────────────────────────────┘
```

#### Sandboxed Execution

The `execute_plot_code` tool runs LLM-generated Python in a restricted environment:
- **Import whitelist**: Only `pandas`, `plotly`, `numpy`, `json`, `datetime`, `math`, `re`, `collections`, `itertools`, `functools`, `textwrap` are allowed.
- **Blocked imports**: `os`, `sys`, `subprocess`, `socket`, `shutil`, `http`, `ctypes`, and all others are rejected with an `ImportError`.
- **Removed builtins**: `exec()`, `eval()`, and `compile()` are stripped from the sandbox's builtins dict.

#### Anti-Hallucination Design

The Plot Agent **never hardcodes data values**. It is instructed to always load from the CSV file generated by the Data Agent. This ensures the visualized data is identical to the queried data — no LLM fabrication of numbers.

---

### 4. Trivia Agent (Web Search)

**File**: `ipl_agent/agent.py` — `trivia_agent` (ReAct agent)  
**Tools**: `search_cricinfo`, `web_search`

The Trivia Agent handles questions that **don't require database queries** — player profiles, cricket rules, news, general knowledge, or ESPNcricinfo lookups.

#### Execution Flow

```
User: "Tell me about Virat Kohli's career"
     │
     ▼
┌─────────────────────────────┐
│  search_cricinfo             │  ← DuckDuckGo search: site:espncricinfo.com
│  query="Virat Kohli IPL"     │     Returns actual result snippets + URLs
│  search_type="player"        │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  web_search                  │  ← Broader DuckDuckGo search for cricket trivia
│  query="Virat Kohli career"  │     Returns top 5 results with sources
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  LLM synthesizes a response │  ← Combines real search data into a friendly answer
│  with source links           │
└─────────────────────────────┘
```

**Real Search**: Both tools use [DuckDuckGo](https://duckduckgo.com/) via the `ddgs` library — **no API key required**. `search_cricinfo` targets `site:espncricinfo.com` specifically, while `web_search` performs broader queries.

---

## Tool Definitions

Each tool is a `@tool`-decorated LangChain function with a Pydantic schema for argument validation.

| Tool | Agent | Description |
|------|-------|-------------|
| `get_schema_info` | Data | Fetches BigQuery table schemas (columns, types, descriptions). Results are cached in `_SCHEMA_CACHE` to avoid repeat API calls. |
| `execute_bq_query` | Data | Runs a SQL query on BigQuery, saves results as a CSV to `query_results/`, and returns a data preview (first 50 rows as JSON). Auto-appends `LIMIT 1000` if missing. |
| `resolve_player_name` | Data | Converts full player names to the abbreviated format used in the database. Uses exact match → surname match → initial narrowing → fuzzy substring fallback. Player names are cached in `_PLAYER_NAME_CACHE`. |
| `execute_plot_code` | Plot | Executes LLM-generated Python code in a sandboxed namespace. Only data/plotting imports are whitelisted. Injects `__OUTPUT_PATH__` for chart location. |
| `search_cricinfo` | Trivia | Performs a DuckDuckGo search targeting `site:espncricinfo.com`. Returns real result snippets with titles, summaries, and URLs. |
| `web_search` | Trivia | Performs a broader DuckDuckGo search for general cricket/IPL trivia. Returns top 5 results with source links. |

---

## LangGraph Workflow (State Machine)

The orchestration layer is a **compiled LangGraph `StateGraph`** with deterministic routing (no LLM-driven edge decisions after the initial classification).

### State Schema

```python
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]  # Full message history
    route: str                                # "data" | "trivia" | "plot" | "end"
    final_answer: str                         # Text response for the UI
    plot_path: str                            # Path to generated HTML chart
    query_data: str                           # Path to CSV from BigQuery
    sql_queries: list                         # All SQL queries executed
    plot_code: str                            # Python code used for plotting
```

### Graph Topology

```mermaid
graph TD
    START([START]) --> route_query
    route_query -->|"route = data"| data_node
    route_query -->|"route = trivia"| trivia_node
    data_node --> check_plot_needed
    check_plot_needed -->|"CSV exists"| plot_node
    check_plot_needed -->|"No CSV"| END_1([END])
    plot_node --> END_2([END])
    trivia_node --> END_3([END])
```

### Node-by-Node Walkthrough

| Node | Function | Input | Output | Description |
|------|----------|-------|--------|-------------|
| `route_query` | `route_query()` | User message | `route: "data" \| "trivia"` | Classifies intent using keywords + LLM fallback |
| `data_node` | `data_node()` | Full messages | `final_answer`, `sql_queries`, `query_data` (CSV path) | Invokes the Data Agent ReAct loop. Extracts SQL queries and CSV path from intermediate tool call messages. |
| `trivia_node` | `trivia_node()` | Full messages | `final_answer` | Invokes the Trivia Agent ReAct loop. |
| `check_plot_needed` | `check_plot_needed()` | `query_data` | `route: "plot" \| "end"` | Checks if a CSV file was generated. If yes → plot. Always auto-plots when data exists. |
| `plot_node` | `plot_node()` | `query_data`, `sql_queries`, user message | `plot_path`, `plot_code` | Constructs a trimmed context for the Plot Agent, invokes it, and extracts the generated HTML path. |

### Edge Definitions

```python
START ──────────────────▶ route_query
route_query ──(conditional)──▶ data_node | trivia_node
data_node ──────────────▶ check_plot_needed
check_plot_needed ──(conditional)──▶ plot_node | END
plot_node ──────────────▶ END
trivia_node ────────────▶ END
```

---

## Data Pipeline

**File**: `data_pipeline/parse_ipl.py`

The data pipeline is a standalone ETL script that downloads, parses, and uploads IPL match data to BigQuery.

### Pipeline Flow

```
1. DOWNLOAD     ── Fetches ipl_male_json.zip from cricsheet.org
        │
        ▼
2. EXTRACT      ── Unzips ~1100+ JSON files (one per match)
        │
        ▼
3. PARSE        ── Iterates every JSON file and extracts:
        │           • Match metadata (teams, venue, toss, result)
        │           • Player of the Match awards
        │           • Player registries (name → Cricsheet ID)
        │           • Ball-by-ball deliveries (runs, extras, wickets)
        │           • DRS reviews
        │           • Impact player replacements
        │
        ▼
4. AGGREGATE    ── Computes derived tables from raw deliveries:
        │           • Overs summary (runs/wickets per over, cumulative)
        │           • Batting scorecards (runs, balls, 4s, 6s, SR, dismissal)
        │           • Bowling scorecards (overs, maidens, wickets, economy)
        │           • Partnerships (per-wicket, individual contributions)
        │
        ▼
5. UPLOAD       ── Pushes all 12 tables to BigQuery via pandas_gbq
        │           • Mode: REPLACE (full refresh)
        │           • Applies column descriptions to BQ schema
        │
        ▼
6. LOG          ── Uploads processing_log table for auditability
```

### BigQuery Tables

| Table | Rows (approx.) | Description |
|-------|-----------------|-------------|
| `matches` | ~1,100 | Match-level metadata: teams, venue, result, toss |
| `mom` | ~1,100 | Player of the Match awards |
| `players` | ~25,000 | Player-team-match mapping with Cricsheet registry IDs |
| `totals` | ~2,500 | Innings-level totals (runs, wickets, overs, extras breakdown) |
| `deliveries` | ~250,000 | Ball-by-ball data with powerplay flags, boundaries, extras |
| `wickets` | ~14,000 | Dismissal details: kind, fielders, team score at fall |
| `reviews` | ~1,500 | DRS review outcomes |
| `replacements` | ~500 | Impact player / concussion substitutions |
| `overs` | ~35,000 | Per-over aggregations with cumulative team score |
| `batting_scorecard` | ~28,000 | Per-batter per-match stats with dismissal info |
| `bowling_scorecard` | ~18,000 | Per-bowler per-match stats with maiden/economy calculations |
| `partnerships` | ~20,000 | Wicket partnerships with individual batter contributions |

---

## Streamlit Frontend

**File**: `app.py`

The Streamlit app provides a chat interface with full transparency into the agent's reasoning.

### UI Features

| Feature | Implementation |
|---------|---------------|
| **Chat History** | Stored in `st.session_state.messages` — persists across reruns |
| **Agent Memory** | `st.session_state.agent_history` — last 10 messages passed to LangGraph for multi-turn context |
| **Status Indicator** | `st.status()` widget shows "Agent Workflow Initiated..." → "Analysis Complete!" |
| **Typewriter Effect** | `st.write_stream()` streams the final answer word-by-word |
| **SQL Expander** | Every SQL query executed is shown in a collapsible `st.expander` with syntax highlighting |
| **Plot Code Expander** | The Python visualization code is shown in a collapsible expander |
| **Raw Data Table** | The CSV query results are rendered as an interactive `st.dataframe` — this prevents LLM hallucination by showing the actual data |
| **Interactive Charts** | Plotly HTML charts are embedded inline via `components.html()` |
| **Auto Cleanup** | On startup, deletes files older than 2 days from `query_results/` and `plots/` |

### Data Flow

```
User types query
     │
     ▼
app.py calls run_agent(prompt, history)
     │
     ▼
LangGraph executes full workflow
     │
     ▼
Returns: { answer, plot_path, sql_queries, plot_code, csv_path }
     │
     ├──▶ Render answer text (typewriter stream)
     ├──▶ Render SQL in expanders
     ├──▶ Render plot code in expander
     ├──▶ Render CSV as st.dataframe
     └──▶ Render Plotly HTML chart inline
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **Orchestration** | LangGraph (StateGraph with deterministic routing) |
| **Agent Framework** | LangChain (`create_react_agent`, `@tool`) |
| **LLM Providers** | Groq, Google Gemini, Mistral, NVIDIA NIM, OpenRouter — via multi-provider router |
| **Database** | Google Cloud BigQuery |
| **Visualization** | Plotly (interactive HTML charts, sandboxed execution) |
| **Web Search** | DuckDuckGo (`ddgs`) — no API key required |
| **Data Source** | [Cricsheet.org](https://cricsheet.org/) (ball-by-ball JSON) |
| **Deployment** | Streamlit Community Cloud / Render |

---

## Project Structure

```
stumped-ai/
├── app.py                          # Streamlit frontend (chat UI)
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (gitignored)
├── .gitignore
│
├── ipl_agent/
│   ├── __init__.py
│   ├── agent.py                    # LangGraph workflow + 3 sub-agents + 6 tools
│   └── llm/                        # Multi-provider LLM router module
│       ├── __init__.py
│       ├── config.py               # Router settings (timeouts, cooldowns)
│       ├── providers.py            # 5 provider endpoints + key reading
│       ├── models.py               # 33-model registry with priority scores
│       ├── router.py               # Async failover engine
│       └── langchain_wrapper.py    # LangChain BaseChatModel bridge
│
├── data_pipeline/
│   └── parse_ipl.py                # ETL: Cricsheet JSON → BigQuery tables
│
├── service-account/                # GCP credentials (gitignored)
│   └── service-account.json
│
├── query_results/                  # Auto-generated CSV outputs (gitignored)
│   └── query_<uuid>.csv
│
└── plots/                          # Auto-generated Plotly HTML charts (gitignored)
    └── plot_<timestamp>.html
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Google Cloud Service Account with **BigQuery Data Viewer** and **BigQuery Job User** roles
- At least one LLM provider API key (see below)

### 1. Clone & Install

```bash
git clone https://github.com/Priyanshukv06/stumped-ai.git
cd stumped-ai
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root. The LLM router accepts **comma-separated** lists of API keys for each provider. You don't need all providers — **just one valid key for one provider is enough** to run the app.

```env
# GCP Config
GCP_PROJECT_ID=your_gcp_project_id
GCP_DATASET_ID=ipl_stats

# LLM Provider Keys (comma-separated for load balancing)
# Provide at least ONE of the following:
GROQ_API_KEYS=gsk_...
GEMINI_API_KEYS=AIza...
MISTRAL_API_KEYS=...
NVIDIA_API_KEYS=nvapi-...
OPENROUTER_API_KEYS=sk-or-v1-...
```

#### Where to Get API Keys

| Provider | Free Tier | Link |
|----------|-----------|------|
| **Groq** | ~14,400 req/day | [console.groq.com/keys](https://console.groq.com/keys) |
| **Google Gemini** | 1M context, generous free tier | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **Mistral** | ~1B tokens/month "Experiment" plan | [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys) |
| **NVIDIA NIM** | Credit-based free tier | [build.nvidia.com](https://build.nvidia.com/) |
| **OpenRouter** | ~20 RPM free (many free models) | [openrouter.ai/keys](https://openrouter.ai/keys) |

> **Note**: DuckDuckGo search (used by the Trivia Agent) is completely free and requires no API key.

### 3. GCP Credentials

Place your service account JSON key at:

```
service-account/service-account.json
```

The application auto-detects this file on startup. Alternatively, set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

### 4. Run the Data Pipeline (First Time Only)

```bash
python data_pipeline/parse_ipl.py
```

This downloads all IPL match data from Cricsheet and populates your BigQuery dataset.

### 5. Launch the App

```bash
streamlit run app.py
```

---

## Deployment

### Streamlit Community Cloud

1. Push your code to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your repo.
3. Set the **Main file path** to `app.py`.
4. Under **Advanced Settings > Secrets**, add your environment variables in TOML format:

```toml
GCP_PROJECT_ID = "your_project_id"
GCP_DATASET_ID = "ipl_stats"
GROQ_API_KEYS = "gsk_key1,gsk_key2"
GEMINI_API_KEYS = "AIza_key1,AIza_key2"
MISTRAL_API_KEYS = "key1,key2"
NVIDIA_API_KEYS = "nvapi-key1,nvapi-key2"
OPENROUTER_API_KEYS = "sk-or-v1-key1,sk-or-v1-key2"
```

5. For GCP credentials, add a `[gcp_service_account]` section in secrets with the contents of your `service-account.json`:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

Then in your app, load it with:
```python
import json, streamlit as st
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
```

### Render

| Setting | Value |
|---------|-------|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| **Environment Variables** | `GCP_PROJECT_ID`, `GCP_DATASET_ID`, all `*_API_KEYS` vars |
| **Secret Files** | Upload `service-account.json` via Render's Secret Files. Set `GOOGLE_APPLICATION_CREDENTIALS` to its path. |

---

## License

This project is for educational and portfolio purposes.  
IPL match data sourced from [Cricsheet.org](https://cricsheet.org/) under their open data license.
