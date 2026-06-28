import os
import uuid
import json
import urllib.parse
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
import requests
import pandas as pd
from dotenv import load_dotenv

from google.cloud import bigquery
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal

# load .env from parent directory
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

# Automatically configure GCP credentials if local file exists
local_service_account = os.path.join(os.path.dirname(__file__), '..', 'service-account', 'service-account.json')
if os.path.exists(local_service_account) and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_service_account

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "adk-mini-project")
GCP_DATASET_ID = os.getenv("GCP_DATASET_ID", "ipl_stats")

# Ensure API key exists
nvidia_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
if not nvidia_key:
    raise ValueError("NVIDIA_NIM_API_KEY not found. Check your .env file.")

# Configure LLM using ChatNVIDIA wrapper for native NIM support
llm = ChatNVIDIA(
    model="google/diffusiongemma-26b-a4b-it", 
    api_key=nvidia_key
)
llm2 = ChatNVIDIA(
    model="qwen/qwen3.5-122b-a10b", 
    api_key=nvidia_key
)

# ==========================================
# 1. TOOL DEFINITIONS (LangChain @tool)
# ==========================================

_SCHEMA_CACHE = {}
_PLAYER_NAME_CACHE = []  # Loaded once, holds all distinct player names from BQ

class SchemaArgs(BaseModel):
    table_names: list[str] = Field(description="A list of table names to get the schema for, e.g. ['matches', 'batting_scorecard'].")

@tool(args_schema=SchemaArgs)
def get_schema_info(table_names: list[str]) -> str:
    """Returns the BigQuery schema (column names, types, descriptions) for the requested tables to help write SQL."""
    global _SCHEMA_CACHE
    client = bigquery.Client(project=GCP_PROJECT_ID)
    results = {}
    for table_name in table_names:
        if table_name in _SCHEMA_CACHE:
            results[table_name] = _SCHEMA_CACHE[table_name]
            continue
        try:
            table_ref = f"{GCP_PROJECT_ID}.{GCP_DATASET_ID}.{table_name}"
            table = client.get_table(table_ref)
            columns = [{"name": field.name, "type": field.field_type, "description": field.description or ""} for field in table.schema]
            schema_data = {"columns": columns, "row_count": table.num_rows}
            results[table_name] = schema_data
            _SCHEMA_CACHE[table_name] = schema_data
        except Exception as e:
            results[table_name] = {"error": str(e)}
    return json.dumps(results, indent=2)

class BQArgs(BaseModel):
    sql_query: str = Field(description="The BigQuery standard SQL query string to execute.")

@tool(args_schema=BQArgs)
def execute_bq_query(sql_query: str) -> str:
    """Executes SQL on BigQuery and saves the result to a unique CSV in GCS or locally."""
    if "LIMIT" not in sql_query.upper():
        sql_query += " LIMIT 1000"
        
    print(f"\n[Tool: execute_bq_query] Running SQL:\n{sql_query}\n")
    client = bigquery.Client(project=GCP_PROJECT_ID)
    
    try:
        query_job = client.query(sql_query, timeout=45)
        df = query_job.to_dataframe()
        
        unique_id = uuid.uuid4().hex
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'query_results')
        os.makedirs(results_dir, exist_ok=True)
        local_path = os.path.join(results_dir, f"query_{unique_id}.csv")
        df.to_csv(local_path, index=False)
            
        return f"Query successful! Saved {len(df)} rows to {local_path}. Data preview: {df.head(50).to_json()}"
        
    except Exception as e:
        return f"BigQuery Execution Error: {str(e)}"

PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

@tool
def execute_plot_code(code: str) -> str:
    """Executes Python plotting code locally to generate an interactive Plotly chart saved as an HTML file."""
    print("\n[Tool: execute_plot_code] Running Plotly code locally...\n")
    
    import io, sys, webbrowser
    
    # Determine output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.abspath(os.path.join(PLOT_OUTPUT_DIR, f"plot_{timestamp}.html"))
    
    # Inject the output path so the code can reference it
    exec_globals = {"__OUTPUT_PATH__": output_path}
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    
    try:
        exec(code, exec_globals)
        console_output = captured.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return f"ERROR executing plot code: {str(e)[:2000]}"
    finally:
        sys.stdout = old_stdout
    
    # Check if the file was created
    if os.path.exists(output_path):
        print(f"[Plot saved to: {output_path}]")
        try:
            webbrowser.open(f"file:///{output_path}")
            print("[Chart opened in browser!]")
        except Exception:
            pass
        return f"SUCCESS! Interactive chart saved to: {output_path} and opened in the browser."
    else:
        # Fallback: check if HTML was printed to stdout
        if "<html" in console_output.lower() or "<div" in console_output.lower():
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(console_output)
            print(f"[Plot saved from stdout to: {output_path}]")
            try:
                webbrowser.open(f"file:///{output_path}")
            except Exception:
                pass
            return f"SUCCESS! Interactive chart saved to: {output_path} and opened in the browser."
        
        return f"Code executed but no chart was generated. Console output:\n{console_output[:2000]}"

@tool
def search_cricinfo(query: str, search_type: str = "") -> str:
    """Provides a Google search link to find ESPNcricinfo profiles or matches.
    Args:
        query: The search string (e.g. 'Virat Kohli').
        search_type: 'player', 'match', or general.
    """
    try:
        # Append 'cricinfo' to ensure we get cricinfo results
        search_term = f"{query.strip()} cricinfo"
        encoded_query = urllib.parse.quote_plus(search_term)
        url = f"https://www.google.com/search?q={encoded_query}"
        
        if search_type == "player":
            return f"Found the Google search link for the player's Cricinfo profile: {url}"
        return f"Found the Google search link for this query: {url}"
    except Exception as e:
        return f"Search Error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Searches the internet for general cricket trivia, rules, or news."""
    return f"Search results for: {query}"

@tool
def resolve_player_name(player_name: str) -> str:
    """Resolves a full player name (e.g. 'Virat Kohli') to the abbreviated format used in the database (e.g. 'V Kohli'). MUST be called before writing SQL that filters by player name."""
    global _PLAYER_NAME_CACHE
    client = bigquery.Client(project=GCP_PROJECT_ID)
    
    # Load cache once
    if not _PLAYER_NAME_CACHE:
        rows = list(client.query(
            f"SELECT DISTINCT player_name FROM `{GCP_PROJECT_ID}.{GCP_DATASET_ID}.players`"
        ).result())
        _PLAYER_NAME_CACHE = [r.player_name for r in rows]
        print(f"[Player Cache] Loaded {len(_PLAYER_NAME_CACHE)} player names")
    
    name_lower = player_name.strip().lower()
    
    # 1. Exact match (case-insensitive)
    for p in _PLAYER_NAME_CACHE:
        if p.lower() == name_lower:
            return f"Exact match found: '{p}'"
    
    # 2. Surname match (last word of the input)
    surname = name_lower.split()[-1] if name_lower.split() else name_lower
    surname_matches = [p for p in _PLAYER_NAME_CACHE if p.lower().endswith(surname) or surname in p.lower().split()]
    
    if len(surname_matches) == 1:
        return f"Resolved: '{player_name}' -> '{surname_matches[0]}'"
    elif len(surname_matches) > 1:
        # Try to narrow down by first initial
        first_initial = name_lower[0] if name_lower else ""
        narrowed = [p for p in surname_matches if p.lower().startswith(first_initial)]
        if len(narrowed) == 1:
            return f"Resolved: '{player_name}' -> '{narrowed[0]}'"
        return f"Multiple matches for '{player_name}': {surname_matches[:8]}. Use the most relevant one in your SQL."
    
    # 3. Fuzzy substring fallback
    parts = name_lower.split()
    for part in parts:
        if len(part) > 2:
            fuzzy_matches = [p for p in _PLAYER_NAME_CACHE if part in p.lower()]
            if fuzzy_matches:
                return f"Possible matches for '{player_name}': {fuzzy_matches[:8]}. Use the most relevant one in your SQL."
    
    return f"No match found for '{player_name}'. Try using LIKE '%{surname}%' in your SQL query."

# ==========================================
# 2. SUB-AGENTS (using create_react_agent)
# ==========================================

data_agent = create_react_agent(
    llm, 
    tools=[get_schema_info, execute_bq_query, resolve_player_name],
    prompt=f"""You are a BigQuery SQL Expert working with IPL cricket data.
The BigQuery project is '{GCP_PROJECT_ID}' and dataset is '{GCP_DATASET_ID}'.
All tables must be referenced as `{GCP_PROJECT_ID}.{GCP_DATASET_ID}.<table_name>`.

Available tables: matches, mom, players, totals, deliveries, wickets, reviews, replacements, overs, batting_scorecard, bowling_scorecard, partnerships.

CRITICAL RULES AND CONSTRAINTS:
1. YOU MUST ALWAYS call 'get_schema_info' FIRST to verify table schemas before writing any SQL.
2. DO NOT guess table or column names. 
3. PLAYER NAME RESOLUTION: Player names in the database use abbreviated format (e.g. 'V Kohli' not 'Virat Kohli', 'C Bosch' not 'Corbin Bosch'). When the user mentions a specific player by full name, you MUST call 'resolve_player_name' FIRST to find the correct abbreviated name before writing SQL.
4. TEMPORAL LOGIC: If a user asks for "this season", "current season", or a specific year, you MUST JOIN the relevant scorecard/deliveries table with the `matches` table on `match_id` to filter by the `season` or `date` column. (Example: `batting_scorecard` does not have a `season` column, it must be joined with `matches`).
5. DATA COMPLETENESS: The database contains matches up to the 2026 season. NEVER claim that 2024, 2025, or 2026 has not occurred yet. Rely strictly on the database.
6. To find the "current" season, you may need to query the max season from the `matches` table first, or use a subquery.
7. CRICKET MATH: NEVER calculate overall strike rate using AVG(strike_rate). You MUST calculate it as: `(SUM(runs) / SUM(balls)) * 100`. 
8. TEAM FILTERING: When asked for a specific team's stats (e.g. "most runs for Mumbai Indians"), ensure you filter by `batting_team = 'Mumbai Indians'` in the scorecard, rather than just pulling all batters from matches where the team played.
9. IMPORTANT NIM LIMITATION: You MUST ONLY output ONE tool call at a time. Do NOT attempt to call 'get_schema_info' and 'execute_bq_query' simultaneously in a single response. Wait for the result of the first tool before calling the next.
10. Use the 'execute_bq_query' tool to run your SQL. Provide a concise text summary of the results. DO NOT output markdown tables of the raw data in your final answer; the raw CSV data will be rendered automatically by the UI.
11. NEVER generate plotting code (matplotlib, plotly, etc). Your job is ONLY to query data and present results as text/tables. Plotting is handled by a separate agent."""
)

plot_agent = create_react_agent(
    llm,
    tools=[execute_plot_code],
    prompt=f"""You are an expert Data Visualization Scientist. You write flawless Python code to generate interactive Plotly charts.

CRITICAL RULES:
1. DATA INGESTION: Read the provided instructions carefully. If a CSV file path is provided, you MUST use `pd.read_csv(r'<path>')` to load the exact data to prevent hallucinations. If no CSV is provided, hardcode the data.
2. CHART ARCHITECTURE: Use Plotly (`import plotly.express as px` or `import plotly.graph_objects as go`) to create a beautiful, interactive chart.
3. SAVE THE CHART: You MUST save the chart using `fig.write_html(__OUTPUT_PATH__, include_plotlyjs='cdn')`. The variable `__OUTPUT_PATH__` is pre-injected and already available — do NOT define it yourself.
4. AESTHETICS: Apply `template='plotly_dark'`, set a clear title, add axis labels, and use a nice color scheme.
5. TOOL INVOCATION (ABSOLUTE MUST): You MUST call the `execute_plot_code` tool with your python code. 
6. NO JSON: NEVER output JSON configurations (like Chart.js). Your ONLY job is to write Python code and pass it to the `execute_plot_code` tool. Do not generate text blocks. Call the tool immediately."""
)

trivia_agent = create_react_agent(
    llm2,
    tools=[search_cricinfo, web_search],
    prompt="""You are the master IPL AI Assistant for web and trivia searches.
If asked about a specific player or match outside the database, use 'search_cricinfo'.
Provide a friendly, concise final answer including any search links."""
)


# ==========================================
# 3. DETERMINISTIC LANGGRAPH WORKFLOW
# ==========================================

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    final_answer: str
    plot_path: str
    query_data: str
    sql_queries: list
    plot_code: str

def get_last_user_message(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def route_query(state: GraphState):
    """Categorizes the user intent to route to Data or Trivia agent."""
    last_msg = get_last_user_message(state["messages"])
    
    print("\n[Router] Analyzing intent...")
    
    # 1. Deterministic Keyword Routing
    query_lower = last_msg.lower()
    data_keywords = [
        "score", "runs", "wickets", "highest", "lowest", "stats", "strike rate", "strike",
        "average", "powerplay", "matches", "points", "table", "economy", "bowled",
        "six", "sixes", "four", "fours", "boundary", "top", "most", "balls", "season", "hitter",
        "plot", "chart", "graph", "visualize", "visualization"
    ]
    if any(kw in query_lower for kw in data_keywords):
        print("[Router] Intent determined via keywords: DATA")
        return {"route": "data"}

    # 2. LLM Fallback Routing
    prompt = f"""Analyze the following user query.
We only have access to raw IPL database tables (matches, ball-by-ball deliveries, player stats).
If the query requires fetching IPL data, calculating statistics, aggregating records, or plotting/graphing results, output strictly "DATA".
If the query is NOT related to IPL, or asks for general information, rules, news, player profile links, or anything that doesn't require querying our database, output strictly "TRIVIA".
Query: {last_msg}"""
    try:
        response = llm.invoke(prompt).content.strip().upper()
        if "DATA" in response:
            print("[Router] Intent determined: DATA")
            return {"route": "data"}
        else:
            print("[Router] Intent determined: TRIVIA")
            return {"route": "trivia"}
    except Exception as e:
         print(f"[Router Warning] Falling back to DATA due to error: {e}")
         return {"route": "data"}

def decide_initial_route(state: GraphState) -> Literal["data_node", "trivia_node"]:
    if state["route"] == "data":
        return "data_node"
    return "trivia_node"

def data_node(state: GraphState):
    print("\n[Data Agent] Starting execution...")
    try:
        res = data_agent.invoke({"messages": state["messages"]}, {"recursion_limit": 30})
        
        # Extract SQL queries and CSV path from intermediate steps
        sql_queries = []
        csv_path = ""
        for msg in res.get("messages", []):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] == "execute_bq_query":
                        sql = tool_call["args"].get("sql_query", "")
                        if sql: sql_queries.append(sql)
            
            if hasattr(msg, "content") and isinstance(msg.content, str):
                if "Saved" in msg.content and ".csv" in msg.content and "Data preview" in msg.content:
                    parts = msg.content.split("to ")
                    if len(parts) > 1:
                        csv_path = parts[1].split(". Data preview")[0].strip()

        final_msg = res["messages"][-1]
        return {"messages": final_msg, "final_answer": final_msg.content, "sql_queries": sql_queries, "query_data": csv_path}
    except Exception as e:
        error_msg = AIMessage(content=f"Data Agent Error: {str(e)}")
        return {"messages": error_msg, "final_answer": error_msg.content, "sql_queries": [], "query_data": ""}

def trivia_node(state: GraphState):
    print("\n[Trivia Agent] Starting execution...")
    try:
        res = trivia_agent.invoke({"messages": state["messages"]}, {"recursion_limit": 30})
        final_msg = res["messages"][-1]
        return {"messages": final_msg, "final_answer": final_msg.content}
    except Exception as e:
        error_msg = AIMessage(content=f"Trivia Agent Error: {str(e)}")
        return {"messages": error_msg, "final_answer": error_msg.content}

def check_plot_needed(state: GraphState):
    """After data is fetched, always plot if a CSV was generated."""
    print("\n[Router] Checking if plot is needed...")
    
    csv_path = state.get("query_data", "")
    if csv_path and os.path.exists(csv_path):
        print("[Router] Plot requested: YES (Automatically plotting query results)")
        return {"route": "plot"}
        
    print("[Router] Plot requested: NO (No CSV data to plot)")
    return {"route": "end"}

def decide_plot_route(state: GraphState) -> Literal["plot_node", END]:
    if state["route"] == "plot":
        return "plot_node"
    return END

def plot_node(state: GraphState):
    print("\n[Plot Agent] Starting execution...")
    
    # Check if we have a CSV path from the Data Agent
    csv_path = state.get("query_data", "")
    if csv_path:
        # Use raw string for Windows paths
        csv_instruction = f"A CSV file containing the exact data has been generated at `{csv_path}`. You MUST use `pd.read_csv(r'{csv_path}')` to load the data. Do NOT hardcode the data."
    else:
        csv_instruction = "No CSV file available. Read the data from the previous messages and hardcode the data points into a pandas DataFrame."
        
    # Get the SQL query that generated the data to provide schema context
    sql_queries = state.get("sql_queries", [])
    sql_context = f"\n\nThe data in this CSV was generated using the following SQL query:\n```sql\n{sql_queries[-1]}\n```\nUse this to understand the schema and what the columns mean." if sql_queries else ""
    
    user_query_text = get_last_user_message(state['messages'])
    
    # Context Trimming: Only pass the original query and the data agent's final answer
    user_query = HumanMessage(content=f"{csv_instruction}{sql_context}\n\nGoal: You must generate a chart that perfectly answers the user's original query: '{user_query_text}'")
    data_answer = AIMessage(content=state.get("final_answer", ""))
    
    try:
        res = plot_agent.invoke({"messages": [user_query, data_answer]}, {"recursion_limit": 10})
        
        # Extract Plot code from intermediate steps
        plot_code = ""
        for msg in res.get("messages", []):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] == "execute_plot_code":
                        code = tool_call["args"].get("code", "")
                        if code: plot_code = code
                        
        final_msg = res["messages"][-1]
        
        # Determine plot path
        import glob
        list_of_files = glob.glob(os.path.join(PLOT_OUTPUT_DIR, '*.html'))
        latest_plot = max(list_of_files, key=os.path.getctime) if list_of_files else ""
        
        return {"messages": final_msg, "plot_path": latest_plot, "plot_code": plot_code}
    except Exception as e:
        error_msg = AIMessage(content=f"Plot Agent Error: {str(e)}")
        return {"messages": error_msg, "plot_path": "", "plot_code": ""}

# Compile the StateGraph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("route_query", route_query)
workflow.add_node("data_node", data_node)
workflow.add_node("trivia_node", trivia_node)
workflow.add_node("check_plot_needed", check_plot_needed)
workflow.add_node("plot_node", plot_node)

# Connect edges
workflow.add_edge(START, "route_query")
workflow.add_conditional_edges("route_query", decide_initial_route)

# Data path
workflow.add_edge("data_node", "check_plot_needed")
workflow.add_conditional_edges("check_plot_needed", decide_plot_route)
workflow.add_edge("plot_node", END)

# Trivia path
workflow.add_edge("trivia_node", END)

# Compile
ipl_graph = workflow.compile()

# ==========================================
# 4. API EXPORT (For Frontend)
# ==========================================
def run_agent(user_input: str, history: list) -> dict:
    """Executes the graph and returns structured output."""
    # Append the new message to history for this run
    inputs = {"messages": history + [HumanMessage(content=user_input)]}
    
    try:
        # invoke() runs the whole graph and returns the final state dict
        final_state = ipl_graph.invoke(inputs, {"recursion_limit": 30})
        
        # If there's a final answer, use it. Otherwise try to grab the last message content.
        answer = final_state.get("final_answer", "")
        if not answer and final_state.get("messages"):
            answer = final_state["messages"][-1].content
                        
        return {
            "answer": answer,
            "plot_path": final_state.get("plot_path", ""),
            "messages_update": final_state["messages"][-1] if final_state.get("messages") else None,
            "sql_queries": final_state.get("sql_queries", []),
            "plot_code": final_state.get("plot_code", "")
        }
    except Exception as e:
        return {
            "answer": f"System Error: {str(e)}",
            "plot_path": "",
            "messages_update": None,
            "sql_queries": [],
            "plot_code": ""
        }

# ==========================================
# 5. CLI LOOP
if __name__ == "__main__":
    print("========================================")
    print("   IPL LangGraph Deterministic Agent")
    print("========================================")
    print("Type 'exit' or 'quit' to stop.\n")
    
    chat_history = []
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                continue
                
            # Append user message to history
            chat_history.append(HumanMessage(content=user_input))
            
            # Keep only last 10 messages (5 turns) to prevent context bloat
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            
            inputs = {"messages": chat_history}
            
            final_state = None
            for output in ipl_graph.stream(inputs, stream_mode="updates"):
                for node_name, state_update in output.items():
                    final_state = state_update
                    if "messages" in state_update:
                        if hasattr(state_update["messages"], "content"):
                            print(f"\n[{node_name}] {state_update['messages'].content}\n")
                        else:
                            print(f"\n[{node_name}] Completed step.\n")
            
            # Update history with the agent's final state messages
            if final_state and "messages" in final_state:
                chat_history.append(final_state["messages"])
            
        except KeyboardInterrupt:
             print("\nExiting...")
             break
        except Exception as e:
             print(f"\n[Error] {str(e)}")