import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import time
from langchain_core.messages import HumanMessage

# Ensure ipl_agent is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ipl_agent.agent import run_agent

# ==========================================
# 1. CLEANUP OLD FILES
# ==========================================
def cleanup_old_files(days=2):
    import glob
    now = time.time()
    cutoff = now - (days * 86400)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for folder in ['query_results', 'plots']:
        dir_path = os.path.join(base_dir, folder)
        if os.path.exists(dir_path):
            for file_path in glob.glob(os.path.join(dir_path, '*')):
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass

# Run cleanup on app startup
cleanup_old_files(2)

# ==========================================
# 2. STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="IPL Analytics AI", page_icon="🏏", layout="wide")
st.title("🏏 IPL Analytics AI Agent")
st.markdown("Ask me anything about IPL history! I can write SQL, fetch data, and write Python to plot it.")

# ==========================================
# 2. SESSION STATE & MEMORY INITIALIZATION
# ==========================================
# UI History (what is displayed)
if "messages" not in st.session_state:
    st.session_state.messages = []
# Agent History (what is passed to LangGraph for memory)
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

# ==========================================
# 3. RENDER CHAT HISTORY
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render SQL if present
        if msg.get("sql_queries"):
            for sql in msg["sql_queries"]:
                with st.expander("🔍 View BigQuery SQL"):
                    st.code(sql, language="sql")
        
        # Render Plot code if present
        if msg.get("plot_code"):
            with st.expander("📊 View Python Plot Code"):
                st.code(msg["plot_code"], language="python")
                
        # Render the raw CSV data
        if msg.get("csv_path") and os.path.exists(msg["csv_path"]):
            import pandas as pd
            try:
                df = pd.read_csv(msg["csv_path"])
                st.dataframe(df, use_container_width=True)
            except Exception:
                pass
                
        # If this message has a saved plot, render it inline
        if msg.get("plot_path") and os.path.exists(msg["plot_path"]):
            with open(msg["plot_path"], 'r', encoding='utf-8') as f:
                components.html(f.read(), height=650, scrolling=True)

# ==========================================
# 4. CHAT EXECUTION LOGIC
# ==========================================
if prompt := st.chat_input("E.g., Which 5 teams hit the most sixes in 2024? Plot it as a bar chart."):
    
    # Display user prompt in the UI
    st.session_state.messages.append({"role": "user", "content": prompt, "plot_path": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Assistant Response
    with st.chat_message("assistant"):
        
        with st.status("Agent Workflow Initiated...", expanded=True) as status:
            st.write("🧠 Analyzing intent and executing tools...")
            
            # Execute the LangGraph Agent
            result = run_agent(prompt, st.session_state.agent_history)
            
            # Update internal agent history for multi-turn context
            st.session_state.agent_history.append(HumanMessage(content=prompt))
            if result.get("messages_update"):
                st.session_state.agent_history.append(result["messages_update"])
            
            # Cap history to prevent context explosion
            if len(st.session_state.agent_history) > 10:
                st.session_state.agent_history = st.session_state.agent_history[-10:]
                
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Stream the final text answer (typewriter effect)
        def stream_text():
            for word in result["answer"].split(" "):
                yield word + " "
                time.sleep(0.02)
                
        st.write_stream(stream_text)

        # Show SQL used
        if result.get("sql_queries"):
            for sql in result["sql_queries"]:
                with st.expander("🔍 View BigQuery SQL"):
                    st.code(sql, language="sql")

        # Show Python plot code used
        if result.get("plot_code"):
            with st.expander("📊 View Python Plot Code"):
                st.code(result["plot_code"], language="python")

        # Render the raw CSV data to avoid LLM hallucination
        csv_path = result.get("csv_path")
        if csv_path and os.path.exists(csv_path):
            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load raw data: {e}")

        # Render the Plotly chart if one was generated
        plot_path = result.get("plot_path")
        if plot_path and os.path.exists(plot_path):
            with open(plot_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=650, scrolling=True)

        # Save to UI History
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "plot_path": plot_path,
            "sql_queries": result.get("sql_queries", []),
            "plot_code": result.get("plot_code", ""),
            "csv_path": csv_path
        })
