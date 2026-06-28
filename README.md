# 🏏 IPL Analytics GenAI Agent

![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)

> **Live Application**: [https://stumped-ai.onrender.com/]

## Overview
The IPL Analytics AI Agent is a powerful, LangGraph-powered conversational assistant that allows you to ask complex questions about the Indian Premier League (IPL) and receive dynamic, accurate data analysis and visual plots. 

Instead of relying on hallucinated LLM knowledge, this agent operates on a **Retrieval-Augmented Generation (RAG) + Tool Execution** architecture. It intelligently routes user intent, writes SQL queries to pull actual statistics from a Google Cloud BigQuery database, and writes Python Plotly code to generate interactive visualizations of the results.

## Features
- **Data Agent (BigQuery Integration)**: Automatically writes and executes SQL queries against a robust dataset containing ball-by-ball IPL statistics.
- **Plot Agent (Dynamic Visualization)**: Parses CSV results returned by the Data Agent and generates interactive `plotly` charts.
- **Trivia Agent (Search Integration)**: Routes general queries to Google Search and Cricinfo for player profiles and external knowledge.
- **Stateful UI**: A Streamlit frontend that provides transparency by displaying the actual SQL queries and Python code executed by the sub-agents.
- **Anomaly Handling**: Automatically detects and handles edge cases like Super Overs in standard aggregations.

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python, LangChain, LangGraph
- **LLMs**: NVIDIA NIM (Llama 3 / Gemma / Qwen)
- **Database**: Google Cloud BigQuery
- **Deployment**: Render

## Setup and Installation

### 1. Prerequisites
- Python 3.10+
- Google Cloud Service Account with `BigQuery Data Viewer` and `BigQuery Job User` roles.
- NVIDIA NIM API Key.

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
NVIDIA_NIM_API_KEY=your_nvidia_key_here
GCP_PROJECT_ID=your_gcp_project_id
```

### 3. Local Credentials
Create a `service-account` folder in the root directory and place your GCP JSON key inside it as `service-account.json`. The app will automatically detect and load it.

### 4. Run the Application
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deployment (Render)
This application is fully configured for deployment on Render.
1. Connect this repository to a Render Web Service.
2. Set Build Command to `pip install -r requirements.txt` and Start Command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Add your `.env` variables in the environment settings.
4. Upload your `service-account.json` via Render's **Secret Files** feature and point `GOOGLE_APPLICATION_CREDENTIALS` to it.
