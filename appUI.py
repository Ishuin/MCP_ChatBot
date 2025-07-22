import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Adjust if hosted elsewhere

st.title("🤖 MCP AI ChatBot Interface")

# --- Load available prompts and tools ---
@st.cache_data
def fetch_prompts():
    try:
        resp = requests.get(f"{API_URL}/prompts")
        return resp.json().get("prompts", [])
    except:
        return []

@st.cache_data
def fetch_tools():
    try:
        resp = requests.get(f"{API_URL}/tools")
        return resp.json().get("tools", [])
    except:
        return []

@st.cache_data
def fetch_resources():
    try:
        resp = requests.get(f"{API_URL}/resources")
        return resp.json()
    except:
        return {}

prompts = fetch_prompts()
resources = fetch_resources()
tools = fetch_tools()

# --- User input ---
query = st.text_area("Enter your query", height=100)

prompt_name = st.selectbox("Select a prompt (optional)", [""] + prompts)
resource_uri = st.selectbox("Select a resource (optional)", [""] + resources.get("static", []))

if st.button("Ask"):
    payload = {
        "query": query,
        "prompt_name": prompt_name or None,
        "resource_uri": resource_uri or None
    }

    with st.spinner("Asking the assistant..."):
        try:
            response = requests.post(f"{API_URL}/chat", json=payload)
            if response.status_code == 200:
                st.markdown("### 💬 Response")
                st.markdown(f"```\n{response.json().get('response')}\n```")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to reach API: {e}")
