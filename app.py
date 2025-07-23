# app.py
import streamlit as st
from mcp_chatbot import ServiceType
import os
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="MCP Universal Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Available Models ---
AVAILABLE_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
}
# --- FastAPI Backend Configuration ---
# The launcher will set this environment variable for us.
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

# ---
# --- Use @st.cache_data to fetch server capabilities once ---
# ---
@st.cache_data(show_spinner="Connecting to MCP Backend...")
def get_server_capabilities():
    """Fetches the lists of tools and resources from the FastAPI backend."""
    try:
        tools_response = requests.get(f"{FASTAPI_BASE_URL}/tools")
        tools_response.raise_for_status()
        tools = tools_response.json().get("tools", [])
        
        resources_response = requests.get(f"{FASTAPI_BASE_URL}/resources")
        resources_response.raise_for_status()
        resources = resources_response.json()

        prompts_response = requests.get(f"{FASTAPI_BASE_URL}/prompts")
        prompts_response.raise_for_status()
        prompts = prompts_response.json()

        return tools, resources, prompts
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the MCP backend at {FASTAPI_BASE_URL}. Is it running? Error: {e}")
        return [], {}, {}

# --- Sidebar (Simplified for API model) ---

# --- Helper Functions (for IDE tab) ---
def get_server_files():
    files = ["server_config.json"]
    try:
        with open("server_config.json", "r") as f: config = json.load(f)
        for server in config.get("mcpServers", {}).values():
            script_path = server.get("args", [None])[0]
            if script_path and os.path.exists(script_path): files.append(script_path)
    except FileNotFoundError: st.sidebar.error("server_config.json not found!")
    return list(set(files))

# --- Collapsible Sidebar for Configuration ---
with st.sidebar.expander("⚙️ Configuration", expanded=True):
    selected_service_str = st.selectbox("Select AI Service", options=list(AVAILABLE_MODELS.keys()))
    selected_service = ServiceType(selected_service_str)
    selected_model = st.selectbox("Select Model", options=AVAILABLE_MODELS[selected_service_str])


# --- Main App Logic ---
tools, resources, prompts = get_server_capabilities()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "🛠️ IDE", "📚 Available Commands"])

# --- Chatbot Tab ---
with tab1:
    st.header("MCP Universal Chatbot")

    # --- Responsive Sticky Chat Input ---
    st.markdown(
        """
        <style>
        /* Make the main container take full height */
        .main .block-container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding-bottom: 6rem; /* Space for sticky input */
        }
        /* Sticky chat input at the bottom, responsive to sidebar */
        .stChatInput {
            position: fixed !important;
            left: var(--stSidebar-width, 0px);
            right: 0;
            bottom: 0;
            width: calc(100vw - var(--stSidebar-width, 0px)) !important;
            background: var(--background-color);
            z-index: 9999;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.04);
            padding-bottom: 1.5rem;
            transition: left 0.3s, width 0.3s;
        }
        /* Adjust for sidebar open/close */
        [data-testid="stSidebar"][aria-expanded="true"] ~ .main .stChatInput {
            left: var(--stSidebar-width, 320px);
            width: calc(100vw - 320px) !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] ~ .main .stChatInput {
            left: 0;
            width: 100vw !important;
        }
        /* Ensure chat messages scroll above the input */
        .chat-scroll-area {
            flex: 1 1 auto;
            overflow-y: auto;
            margin-bottom: 6rem; /* Height of sticky input */
            padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Chat Display Area (scrollable, always above input) ---
    chat_display = st.container()
    with chat_display:
        st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Sticky chat input at the bottom ---
    prompt = st.chat_input("Ask your question...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    api_payload = {
                        "query": prompt,
                        "message_history": st.session_state.chat_history
                    }
                    response = requests.post(f"{FASTAPI_BASE_URL}/chat", json=api_payload)
                    response.raise_for_status()
                    assistant_response_data = response.json()
                    if "error" in assistant_response_data:
                        final_response = f"An error occurred: {assistant_response_data['error']}"
                        st.error(final_response)
                    else:
                        final_response = assistant_response_data.get("response", "No response received.")
                        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                        st.markdown(final_response)
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to get response from backend: {e}")

# --- Available Commands Tab ---
with tab2:
    st.header("🔧 Integrated MCP Server IDE")
    st.info("Edit your server configuration or scripts. Restart the launcher for changes to take effect.")
    server_files = get_server_files()
    selected_file = st.selectbox("Select a file to edit", options=server_files)
    if selected_file:
        with open(selected_file, "r") as f: file_content = f.read()
        from streamlit_ace import st_ace
        new_content = st_ace(value=file_content, language="python" if selected_file.endswith(".py") else "json", theme="monokai", keybinding="vscode", height=500, auto_update=True)
        if st.button("Save Changes"):
            with open(selected_file, "w") as f: f.write(new_content)
            st.success(f"Successfully saved {selected_file}!")

with tab3:
    st.header("📖 Discovered MCP Commands")
    st.info("These commands were discovered from the MCP backend.")
    
    if tools:
        with st.expander("🛠️ Available Tools (Used automatically by the AI)"):
            st.json(tools)
    
    if resources:
        with st.expander("📦 Available Resources (Use with `@<name>` or just `<name>`)"):
            st.json(resources)

    if prompts:
        with st.expander("📝 Available Prompts (Use with `/prompt <name>` or select in chat)"):
            st.json(prompts)