# app.py
import streamlit as st
import asyncio
from mcp_chatbot import MCP_ChatBot, ServiceType
import os
import json
from threading import Thread

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

# ---
# --- DEFINITIVE FIX: Run the asyncio event loop in a dedicated background thread ---
# ---
@st.cache_resource
def get_async_loop() -> asyncio.AbstractEventLoop:
    """
    This is the core of the solution. It creates a new asyncio event loop
    and runs it in a separate, daemonized thread. Streamlit's @st.cache_resource
    ensures this function is run only ONCE for the entire session.
    """
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    print("Started background asyncio event loop.")
    return loop

st_event_loop = get_async_loop()

# ---
# --- Use @st.cache_resource for the expensive, one-time initialization ---
# ---
@st.cache_resource(show_spinner="Initializing chatbot engine for the first time...")
def get_chatbot_instance(service_type: ServiceType, model: str) -> MCP_ChatBot:
    """
    This function is run ONLY ONCE. It schedules the async initialization
    on the background event loop and safely blocks to wait for the result.
    This is safe because the async code is running on a different thread.
    """
    async def initialize_async():
        chatbot = MCP_ChatBot(service_type=service_type, model=model)
        await chatbot.connect_to_servers()
        return chatbot

    # Schedule the coroutine on the background loop and wait for the result.
    future = asyncio.run_coroutine_threadsafe(initialize_async(), st_event_loop)
    return future.result()


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

# --- Sidebar for Configuration ---
st.sidebar.title("Configuration")
selected_service_str = st.sidebar.selectbox("Select AI Service", options=list(AVAILABLE_MODELS.keys()))
selected_service = ServiceType(selected_service_str)
selected_model = st.sidebar.selectbox("Select Model", options=AVAILABLE_MODELS[selected_service_str])


# --- Main App Logic ---

# Get the singleton chatbot instance. Streamlit handles the one-time execution and caching.
chatbot = get_chatbot_instance(selected_service, selected_model)

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "🛠️ IDE", "📚 Available Commands"])

# --- Chatbot Tab ---
with tab1:
    st.header("MCP Universal Chatbot")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use the same thread-safe method to run the 'invoke' coroutine
                # on the background event loop. This will not deadlock.
                future = asyncio.run_coroutine_threadsafe(
                    chatbot.invoke(prompt, st.session_state.chat_history), st_event_loop
                )
                final_response, updated_history = future.result()
                st.markdown(final_response)
        
        st.session_state.chat_history = updated_history
        st.rerun()

# --- IDE and Commands Tabs ---
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
    if chatbot.tool_to_session:
        with st.expander("🛠️ Available Tools (Used automatically by the AI)"): st.json(list(chatbot.tool_to_session.keys()))
    if chatbot.available_prompts:
        with st.expander("📝 Available Prompts (Use with `/prompt <name> <query>`)"): st.json(list(chatbot.available_prompts.keys()))
    if chatbot.static_resources:
        with st.expander("📦 Available Static Resources (Use with `@<name>` or just `<name>`)"): st.json(list(chatbot.static_resources.keys()))