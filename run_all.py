# run_all.py
import subprocess
import json
import time
import sys
import os

def run_all():
    processes = []
    config_path = "server_config.json"
    fastapi_host = "127.0.0.1"
    fastapi_port = 8000
    
    # Set the base URL for the Streamlit app
    os.environ["FASTAPI_BASE_URL"] = f"http://{fastapi_host}:{fastapi_port}"

    print("--- MCP Universal Application Launcher ---")

    try:
        with open(config_path, "r") as f:
            mcp_servers = json.load(f).get("mcpServers", {})
    except FileNotFoundError:
        print(f"Warning: {config_path} not found.")
        mcp_servers = {}

    print("\nLaunching MCP servers...")
    for name, server_config in mcp_servers.items():
        command = [server_config.get("command")] + server_config.get("args", [])
        try:
            proc = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
            processes.append((f"MCP Server '{name}'", proc))
            print(f"  - Started server '{name}' (PID: {proc.pid})")
        except Exception as e:
            print(f"  - ERROR starting '{name}': {e}")
    
    time.sleep(3)

    print("\nLaunching FastAPI backend...")
    try:
        fastapi_command = ["uvicorn", "mcp_chatbot:app", "--host", fastapi_host, "--port", str(fastapi_port)]
        proc = subprocess.Popen(fastapi_command, stdout=sys.stdout, stderr=sys.stderr)
        processes.append(("FastAPI Backend", proc))
        print(f"  - Started FastAPI backend (PID: {proc.pid}) at {os.environ['FASTAPI_BASE_URL']}")
    except Exception as e:
        print(f"  - FATAL ERROR starting FastAPI backend: {e}")
        # Clean up and exit if backend fails
        for name, proc in processes: proc.terminate()
        return

    time.sleep(3)

    print("\nLaunching Streamlit frontend...")
    try:
        # --- THIS IS THE FIX ---
        # Create a copy of the current environment and add our special variable
        st_env = os.environ.copy()
        st_env["IS_STREAMLIT_CONTEXT"] = "true"
        
        streamlit_command = ["streamlit", "run", "app.py"]
        # Pass the modified environment to the Streamlit process
        proc = subprocess.Popen(streamlit_command, stdout=sys.stdout, stderr=sys.stderr, env=st_env)
        # --- END OF FIX ---
        processes.append(("Streamlit Frontend", proc))
        print("  - Streamlit is running. Access it in your browser.")
        proc.wait()
    except Exception as e:
        print(f"\nAn error occurred while running Streamlit: {e}")
    finally:
        print("\nShutting down all background processes...")
        for name, proc in reversed(processes):
            try:
                proc.terminate()
                proc.wait(timeout=2)
                print(f"  - Terminated {name} (PID: {proc.pid})")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  - Force-killed {name} (PID: {proc.pid})")
            except Exception: pass
        print("Cleanup complete.")

if __name__ == "__main__":
    run_all()