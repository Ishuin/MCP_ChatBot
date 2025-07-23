# run_all.py
import subprocess
import json
import time
import sys

def run_all():
    """
    Reads server_config.json, launches all MCP servers as background processes,
    and then starts the Streamlit frontend.
    """
    server_processes = []
    config_path = "server_config.json"

    print("--- MCP Universal Chatbot Launcher ---")

    # 1. Read the server configuration
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        mcp_servers = config.get("mcpServers", {})
        if not mcp_servers:
            print("Warning: No MCP servers found in server_config.json.")
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Cannot start MCP servers.")
        mcp_servers = {}
    except json.JSONDecodeError:
        print(f"Error: Could not parse {config_path}. Please check for syntax errors.")
        return

    # 2. Launch each MCP server as a background process
    print("\nLaunching MCP servers...")
    for name, server_config in mcp_servers.items():
        command = server_config.get("command")
        args = server_config.get("args", [])
        
        if not command or not args:
            print(f"Skipping server '{name}': 'command' or 'args' not defined.")
            continue
            
        full_command = [command] + args
        try:
            # We use Popen to run the process in the background
            process = subprocess.Popen(full_command, stdout=sys.stdout, stderr=sys.stderr)
            server_processes.append(process)
            print(f"  - Started server '{name}' (PID: {process.pid})")
        except FileNotFoundError:
            print(f"  - ERROR: Could not start server '{name}'. Command '{command}' not found.")
        except Exception as e:
            print(f"  - ERROR: Could not start server '{name}'. Details: {e}")
    
    # Give servers a moment to initialize
    time.sleep(3)

    # 3. Launch the Streamlit frontend
    print("\nLaunching Streamlit frontend...")
    streamlit_process = None
    try:
        streamlit_process = subprocess.Popen(["streamlit", "run", "app.py"], stdout=sys.stdout, stderr=sys.stderr)
        print("  - Streamlit is running. Access it in your browser (usually http://localhost:8501)")
        # Wait for the Streamlit process to complete (which it won't until you close it)
        streamlit_process.wait()

    except FileNotFoundError:
        print("\nFATAL ERROR: `streamlit` command not found.")
        print("Please run `pip install streamlit` to install it.")
    except Exception as e:
        print(f"\nAn error occurred while running Streamlit: {e}")
    finally:
        # 4. Cleanup: Terminate all background servers when Streamlit is closed
        print("\nStreamlit has been closed. Shutting down all background MCP servers...")
        if streamlit_process:
            streamlit_process.terminate()
        for process in server_processes:
            try:
                process.terminate()
                print(f"  - Terminated server with PID: {process.pid}")
            except Exception:
                pass # Process might have already terminated
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    run_all()