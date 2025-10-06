# run.py
import subprocess
import sys
import os

script_path = os.path.abspath("votxt.py")

# Only launch if default port is free
import socket
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

port = 8501
if not is_port_in_use(port):
    # Use Popen instead of run() to avoid blocking issues
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path, "--server.port", str(port)])
    print(f"Streamlit launched at http://localhost:{port}")
else:
    print(f"Streamlit already running at http://localhost:{port}")
