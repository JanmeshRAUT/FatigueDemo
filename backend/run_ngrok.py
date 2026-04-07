
import os
import sys
import time
import subprocess
import signal
import urllib.request
import json

# Configuration
NGROK_PORT = 5000
NGROK_DOMAIN = "nulliporous-carbolic-lianne.ngrok-free.dev"  # Your static domain

def check_ngrok_installed():
    """Check if ngrok is available in the system PATH."""
    try:
        subprocess.run(["ngrok", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def start_ngrok():
    """Start ngrok process."""
    if not check_ngrok_installed():
        print("❌ Error: 'ngrok' command not found. Please install ngrok and add it to your PATH.")
        sys.exit(1)

    print(f"🚀 Starting Ngrok on port {NGROK_PORT}...")
    
    try:
        # Let's try the static domain first
        print(f"Attempting to use static domain: {NGROK_DOMAIN}")
        process = subprocess.Popen(
            ["ngrok", "http", f"--domain={NGROK_DOMAIN}", str(NGROK_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(3)
        
        if process.poll() is not None:
             # Process exited immediately, likely error (e.g. domain invalid)
             _, stderr = process.communicate()
             print(f"⚠️ Failed to start with static domain: {stderr}")
             print("🔄 Falling back to random domain...")
             process = subprocess.Popen(
                ["ngrok", "http", str(NGROK_PORT)],
                stdout=subprocess.DEVNULL, # We'll inspect via API
                stderr=subprocess.DEVNULL
            )
             time.sleep(3)

        return process

    except Exception as e:
        print(f"❌ Failed to start ngrok: {e}")
        sys.exit(1)

def get_public_url():
    """Query local ngrok API to get the public URL."""
    retries = 10
    url = "http://localhost:4040/api/tunnels"
    
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    tunnels = data.get("tunnels", [])
                    if tunnels:
                        return tunnels[0]["public_url"]
        except Exception:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    return None

def main():
    process = start_ngrok()
    
    print("\n🔍 resolving public URL...")
    public_url = get_public_url()
    
    if public_url:
        print(f"\n\n✅ Ngrok is RUNNING!")
        print(f"🌐 Public URL: {public_url}")
        print("-" * 50)
        print(f"👉 COPY this URL and update your Vercel Environment Variables:")
        print(f"   Key:   REACT_APP_API_URL")
        print(f"   Value: {public_url}")
        print("-" * 50)
        print("\nPress Ctrl+C to stop ngrok.")
        
        try:
            # Keep script running
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping ngrok...")
            process.terminate()
    else:
        print("\n❌ Could not resolve public URL. Is ngrok running?")
        process.terminate()

if __name__ == "__main__":
    main()
