#!/usr/bin/env python3
"""
Wrapper script to run the lerobot policy server with necessary patches applied.
This ensures the Pi0 transformers check bypass is loaded before any lerobot code.
"""

# Apply monkey-patches BEFORE any lerobot imports
import sys
import os
import argparse
# Add RoboHack/client to path so we can import camera_fix (file is in RoboHack/server)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'client'))

# Import patches - this auto-applies all fixes including Pi0 transformer check
try:
    # When package is installed/available
    import RoboHack.client.camera_fix   # type: ignore
except Exception:
    try:
        # When running from source, import the local module directly
        import camera_fix  # type: ignore
    except Exception:
        print("⚠️  Warning: camera_fix could not be imported; patches may not be applied")

# Now we can safely import and run lerobot's policy server
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve





def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lerobot policy server with local patches applied. Unknown args are ignored.")
    parser.add_argument("--host", default="localhost", help="Host/IP to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port for the policy server")
    # Use parse_known_args so callers (e.g. Docker CMD) can pass extra positional/subcommand args
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = PolicyServerConfig(host=args.host, port=args.port)
    serve(config)
