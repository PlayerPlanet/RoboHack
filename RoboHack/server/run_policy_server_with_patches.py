#!/usr/bin/env python3
"""
Wrapper script to run the lerobot policy server with necessary patches applied.
This ensures the Pi0 transformers check bypass is loaded before any lerobot code.
"""

# Apply monkey-patches BEFORE any lerobot imports
import sys
import os

# Add RoboHack/client to path so we can import camera_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RoboHack', 'client'))

# Import patches - this auto-applies all fixes including Pi0 transformer check
import camera_fix

# Now we can safely import and run lerobot's policy server
from lerobot.async_inference.cli import main

if __name__ == "__main__":
    # Run the policy server CLI with all patches applied
    main()
