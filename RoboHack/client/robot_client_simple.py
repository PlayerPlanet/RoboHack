# robot_client_VLA_ONLY.py

import threading
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# CRITICAL: Import patches BEFORE any lerobot imports
from RoboHack.client.camera_fix import patch_opencv_backend

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.camera import CameraConfig
from lerobot.robots.so100_follower import SO100FollowerConfig


SO101_PORT = "COM6"
SERVER_IP = "65.108.32.147"
SERVER_PORT = 8000
CAMERA_INDEX = 0


def main():
    while True:
        instruction = input("\nEnter task (or 'q' to quit): ")
        if instruction.lower() in ("q", "quit"):
            print("Exiting...")
            break

        print("Initializing client...")

        camera_cfg: dict[str, CameraConfig] = {
            "image": OpenCVCameraConfig(
                index_or_path=CAMERA_INDEX,
                width=640,
                height=480,
                fps=15,
            )
        }

        robot_cfg = SO100FollowerConfig(
            port=SO101_PORT,
            id="follower_so101",
            cameras=camera_cfg,
        )

        client_cfg = RobotClientConfig(
            robot=robot_cfg,
            server_address=f"{SERVER_IP}:{SERVER_PORT}",
            policy_device="cuda",
            policy_type="smolvla",  # Diffusion policy - flexible with action dims
            pretrained_name_or_path="lerobot/smolvla_base",  # Diffusion model
            chunk_size_threshold=0.7,
            actions_per_chunk=50,
        )

        # Create client (SO101 is already patched for Pi0.5 compatibility)
        client = RobotClient(client_cfg)

        print(f"Connecting to server at {client_cfg.server_address}...")

        if client.start():
            print("Connected to server!")
            action_receiver_thread = threading.Thread(
                target=client.receive_actions, daemon=True
            )
            action_receiver_thread.start()

            try:
                print(f"Executing: '{instruction}'. Press Ctrl+C to stop.")
                client.control_loop(instruction)

            except KeyboardInterrupt:
                print("\nStopping current task...")
            finally:
                client.stop()
                action_receiver_thread.join()
                print("Robot client shut down.")
        else:
            print("Failed to connect to the policy server.")

        print("\nTask finished. Ready for new task.")

    print("Goodbye!")


if __name__ == "__main__":
    main()
