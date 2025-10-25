# robot_client_VLA_continuous.py

import threading
import time
import numpy as np
import cv2
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
# Import the actual robot class, not just the config
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

SO101_PORT = "COM6"
SERVER_IP = "65.108.32.147"
SERVER_PORT = 8000
CAMERA_INDEX = 1


def main():
    # --- PHASE 1: SETUP (Done ONCE) ---
    print("Initializing robot and client configurations...")

    camera_cfg: dict[str, CameraConfig] = {
        "primary": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            width=640,
            height=480,
            fps=15
        )
    }

    # 2. Create robot config
    robot_cfg = SO101FollowerConfig(
        port=SO101_PORT,
        id="follower_so101",
        cameras=camera_cfg
    )

    # 3. Create client configuration
    client_cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address=f"{SERVER_IP}:{SERVER_PORT}",
        policy_device="cuda",
        policy_type="smolvla",  # Diffusion policy - flexible with action dims
        pretrained_name_or_path="lerobot/smolvla_base",  # Diffusion model
        chunk_size_threshold=0.7,
        actions_per_chunk=50,
    )

    # 4. Create and start client (Done ONCE)
    client = RobotClient(client_cfg)

    print(f"Connecting to server at {client_cfg.server_address}...")

    if not client.start():
        print("Failed to connect to the policy server.")
        return

    print("Connected to server!")
    # Start the background thread that receives actions (Done ONCE)
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    # Create a window for the live feed
    window_name = 'VLA Live Feed'
    cv2.namedWindow(window_name)

    # Get the robot object (created by client.start())
    robot = client.robot

    # --- PHASE 2: Main Instruction Loop ---
    try:
        while True:
            instruction = input("\nEnter task (or 'q' to quit): ")
            if instruction.lower() in ('q', 'quit'):
                print("Exiting...")
                break  # Exit the instruction loop

            print(f"Executing: '{instruction}'. Press ESC in window to stop.")

            # --- PHASE 3: Task Execution Loop (for one task) ---


            obs = robot.get_observation()
            obs_with_instruction = obs.copy()
            obs_with_instruction["instruction"] = instruction
            info = {}

            client.send_observation(obs_with_instruction)

            task_running = True
            while task_running:
                # Get an action from the server (this waits)
                action = client.action_queue.get()
                if action is None:
                    print("Task finished (received None action).")
                    task_running = False
                    break

                # Apply the action to the robot
                obs, reward, terminated, truncated, info = robot.step(action)

                # Send the new observation back to the server
                client.send_observation(obs)

                # --- Display the live image ---
                image_rgb = obs["image"]
                image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, image_display)

                if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
                    print("Task cancelled by user.")
                    task_running = False
                    break

                if terminated or truncated:
                    print("Task finished (episode ended).")
                    task_running = False
                    break

            print("\nTask complete. Ready for new task.")


    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cv2.destroyAllWindows()
        client.stop()
        action_receiver_thread.join()
        print("Robot client shut down.")
        print("Goodbye!")


if __name__ == "__main__":
    main()