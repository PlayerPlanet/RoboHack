# robot_client_VLA_continuous.py

import threading
import time
import numpy as np
import cv2
import sys
from pathlib import Path


# --- Assuming camera_fix is needed ---
# CRITICAL: Import patches BEFORE any lerobot imports
try:
    from RoboHack.client.camera_fix import patch_opencv_backend

    print("Camera patch imported.")
except ImportError:
    print("Could not import RoboHack camera patch. Proceeding with standard lerobot.")
# --- End of custom setup ---

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
# Import the actual robot class
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import conversation_hub  # Assuming this is your module for getting instructions
import mediapipe as mp
# Added missing import for drawing
import mediapipe.python.solutions.drawing_utils as mp_drawing

SO101_PORT = "COM6"
SERVER_IP = "65.108.32.147"
SERVER_PORT = 8000
CAMERA_INDEX = 1


class FaceTrackerThread(threading.Thread):

    # Updated __init__ to accept window_name
    def __init__(self, env, stop_event, window_name):
        super().__init__()
        self.env = env
        self.stop_event = stop_event
        self.window_name = window_name  # Store window name
        self.daemon = True

        self.mp_face_detection = mp.solutions.face_detection

        self.P_GAIN_YAW = -0.1
        self.P_GAIN_Z = -0.1

    def run(self):
        print("Starting idle facetracking routine")

        # Window is now created and managed by main()
        # cv2.namedWindow(self.window_name)

        try:
            with self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
            ) as face_detection:

                # Use correct method get_observation()
                obs = self.env.get_observation()

                while not self.stop_event.is_set():
                    # Get image from the correct key "primary"
                    image_rgb = obs["images"]["primary"]

                    image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    # Make image non-writable for mediapipe processing
                    image_rgb.flags.setflags(write=0)
                    results = face_detection.process(image_rgb)

                    action = np.zeros_like(self.env.action_space.sample())

                    if results.detections:
                        for detection in results.detections:
                            mp_drawing.draw_detection(image_display, detection)

                        first_detection = results.detections[0]
                        bbox = first_detection.location_data.relative_bounding_box

                        cx = bbox.xmin + bbox.width / 2
                        cy = bbox.ymin + bbox.height / 2

                        error_x = cx - 0.5
                        error_y = cy - 0.5

                        action[5] = self.P_GAIN_YAW * error_x
                        action[2] = self.P_GAIN_Z * error_y

                        action = np.clip(action, -1.0, 1.0)

                    # Use the shared window name
                    cv2.imshow(self.window_name, image_display)

                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        self.stop_event.set()
                        break

                    # Check stop_event again before stepping
                    if self.stop_event.is_set():
                        break

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    if terminated or truncated:
                        # Use correct method reset()
                        obs = self.env.reset()

                    time.sleep(0.02)

        finally:
            print("Stopping idle process")
            # Let main() handle destroying the window
            # cv2.destroyAllWindows()


def main():
    # --- PHASE 1: SETUP (Done ONCE) ---
    print("Initializing robot and client configurations...")

    camera_cfg: dict[str, CameraConfig] = {
        # Camera is named "primary"
        "primary": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            width=640,
            height=480,
            fps=15
        )
    }

    # 2. Create robot config (Added action_scale for safety)
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
        policy_type="smolvla",
        pretrained_name_or_path="lerobot/smolvla_base",
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

    # Create a window for the live feed (Done ONCE)
    window_name = 'Robot View'
    cv2.namedWindow(window_name)

    # Get the robot object (created by client.start())
    robot = client.robot

    # --- PHASE 2: Main Instruction Loop ---
    try:
        while True:
            # --- START IDLE MODE ---
            print("Starting idle mode. Waiting for instruction...")
            stop_event = threading.Event()
            face_tracker = FaceTrackerThread(robot, stop_event, window_name)
            face_tracker.start()

            # This blocks until an instruction is received
            instruction = conversation_hub.main_loop()

            # --- STOP IDLE MODE ---
            print("Instruction received. Stopping idle mode.")
            stop_event.set()
            face_tracker.join()  # Wait for thread to finish cleanly

            if instruction.lower() in ('q', 'quit'):
                print("Exiting...")
                break  # Exit the instruction loop

            print(f"Executing: '{instruction}'. Press ESC in window to stop.")

            # --- PHASE 3: Task Execution Loop (for one task) ---

            # Reset robot to a known state before starting VLA task
            obs = robot.reset()
            obs_with_instruction = obs.copy()
            # The VLA model expects the "primary" camera
            obs_with_instruction["images"] = {"primary": obs["images"]["primary"]}
            obs_with_instruction["instruction"] = instruction
            info = {}

            client.send_observation(obs_with_instruction)

            task_running = True
            start = time.time()
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
                # Only send the "primary" image
                obs_to_send = obs.copy()
                obs_to_send["images"] = {"primary": obs["images"]["primary"]}
                client.send_observation(obs_to_send)

                # --- Display the live image ---
                # Get image from the correct key
                image_rgb = obs["images"]["primary"]
                image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, image_display)

                if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
                    print("Task cancelled by user.")
                    task_running = False
                    break

                if time.time() - start > 15:  # 15 second timeout
                    print("Time's up!")
                    task_running = False
                    break

                if terminated or truncated:
                    print("Task finished (episode ended).")
                    task_running = False
                    break

            # Stop the robot (send zero action) and reset
            print("Task complete. Resetting robot.")
            robot.step(np.zeros_like(robot.action_space.sample()))
            robot.reset()

            print("\nReady for new task.")


    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Clean up window and client
        cv2.destroyAllWindows()
        client.stop()
        if action_receiver_thread.is_alive():
            action_receiver_thread.join()
        print("Robot client shut down.")
        print("Goodbye!")


if __name__ == "__main__":
    main()