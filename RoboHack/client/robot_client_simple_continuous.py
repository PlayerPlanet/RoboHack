# robot_client_VLA_continuous.py

import threading
import time
import numpy as np
import cv2
import sys
from pathlib import Path
from typing import Any
from lerobot.async_inference.helpers import TimedAction


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

# Helper to create a zeroed action dict using the robot's declared action_features
def _action_dict_for(robot) -> dict[str, float]:
    """Return a dict with keys '<motor>.pos' -> 0.0 using robot.action_features ordering.

    This avoids accessing `action_space` and helps pylance resolve attributes.
    """
    keys = []
    if hasattr(robot, "action_features"):
        try:
            keys = list(robot.action_features.keys())
        except Exception:
            keys = []

    if not keys:
        # conservative fallback order matching SO101Follower expected motors
        keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

    return {k: 0.0 for k in keys}

SO101_PORT = "COM6"
SERVER_IP = "65.108.32.147"
SERVER_PORT = 8000
CAMERA_INDEX = 0

home = {
    "shoulder_pan.pos": 0.1,
    "shoulder_lift.pos": -0.3,
    "elbow_flex.pos": 0.2,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.5,
    "gripper.pos": 0.0,
}

class FaceTrackerThread(threading.Thread):

    # Updated __init__ to accept window_name
    def __init__(self, env: Any, stop_event, window_name: str):
        super().__init__()
        self.env = env
        self.stop_event = stop_event
        self.window_name = window_name  # Store window name
        self.daemon = True
        # Use public mediapipe API objects (pylance-friendly).
        # Use type-ignore to silence pylance attribute issues when stubs differ locally.
        try:
            self.mp_face_detection = mp.solutions.face_detection  # type: ignore[attr-defined]
            self.mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
        except Exception:
            self.mp_face_detection = None
            self.mp_drawing = None

        self.P_GAIN_YAW = -0.1
        self.P_GAIN_Z = -0.1

    def run(self):
        print("Starting idle facetracking routine")

        # Window is now created and managed by main()
        # cv2.namedWindow(self.window_name)

        try:
            while not self.stop_event.is_set():
                # Acquire fresh observation each loop using SO101Follower API
                try:
                    obs = self.env.get_observation()
                except Exception:
                    # If we can't get observations, stop the idle routine
                    self.stop_event.set()
                    break

                image_rgb = obs.get("images", {}).get("primary")
                if image_rgb is None:
                    # no image available, sleep and continue
                    time.sleep(0.02)
                    continue

                image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                try:
                    image_rgb.flags.setflags(write=0)
                except Exception:
                    pass

                # Run face detection using a short-lived detector (keeps code simple)
                FaceDetection = getattr(self.mp_face_detection, "FaceDetection", None)
                if FaceDetection is None:
                    # mediapipe not available in this environment; skip detection but continue showing image
                    time.sleep(0.02)
                    continue

                with FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                    results = face_detection.process(image_rgb)

                # Build an action dict keyed by '<motor>.pos' using robot's action_features order
                action_dict = _action_dict_for(self.env)
                detections = getattr(results, "detections", None)
                if detections:
                    for detection in detections:
                        # draw each detection onto display (guard drawing util)
                        if self.mp_drawing is not None:
                            try:
                                self.mp_drawing.draw_detection(image_display, detection)
                            except Exception:
                                pass

                    first_detection = detections[0]
                    bbox = first_detection.location_data.relative_bounding_box

                    cx = bbox.xmin + bbox.width / 2
                    cy = bbox.ymin + bbox.height / 2

                    error_x = cx - 0.5
                    error_y = cy - 0.5

                    # assign gains to robust keys if present
                    keys = list(action_dict.keys())
                    if len(keys) > 5:
                        action_dict[keys[5]] = float(self.P_GAIN_YAW * error_x)
                    if len(keys) > 2:
                        action_dict[keys[2]] = float(self.P_GAIN_Z * error_y)

                    # clip values conservatively
                    for k in action_dict:
                        action_dict[k] = float(np.clip(action_dict[k], -1.0, 1.0))

                # show image
                cv2.imshow(self.window_name, image_display)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    self.stop_event.set()
                    break

                if self.stop_event.is_set():
                    break

                # Send the action to the follower using its send_action API
                try:
                    if hasattr(self.env, "send_action"):
                        _ = self.env.send_action(action_dict)
                except Exception:
                    # on error, attempt to stop the routine
                    self.stop_event.set()
                    break

                # handle any termination signal returned in observations if applicable
                # (SO101Follower uses separate reset() so we don't expect a terminated flag here)
                time.sleep(0.02)

        finally:
            print("Stopping idle process")
            # Let main() handle destroying the window
            # cv2.destroyAllWindows()
            if hasattr(self.env, "reset"):
                try:
                    self.env.send_action(home)
                except Exception:
                    pass


def main():
    # --- PHASE 1: SETUP (Done ONCE) ---
    print("Initializing robot and client configurations...")

    camera_cfg: dict[str, CameraConfig] = {
            "handeye": OpenCVCameraConfig(
                index_or_path=CAMERA_INDEX,
                width=640,
                height=360,
                fps=30,
            ),
            "fixed": OpenCVCameraConfig(
                index_or_path=1,
                width=640,
                height=360,
                fps=30,
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
        policy_type="smolvla",  # Diffusion policy - flexible with action dims
        pretrained_name_or_path="y1y2y3/so101_test8_smolvla200k_augmented100",  # Diffusion model
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
            # Run the control loop but ensure it exits after 15s by scheduling a safe-home injection.
            # We must NOT call client.stop() (it disconnects hardware). Instead we enqueue a
            # TimedAction containing the home pose and then set shutdown_event after the
            # client has reported it performed that action.
            timeout_s = 15.0

            def _inject_home_and_shutdown():
                try:
                    import torch

                    # compute next timestep (make it strictly newer than latest_action)
                    with client.latest_action_lock:
                        next_timestep = client.latest_action + 1

                    # build tensor in the order of robot.action_features
                    keys = list(client.robot.action_features)
                    values = [float(home.get(k, 0.0)) for k in keys]
                    action_tensor = torch.tensor(values, dtype=torch.float32)

                    ta = TimedAction(timestamp=time.time(), timestep=next_timestep, action=action_tensor)

                    # put into the client's action queue
                    with client.action_queue_lock:
                        client.action_queue.put(ta)

                    # wait for the client to report it performed the action
                    start = time.time()
                    grace = 5.0
                    while time.time() - start < grace:
                        with client.latest_action_lock:
                            if client.latest_action >= next_timestep:
                                break
                        time.sleep(0.01)

                    # now request the control loop to exit by setting the shutdown event
                    client.shutdown_event.set()

                except Exception:
                    # If anything fails, fall back to setting the shutdown flag only
                    try:
                        client.shutdown_event.set()
                    except Exception:
                        pass

            stop_timer = threading.Timer(timeout_s, _inject_home_and_shutdown)
            stop_timer.start()
            try:
                client.control_loop(instruction)
            except KeyboardInterrupt:
                print("\nTask cancelled by user.")
                # request shutdown if not already requested
                if client.running:
                    client.stop()
            finally:
                # cancel timer if control_loop finished earlier
                stop_timer.cancel()
                # Ensure robot is in a safe home position when the task ends.
                try:
                    if hasattr(robot, "send_action"):
                        robot.send_action(home)
                except Exception:
                    pass

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