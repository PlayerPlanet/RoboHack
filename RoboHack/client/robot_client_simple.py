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

"""
# --- IDLE MODE COMMENTED OUT ---

class FaceTrackerThread(threading.Thread):

    def __init__(self, env, stop_event):
        super().__init__()
        self.env = env
        self.stop_event = stop_event
        self.daemon = True

        self.mp_face_detection = mp.solutions.face_detection

        self.P_GAIN_YAW = -0.1
        self.P_GAIN_Z = -0.1

    def run(self):
        print("Starting idle facetracking routine")

        window_name = 'Camera view'
        cv2.namedWindow(window_name)

        try:
            with self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
            ) as face_detection:

                obs, info = self.env.reset()

                while not self.stop_event.is_set():
                    image_rgb = obs["image"]

                    image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
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

                    cv2.imshow(window_name, image_display)

                    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
                        self.stop_event.set()
                        break

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    if terminated or truncated:
                        obs, info = self.env.reset()

                    time.sleep(0.02)  # ~50 Hz loop

        finally:
            print("Stopping idle process")
            cv2.destroyAllWindows()
"""


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
