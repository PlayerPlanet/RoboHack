# robot_client.py

import threading
import time
import numpy as np
import cv2
import cv2
from lerobot.hardware.so101 import SO101Env
from lerobot.async_inference.robot_client import RobotClient


SO101_PORT = "COM6"  #USB PORT
SERVER_IP = "65.108.32.147"
SERVER_PORT = 8000


class FaceTrackerThread(threading.Thread):

    def __init__(self, env, stop_event, camera_index=0):
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

                    if cv2.waitKey(1) & 0xFF == 27:  
                        self.stop_event.set()
                        break

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    if terminated or truncated:
                        obs, info = self.env.reset()

                    time.sleep(0.02) 

        finally:
            print("Stopping idle process")
            cv2.destroyAllWindows()


def main():
    print("Connecting to robot... ")

    try:
        env = SO101Env(port=SO101_PORT, camera_index=CAMERA_INDEX)
    except Exception as e:
        print(f"Failed to connect on {SO101_PORT} with camera {CAMERA_INDEX}")
        print(f"Error: {e}")
        return

    stop_event = threading.Event()
    tracker_thread = FaceTrackerThread(env=env, stop_event=stop_event, camera_index=CAMERA_INDEX)
    tracker_thread.start()

    print("\n Idle mode... ")
    instruction = input("Enter task: ")

    print("Command received. Stopping idle mode... ")
    stop_event.set()
    tracker_thread.join()
    print("Idle mode stopped.")

    if instruction.lower() in ('q', 'quit'):
        env.close()
        print("Connection closed")
        return


    print("Initializing client for robot...")
    client = RobotClient(env=env, host=SERVER_IP, port=SERVER_PORT)

    print(f"Connecting to policy server at {SERVER_IP}:{SERVER_PORT}...")

    if client.start():
        print("Connected to server!")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            print(f"Executing: '{instruction}'. Press Ctrl+C to stop.")
            client.control_loop(instruction)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            client.stop()
            action_receiver_thread.join()
            print("Robot client shut down.")
    else:
        print("Failed to connect to the policy server.")

    env.close()
    print("Robot connection closed.")


if __name__ == "__main__":
    main()
