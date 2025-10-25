import threading
import time
import numpy as np

# Lerobot imports for robot connection
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# Import the animation thread class from the other file
from idle_animation import IdleAnimationThread

SO101_PORT = "COM6"  # USB PORT
CAMERA_INDEX = 1


def main():
    print("Connecting to robot for idle animation...")
    try:
        # 1. Create a dummy camera config (required by robot config)
        idle_camera_cfg = {
            "primary": OpenCVCameraConfig(
                index_or_path=CAMERA_INDEX,
                width=640,
                height=480,
                fps=30
            ),
        }

        # 2. Create the robot config
        idle_robot_cfg = SO101FollowerConfig(
            port=SO101_PORT,
            id="follower_so101_idle",  # Use a unique ID
            cameras=idle_camera_cfg
        )

        # 3. Create the robot object
        idle_env = SO101Follower(idle_robot_cfg)

        # 4. Connect to the hardware
        idle_env.connect()
        print("Connected successfully!")

    except Exception as e:
        print(f"Failed to connect on {SO101_PORT} with camera {CAMERA_INDEX}")
        print(f"Error: {e}")
        return  # Exit if connection fails

    stop_event = threading.Event()
    animation_thread = IdleAnimationThread(env=idle_env, stop_event=stop_event)
    animation_thread.start()

    print("\nIdle animation running...")
    print("Press Ctrl+C to stop.")

    try:
        # Keep the main thread alive while the animation runs
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping animation...")
    finally:
        # Cleanly stop the thread and disconnect
        stop_event.set()
        animation_thread.join()
        idle_env.disconnect()
        print("Robot connection closed.")


if __name__ == "__main__":
    main()