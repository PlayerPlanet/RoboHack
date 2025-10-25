import threading
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.helpers import visualize_action_queue_size

# 1. Create the robot instance
"""Check out the cameras available in your setup by running `python lerobot/find_cameras.py`"""
# these cameras must match the ones expected by the policy
# check the config.json on the Hub for the policy you are using
camera_cfg = {
    "top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "side": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)
}

robot_cfg = SO101FollowerConfig(
  port="COM6",
  id="slave",
  cameras=camera_cfg
)

# 3. Create client configuration
client_cfg = RobotClientConfig(
    robot=robot_cfg,
    server_address="localhost:8080",
    policy_device="mps",
    policy_type="smolvla",
    pretrained_name_or_path="fracapuano/smolvla_async",
    chunk_size_threshold=0.5,
    actions_per_chunk=50,  # make sure this is less than the max actions of the policy
)

# 4. Create and start client
client = RobotClient(client_cfg)

# 5. Specify the task
task = "Don't do anything, stay still"

if client.start():
    # Start action receiver thread
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        # Run the control loop
        client.control_loop(task)
    except KeyboardInterrupt:
        client.stop()
        action_receiver_thread.join()
        # (Optionally) plot the action queue size
        visualize_action_queue_size(client.action_queue_size)