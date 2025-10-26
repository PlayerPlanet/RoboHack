# robot_client_VLA_continuous.py

import threading
import asyncio
import queue
from RoboHack.common.event_router import EventRouter
import time
import numpy as np
import cv2
import sys
from pathlib import Path
from typing import Any
from lerobot.async_inference.helpers import TimedAction
import dotenv
dotenv.load_dotenv()

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
from RoboHack.client import conversation_hub  # Assuming this is your module for getting instructions



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
    "shoulder_pan.pos": -5.6,
    "shoulder_lift.pos": -50,
    "elbow_flex.pos": -50,
    "wrist_flex.pos": 100,
    "wrist_roll.pos": -100,
    "gripper.pos": -98.5,
}
safe ={
    "shoulder_pan.pos": 0,
    "shoulder_lift.pos": -90,
    "elbow_flex.pos": 80,
    "wrist_flex.pos": -100,
    "wrist_roll.pos": -2.6,
    "gripper.pos": -98.5,
}

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

    # 3. Create client configuration (reused for each task)
    client_cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address=f"{SERVER_IP}:{SERVER_PORT}",
        policy_device="cpu",
        policy_type="smolvla",  # Diffusion policy - flexible with action dims
        pretrained_name_or_path="y1y2y3/so101_test8_smolvla200k_augmented100",  # Diffusion model
        chunk_size_threshold=0.7,
        actions_per_chunk=50,
    )

    # Client will be created fresh for each task (not reusable after stop())
    client = None
    action_receiver_thread = None
    robot = None

    # --- EventRouter integration: subscribe to 'turn_status' and translate into stances
    stance_queue: "queue.Queue[dict]" = queue.Queue()
    stance_stop_event = threading.Event()
    _start_time = time.time()

    def _apply_stance_on_robot(msg: dict, env):
        """Translate a turn_status message into an action and send it to the robot.

        Handles 'speaking' state with jaw/gesture animation,
        and 'idle' state with a slow sway animation.

        Args:
            msg: Dictionary possibly containing 'turn', 'message'. Can be empty for idle.
            env: The robot environment object (e.g., SO101Follower instance).
        """
        nonlocal _start_time  # Use nonlocal to reference the outer scope variable
        # --- Dynamically get action dim ---
        try:
            keys = list(getattr(env, "action_features", {}).keys())
            if not keys:  # Fallback keys if action_features fails
                keys = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
                        "gripper.pos"]
            _action_dim = len(keys)
        except Exception:
            _action_dim = 7  # Final fallback
            keys = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
                    "gripper.pos"]
        # -----------------------------------

        # --- Animation Parameters (in actual position units, not normalized) ---
        # Speaking
        _jaw_frequency = 1.5
        _jaw_amplitude = 30.0  # gripper position range (degrees or raw units)
        _gesture_frequency = 0.5
        _gesture_amplitude = 0.15  # wrist_roll position range
        # Idle Sway
        _sway_frequency_lift = 0.2
        _sway_frequency_flex = 0.15
        _sway_amplitude_lift = 0.16  # shoulder_lift position range
        _sway_amplitude_flex = 0.1  # elbow_flex position range
        # --------------------------

        # --- Build Action Dictionary starting from home position ---
        action_dict = {k: float(home.get(k, 0.0)) for k in keys}  # Initialize with home values
        current_time = time.time() - _start_time

        # --- Determine State and Apply Animation ---
        turn = msg.get('turn')
        message = msg.get('message')

        if turn == 'assistant' and message == 'speaking':
            # --- Speaking Animation ---
            # print("Assistant speaking - animating robot...") # Reduce noise
            # Jaw (gripper) oscillates between open and closed
            jaw_value = _jaw_amplitude * (np.sin(current_time * 2 * np.pi * _jaw_frequency) * 0.5 + 0.5)
            # Gesture (wrist_roll) sways side to side
            gesture_value = home.get("wrist_roll.pos", 0.0) + _gesture_amplitude * np.sin(current_time * 2 * np.pi * _gesture_frequency)

            # Assign to dictionary using safe indices/keys
            if "gripper.pos" in action_dict:
                action_dict["gripper.pos"] = float(jaw_value)
            if "wrist_roll.pos" in action_dict:
                action_dict["wrist_roll.pos"] = float(gesture_value)

        else:  # Default to Idle Sway Animation
            # print("Idle/Listening - applying sway...") # Reduce noise
            # Sway relative to home position
            sway_lift_value = home.get("shoulder_lift.pos", -0.5) + _sway_amplitude_lift * np.sin(current_time * 2 * np.pi * _sway_frequency_lift)
            sway_flex_value = home.get("elbow_flex.pos", 1.0) + _sway_amplitude_flex * np.sin(current_time * 2 * np.pi * _sway_frequency_flex)

            # Assign sway
            if "shoulder_lift.pos" in action_dict:
                action_dict["shoulder_lift.pos"] = float(sway_lift_value)
            if "elbow_flex.pos" in action_dict:
                action_dict["elbow_flex.pos"] = float(sway_flex_value)

            # Gripper alternates between -100 and 100 every 1 second
            if "gripper.pos" in action_dict:
                # Alternate based on integer seconds: even seconds = -100, odd seconds = 100
                second = int(current_time) % 2
                gripper_value = -100.0 if second == 0 else 100.0
                action_dict["gripper.pos"] = gripper_value

        # --- Send action using send_action (NO CLIPPING - robot handles limits) ---
        try:
                env.send_action(action_dict)
        except Exception as e:
            print(f"Error sending stance action: {e}")
        # ------------------------------------
        return

    def _stance_consumer(env):
        """Thread that consumes stance messages OR triggers idle animation."""
        last_msg_time = time.time()
        current_msg = {}  # Start in idle state

        while not stance_stop_event.is_set():
            msg_received = False
            try:
                # Check for a new message without blocking indefinitely
                msg = stance_queue.get_nowait()
                current_msg = msg  # Update state if message received
                last_msg_time = time.time()
                msg_received = True
            except queue.Empty:

                pass
            except Exception as e:
                print(f"Stance queue error: {e}")
                # Continue loop even if queue read fails

            # --- Apply stance based on current_msg ---
            # This will run repeatedly, applying either speaking or idle animation
            try:
                _apply_stance_on_robot(current_msg, env)
            except Exception as e:
                print(f"Error applying stance: {e}")

            # If a message was received, mark it as done
            if msg_received:
                try:
                    stance_queue.task_done()
                except ValueError:
                    pass
                except Exception as e:
                    print(f"Stance task_done error: {e}")


            # Control loop speed
            time.sleep(0.05)  # Run at ~20Hz
    
    # --- Initialize EventRouter for conversation_hub ---
    # Create an EventRouter and asyncio event loop to enable turn-status events
    event_router = EventRouter()
    event_loop = asyncio.new_event_loop()
    
    # Start the event loop in a background thread
    def _run_event_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    event_loop_thread = threading.Thread(target=_run_event_loop, args=(event_loop,), daemon=True)
    event_loop_thread.start()
    
    # Configure conversation_hub to use this EventRouter
    conversation_hub.set_event_router(event_router, event_loop)
    print("EventRouter initialized and set for conversation_hub.")
    
    # --- Initialize robot for animations (outside of client/server connection) ---
    # Create a standalone robot instance just for animations during conversation
    print("Initializing robot for animations...")
    animation_robot_cfg = SO101FollowerConfig(
        port=SO101_PORT,
        id="follower_so101_animations",
        cameras=camera_cfg
    )
    animation_robot = SO101Follower(animation_robot_cfg)
    
    # Connect to the robot
    print("Connecting animation robot...")
    try:
        animation_robot.connect()
        print("Animation robot connected successfully.")
    except Exception as e:
        print(f"Error: Failed to connect animation robot: {e}")
        print("Cannot proceed without robot connection.")
        return
    
    # Move robot to home position before starting
    print("Moving robot to home position...")
    try:
        animation_robot.send_action(home)
        time.sleep(2.0)
        print("Robot is now at home position.")
    except Exception as e:
        print(f"Warning: Could not send home action: {e}")
    
    # Start the stance consumer thread for animations
    print("Starting animation stance consumer...")
    async def _turn_status_listener():
        q = await event_router.subscribe("turn_status")
        while not stance_stop_event.is_set():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.5)
                stance_queue.put_nowait(msg)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
    
    # Schedule the listener on the event loop
    asyncio.run_coroutine_threadsafe(_turn_status_listener(), event_loop)
    
    # Start consumer thread
    animation_consumer_thread = threading.Thread(target=_stance_consumer, args=(animation_robot,), daemon=True)
    animation_consumer_thread.start()
    print("Animation stance consumer started.")
    
    # --- PHASE 2: Main Instruction Loop ---
    # Each iteration connects to server, executes task, then properly closes the stream
    try:
        instruction = None
        while True:
            # --- START IDLE MODE ---
            print("\nStarting idle mode. Waiting for instruction...")

            # This blocks until an instruction is received
            instruction = conversation_hub.main_loop(instruction)

            # --- STOP IDLE MODE ---
            print("Instruction received. Stopping idle mode.")

            if instruction.lower() in ('q', 'quit'):
                print("Exiting...")
                break  # Exit the instruction loop

            print(f"Executing: '{instruction}'")
            
            # --- CREATE A FRESH CLIENT FOR THIS TASK ---
            # CRITICAL: RobotClient cannot be reused after stop() - must recreate
            print(f"Creating new client for server at {client_cfg.server_address}...")
            client = RobotClient(client_cfg)
            
            # --- CONNECT TO SERVER FOR THIS TASK ---
            print("Connecting to server...")
            
            if not client.start():
                print("Failed to connect to the policy server. Retrying in 2 seconds...")
                time.sleep(2.0)
                continue
            
            print("Connected to server!")
            
            # Start the background thread that receives actions
            action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
            action_receiver_thread.start()
            
            # Get the robot object (created by client.start())
            robot = client.robot
            
            # Note: Robot is already at home position from initialization
            # The animation_robot handles animations during conversation
            # This robot instance is only for executing the VLA policy task
            
            # --- PHASE 3: Task Execution Loop (for one task) ---
            # CRITICAL: Clear the shutdown_event before starting a new control loop
            client.shutdown_event.clear()
            
            # Run the control loop but ensure it exits after timeout by scheduling a safe-home injection.
            timeout_s = 30.0

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
                print("Starting control loop...")
                client.control_loop(instruction)
                print("Control loop completed.")
            except KeyboardInterrupt:
                print("\nTask cancelled by user.")
            except Exception as e:
                print(f"Error during control loop: {e}")
            finally:
                # cancel timer if control_loop finished earlier
                stop_timer.cancel()
                
                # Ensure robot is in a safe home position when the task ends.
                # CRITICAL: Do this BEFORE stopping the client so the action can execute
                try:
                    if robot is not None and hasattr(robot, "send_action"):
                        print("Returning robot to home position...")
                        robot.send_action(home)
                        # Give the robot time to move to home position before disconnecting
                        time.sleep(2.0)
                        print("Robot returned to home position.")
                except Exception as e:
                    print(f"Warning: Could not return to home: {e}")
                
                # --- CRITICAL: Properly close the action stream ---
                # Only stop AFTER the robot has moved to home
                print("Closing action stream...")
                try:
                    client.stop()
                except Exception as e:
                    print(f"Warning during client.stop(): {e}")
                
                # Join the action receiver thread
                if action_receiver_thread is not None and action_receiver_thread.is_alive():
                    try:
                        action_receiver_thread.join(timeout=2.0)
                    except Exception as e:
                        print(f"Warning during thread join: {e}")
                
                # Note: stance consumer runs globally and persists across tasks
                
                # Brief delay to allow gRPC connection to fully close
                time.sleep(0.3)
                
                print("Action stream closed. Ready for new task.\n")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Final cleanup
        cv2.destroyAllWindows()
        
        # Stop the stance consumer
        try:
            stance_stop_event.set()
            if animation_consumer_thread is not None and animation_consumer_thread.is_alive():
                animation_consumer_thread.join(timeout=1.0)
                print("Animation stance consumer stopped.")
        except Exception as e:
            print(f"Warning during stance consumer cleanup: {e}")
        
        # Stop the client if running
        try:
            if client is not None and client.running:
                client.stop()
        except Exception:
            pass
        
        # Disconnect the animation robot
        try:
            if animation_robot is not None:
                animation_robot.disconnect()
                print("Animation robot disconnected.")
        except Exception as e:
            print(f"Warning during animation robot cleanup: {e}")
        
        # Stop the EventRouter event loop
        try:
            if event_loop is not None and event_loop.is_running():
                event_loop.call_soon_threadsafe(event_loop.stop)
                event_loop_thread.join(timeout=1.0)
                print("EventRouter event loop stopped.")
        except Exception as e:
            print(f"Warning during EventRouter cleanup: {e}")
        
        print("Robot client shut down.")
        print("Goodbye!")


if __name__ == "__main__":
    main()