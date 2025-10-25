# robot_client.py

import threading
from lerobot.hardware.so101 import SO101Env
from lerobot.async_inference.robot_client import RobotClient


SO101_PORT = "COM6"  #USB PORT
SERVER_IP = "65.108.32.147"
SERVER_PORT = 8000


def main():
    print("Initializing SO101 env...")
    try:
        env = SO101Env(port=SO101_PORT)
    except Exception as e:
        print(f"Failed to connect to SO101 on port {SO101_PORT}: {e}")
        return

    print("Initializing client...")
    client = RobotClient(env=env, host=SERVER_IP, port=SERVER_PORT)

    print(f"Connecting to policy server at {SERVER_IP}:{SERVER_PORT}...")

    if client.start():
        print("Connected to server!")

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            instruction = input("Enter your command (e.g., 'pick up the red block'): ")          # TODO: LLM pipeline here


            print(f"Executing: '{instruction}'...")
            client.control_loop(instruction)

        except KeyboardInterrupt:
            print("\nStopping control loop.")
        finally:
            client.stop()
            action_receiver_thread.join()
            env.close()
            print("Robot client shut down.")
    else:
        print("Failed to connect to the policy server.")


if __name__ == "__main__":
    main()