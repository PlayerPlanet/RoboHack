# policy_server.py

import torch
from lerobot.common.policies.hf_policy import HFLeRobotPolicy
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.hardware.so101 import SO101Env


def main():
    print("Initializing environment...")
    dummy_env = SO101Env(fake_hardware=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading OpenVLA 7-b policy on {device}")

    policy = HFLeRobotPolicy(
        repo_id="openvla/openvla-7b",
        observation_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        device=device
    )
    print("Policy loaded.")

    dummy_env.close()

    host = "0.0.0.0"
    port = 8000

    server = PolicyServer(policy=policy, host=host, port=port)

    print(f"Starting policy server at {host}:{port}")
    print("Waiting for client to connect...")

    try:
        server.start()
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()