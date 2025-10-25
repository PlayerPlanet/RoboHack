from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

config = PolicyServerConfig(
    host="localhost",
    port=8000,
)
serve(config)
