<!-- .github/copilot-instructions.md for RoboHack -->
# RoboHack — Copilot / AI agent instructions

Purpose: give an AI coding agent the minimal, concrete facts it needs to be productive in this repository (architecture, run/debug flows, conventions, and where to look for important patterns).

- Big picture
  - This repo is a small robotics demo with two runnable roles: a policy server and a robot client.
  - Key entrypoints:
    - `RoboHack/server/policy_server.py` — loads a policy (HFLeRobotPolicy) and starts a `PolicyServer` that waits for client connections.
    - `RoboHack/client/robot_client.py` — connects to the policy server and runs a control loop against an `SO101Env` hardware abstraction.
    - `main.py` is a trivial top-level script (prints a greeting) and not used for the core client/server flow.

- Architecture & dataflow (what to know fast)
  - Policy server creates a fake or real env (`SO101Env(fake_hardware=True)` used for local testing), instantiates `HFLeRobotPolicy`, then wraps it in `PolicyServer` to accept networked clients.
  - Robot client creates a physical env `SO101Env(port=SO101_PORT)` and a `RobotClient` that connects to the policy server; it then receives actions and calls `client.control_loop(instruction)`.
  - Communication is over TCP (server host/port constants found in `robot_client.py` and `policy_server.py`).

- Important, discoverable patterns and conventions
  - Hardware abstraction: `lerobot.hardware.so101.SO101Env` is the layer that isolates real vs fake hardware. Look for `fake_hardware=True` to run without device access.
  - Policy loading: `HFLeRobotPolicy(repo_id=..., device=...)` — model repo id is passed directly; device selection uses `torch.device("cuda" if available else "cpu")`.
  - Hard-coded values to be aware of (edit or override when changing behavior):
    - `SO101_PORT = "COM6"` in `RoboHack/client/robot_client.py` — Windows COM port for robot.
    - `SERVER_IP = "65.108.32.147"` and `SERVER_PORT = 8000` in `robot_client.py`.
  - Tickets & workflow: `sprints.json` drives ticket selection; `HOW_TO_CONTRIBUTE.md` lists the dev flow (pick role → select ticket → vibecode → pytest).

- Dev, run & debug commands (examples found by reading files)
  - Create a Python 3.11+ environment and install dependencies from `pyproject.toml`.
  - Run policy server (local, fake hardware):
    ```powershell
    python RoboHack/server/policy_server.py
    # or, as a package module if you prefer
    python -m RoboHack.server.policy_server
    ```
  - Run robot client (connects to server IP/port and tries COM6):
    ```powershell
    python RoboHack/client/robot_client.py
    # or
    python -m RoboHack.client.robot_client
    ```
  - Unit tests: repository references `pytest` in contribution flow; on Windows PowerShell we prefer using the project's `uv` helper to ensure the correct environment. Run all tests with:
    ```powershell
    uv run python -m pytest -q
    ```

- Project-specific notes for the agent
  - Prefer edits that preserve the simple script style used here (small scripts under `RoboHack/client` and `RoboHack/server`). Avoid large refactors unless requested.
  - When changing run configuration, search for the constants above (`SO101_PORT`, `SERVER_IP`, `SERVER_PORT`) — they're intentionally colocated for quick edits.
  - To test without hardware, run the server using `SO101Env(fake_hardware=True)` (already present in `policy_server.py`). This is the canonical way to run logic without physical device access.
  - The code expects large ML dependencies (torch, transformers, accelerate, bitsandbytes). Avoid adding heavy CI steps that install full GPU toolchains unless explicitly requested.

- Files to inspect for implementation patterns
  - `RoboHack/client/robot_client.py` — client lifecycle, threading, and how `RobotClient` is used.
  - `RoboHack/server/policy_server.py` — policy loading and server lifecycle (start/stop pattern).
  - `pyproject.toml` — authoritative dependency list and minimum Python version (>=3.11).
  - `HOW_TO_CONTRIBUTE.md` — existing dev workflow summary (tickets, vibecode, pytest).

- Minimal checklist for PRs an AI agent should follow
  - Keep changes small and focused to one role (client or server) per PR.
  - Preserve existing run shortcuts and constants; if changing them, update README and `HOW_TO_CONTRIBUTE.md` if necessary.
  - If you add or change a runnable script, include a short example command in the README or this file.

If anything here is unclear or you want extra examples (for instance: how to stub `SO101Env` in unit tests, or a recommended env var pattern to replace hard-coded constants), tell me which section to expand and I will iterate.
