"""Simulated robot client for local demos.

This script watches for a task file (final_task.txt) placed by the conversation hub and
simulates executing the reported task. It intentionally has no dependency on `lerobot` so
it can run in any environment.

Usage:
    python simulated_client.py

The script writes a small feedback file to `outputs/simulated_feedback.txt` after simulating
execution. It renames the processed task file to avoid re-processing.
"""
from __future__ import annotations
import time
from pathlib import Path
import datetime

# Locate the project's RoboHack directory and the shared final_task file
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
TASK_FILE = PROJECT_ROOT / "final_task.txt"
OUTPUTS_DIR = PROJECT_ROOT.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = OUTPUTS_DIR / "simulated_feedback.txt"

PRINT_PREFIX = "[simulated-client]"


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def simulate_task(task: str) -> None:
    print(f"{PRINT_PREFIX} Starting simulation for task: {task}")
    # Very small sequence of steps for the demo
    steps = [
        ("planning", 1.0),
        ("moving", 1.5),
        ("acting", 1.0),
        ("finishing", 0.5),
    ]
    for name, dur in steps:
        print(f"{PRINT_PREFIX} {name}...")
        time.sleep(dur)
    # Write feedback
    FEEDBACK_FILE.write_text(f"{now_iso()}\t{task}\tstatus=completed\n")
    print(f"{PRINT_PREFIX} Simulation completed; feedback written to {FEEDBACK_FILE}")


def main():
    print(f"{PRINT_PREFIX} Simulated client started; watching {TASK_FILE}")
    try:
        while True:
            if TASK_FILE.exists():
                try:
                    task = TASK_FILE.read_text().strip()
                    if task:
                        simulate_task(task)
                    # Move the processed task to avoid re-processing
                    processed = TASK_FILE.with_suffix('.processed')
                    TASK_FILE.replace(processed)
                except Exception as e:
                    print(f"{PRINT_PREFIX} Error processing task: {e}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"{PRINT_PREFIX} Shutting down (KeyboardInterrupt)")


if __name__ == "__main__":
    main()
