import re
from pathlib import Path
import pytest


BASE = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


@pytest.mark.parametrize("task,path,required", [
    ("S1-02 common schemas", BASE / "RoboHack" / "common" / "schemas.py", ["Prompt", "Action", "Feedback"]),
    ("S1-03 event router", BASE / "RoboHack" / "common" / "event_router.py", ["EventRouter", "asyncio.Queue"]),
    ("S1-04 STT skeleton", BASE / "RoboHack" / "stt.py", ["record_audio", "transcribe"]),
    ("S1-06 TTS skeleton", BASE / "RoboHack" / "tts.py", ["init_tts_engine", "speak_text"]),
    ("S1-07 FrameBuffer", BASE / "RoboHack" / "vlm_server" / "frame_buffer.py", ["FrameBuffer"]),
    ("S1-08 VLM app", BASE / "RoboHack" / "vlm_server" / "vlm_app.py", ["FastAPI", "/prompt"]),
    ("S1-09 PolicyServer skeleton", BASE / "RoboHack" / "server" / "policy_server.py", ["PolicyServer", "on_message"]),
    ("S1-10 RobotClient subclass", BASE / "RoboHack" / "server" / "robot_client.py", ["RobotClient", "observe"]),
    ("S2-01 VLM /action endpoint", BASE / "RoboHack" / "vlm_server" / "vlm_app.py", ["/action", "action"]),
    ("S2-03 VLM->TTS integration", BASE / "RoboHack" / "vlm_server" / "vlm.py", ["tts", "speak_text"]),
    ("S2-04 VLM->PolicyServer client", BASE / "RoboHack" / "vlm_server" / "vlm.py", ["PolicyServer", "http", "post"]),
    ("S2-05 PolicyServer.on_message", BASE / "RoboHack" / "server" / "policy_server.py", ["def on_message", "async def on_message"]),
    ("S2-06 Streaming commands", BASE / "RoboHack" / "server" / "policy_server.py", ["stream", "async"]),
    ("S2-07 RobotClient.observe mock", BASE / "RoboHack" / "server" / "robot_client.py", ["def observe", "observe(self"]),
    ("S2-08 EventRouter integration", BASE / "RoboHack" / "common" / "event_router.py", ["subscribe", "publish"]),
    ("S2-10 Logging / validation", BASE / "RoboHack" / "server" / "policy_server.py", ["logging", "pydantic", "json"]),
    ("S3-01 Integration test presence", BASE / "tests" / "test_sprints_subtasks.py", ["S3-01"]),
    ("S3-04 Health/version endpoints", BASE / "RoboHack" / "vlm_server" / "vlm_app.py", ["/version", "/health", "health"]),
])
def test_subtask_files_and_symbols(task, path: Path, required):
    """Basic checks that the repository contains files and key symbol names mentioned in the sprint docs.

    These tests intentionally avoid importing heavy dependencies. They assert file presence
    and that the file text contains expected identifiers (class/function names, endpoints,
    or keywords referenced in the sprint/subtask descriptions).
    """

    # File must exist for the subtask to be considered implemented
    assert path.exists(), f"Expected file for '{task}' at {path} does not exist"

    text = _read(path)
    assert text, f"File {path} is empty or could not be read"

    for keyword in required:
        # do a case-insensitive search for keywords
        assert re.search(re.escape(keyword), text, re.IGNORECASE), (
            f"Expected keyword '{keyword}' in {path} for task '{task}'"
        )


def test_sprints_json_has_subtasks():
    sprints = Path(BASE / "sprints.json")
    assert sprints.exists(), "sprints.json file is missing from repository root"
    txt = sprints.read_text(encoding="utf-8")
    assert "Sprint" in txt and "subtasks" in txt
