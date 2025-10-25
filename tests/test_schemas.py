import time
from datetime import datetime

import pytest

from common import schemas


def test_prompt_meta_mapping():
    inp = {
        "id": "p1",
        "type": "user_intent",
        "text": "Pick up the red cube",
        "metadata": {"source": "stt"},
    }
    p = schemas.prompt_from_dict(inp)
    assert p.type == "user_intent"
    assert p.text == "Pick up the red cube"
    # meta field should be populated from legacy metadata
    assert p.meta == {"source": "stt"}
    assert p.metadata == {"source": "stt"}


def test_action_params_to_payload_mapping():
    inp = {
        "id": "a1",
        "type": "move",
        "params": {"object": "red_cube", "target": "shelf"},
    }
    a = schemas.action_from_dict(inp)
    assert a.payload == {"object": "red_cube", "target": "shelf"}
    # params should also be present for backward compatibility
    assert a.params == {"object": "red_cube", "target": "shelf"}
    assert a.version == "1.0"


def test_feedback_timestamp_parsing_epoch_and_iso():
    epoch = int(time.time())
    f1 = schemas.feedback_from_dict({"status": "completed", "timestamp": epoch})
    assert isinstance(f1.timestamp, datetime)

    iso = datetime.utcnow().isoformat()
    f2 = schemas.feedback_from_dict({"status": "completed", "timestamp": iso})
    assert isinstance(f2.timestamp, datetime)


def test_schemas_exported():
    assert set(schemas.SCHEMAS.keys()) >= {"Prompt", "Action", "Feedback"}
