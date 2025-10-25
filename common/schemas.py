from __future__ import annotations
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator


class ActionType(str, Enum):
    MOVE = "move"
    PICK = "pick"
    PLACE = "place"
    SPEAK = "speak"
    NOOP = "noop"
    CUSTOM = "custom"


class Prompt(BaseModel):
    id: str = Field(..., description="Unique prompt id")
    # According to robothinker_docs.md, `type` and `text` are required
    type: str = Field(..., description="Prompt semantic type (e.g. 'user_intent')")
    text: str = Field(..., description="Natural language text for the prompt")
    modalities: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional multimodal content")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary metadata (preferred name in docs)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Legacy metadata (kept for compatibility)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp")

    @validator("id")
    def id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("id must be non-empty")
        return v

    @root_validator(pre=True)
    def unify_meta(cls, values):
        # Prefer explicit `meta`, fall back to `metadata` for backward compatibility
        if "meta" not in values and "metadata" in values:
            values["meta"] = values.get("metadata", {})
        if "metadata" not in values and "meta" in values:
            values["metadata"] = values.get("meta", {})
        return values


class Action(BaseModel):
    version: str = Field("1.0", description="Action schema version")
    id: str = Field(..., description="Unique action id")
    type: ActionType = Field(..., description="Action type")
    # payload is required per robothinker_docs.md; keep params as legacy mapping
    payload: Dict[str, Any] = Field(..., description="Action-specific payload (preferred name in docs)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Legacy params (kept for compatibility)")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    issued_by: Optional[str] = Field(None, description="Component that issued the action")
    issued_at: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp")

    @validator("id")
    def action_id_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("id must be non-empty")
        return v

    @root_validator(pre=True)
    def unify_payload(cls, values):
        # Accept legacy 'params' and map to 'payload' when needed, and keep both for compatibility
        if "payload" not in values and "params" in values:
            values["payload"] = values.get("params", {})
        if "params" not in values and "payload" in values:
            values["params"] = values.get("payload", {})
        return values


class FeedbackStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class Feedback(BaseModel):
    # Docs show feedback may be simple; accept missing action_id (e.g., system-level status)
    action_id: Optional[str] = Field(None, description="Related action id")
    status: FeedbackStatus = Field(..., description="Result status")
    message: Optional[str] = Field(None, description="Human-readable message")
    metrics: Optional[Dict[str, float]] = Field(default_factory=dict, description="Numeric metrics")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary extra details")
    # timestamp is required per docs; accept numeric epoch or datetime
    timestamp: datetime = Field(..., description="UTC timestamp")

    @validator("action_id")
    def action_id_present(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("action_id must be non-empty")
        return v

    @validator("timestamp", pre=True, always=True)
    def parse_timestamp(cls, v):
        if v is None:
            raise ValueError("timestamp is required")
        # allow numeric epoch
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v)
        # allow ISO string or numeric string
        if isinstance(v, str):
            try:
                num = float(v)
            except Exception:
                # try to parse ISO format
                try:
                    return datetime.fromisoformat(v)
                except Exception:
                    raise ValueError("timestamp string not a valid ISO or epoch")
            else:
                return datetime.fromtimestamp(num)
        if isinstance(v, datetime):
            return v
        raise ValueError("Unsupported timestamp type")


def prompt_from_dict(d: Dict[str, Any]) -> Prompt:
    return Prompt.parse_obj(d)


def action_from_dict(d: Dict[str, Any]) -> Action:
    if "type" in d and isinstance(d["type"], str):
        try:
            d["type"] = ActionType(d["type"])
        except ValueError:
            d["type"] = ActionType.CUSTOM
    return Action.parse_obj(d)


def feedback_from_dict(d: Dict[str, Any]) -> Feedback:
    return Feedback.parse_obj(d)


SCHEMAS = {
    "Prompt": Prompt.schema(),
    "Action": Action.schema(),
    "Feedback": Feedback.schema(),
}
