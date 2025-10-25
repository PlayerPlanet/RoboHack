## README.md

# 🤖 RoboThinker Framework
> A distributed, multimodal robotics framework that connects speech, vision, and action — enabling robots to "think out loud."

---

## 🚀 Overview

**RoboThinker** is a modular framework for **embodied AI**, combining:
- **STT/TTS** — Speech understanding and synthesis (via SpeechT5)
- **VLM Server** — Vision-Language understanding and reasoning
- **PolicyServer** — Task execution and motion policy orchestration
- **RobotClient** — Real robot or simulated execution

The architecture is **distributed**: each cognitive component runs as an independent async server.

---

## 🧠 Motivation

> “What if a robot could understand you, talk back, and act intelligently — in real time?”

This project enables that — through modular reasoning, vision, and control layers that communicate via async WebSockets and structured JSON messages.

---

## 🧩 Architecture Overview

```
User → (speech)
  ↓
[STT/TTS Server] --text--> [VLM Server] --action prompt--> [PolicyServer] ↔ [RobotClient]

VLM Server  : Understands user intent and visual input
PolicyServer: Converts intent → robot actions (via LeRobot Policy API)
RobotClient : Executes commands on hardware or simulation
```

Each subsystem runs independently and communicates asynchronously via HTTP/WebSocket APIs.

---

## 🛠️ Setup

### Requirements
- Python 3.11+
- `uv` for dependency management
- FastAPI, Transformers, Torchaudio, OpenCV, pytest
- GPU (16–200GB VRAM)
- LeRobot async SDK (for PolicyServer and RobotClient)

### Installation
```bash
uv venv
uv pip install -r requirements.txt
```

### Run Components
```bash
uv run python -m stt_tts.app
uv run python -m vlm_server.vlm_app
uv run python -m policy_server.policy_server
uv run python -m policy_server.robot_client
```

---

## 🧰 Core Components

| Component | Description | Status |
|------------|--------------|--------|
| STT/TTS | Speech-to-text and text-to-speech via SpeechT5 | ✅ Defined |
| VLM Server | Vision-language reasoning | ✅ Defined |
| PolicyServer | Task reasoning and policy execution | ✅ Defined |
| RobotClient | Executes robot actions via LeRobot SDK | ✅ Defined |
| FrameBuffer | Stores last 10s of frames (weighted sampling) | ✅ Defined |
| EventRouter | Async message bus per service | ✅ Defined |

---

## 🔄 Communication Flow

1. User speech → STT → text
2. Text prompt → VLM Server → reasoning & response
3. Response → TTS → robot speaker output
4. When confident → VLM sends JSON action request → PolicyServer
5. PolicyServer streams policy actions → RobotClient
6. RobotClient executes & provides feedback

---

## 🧩 Data Schemas

### Prompt
```json
{
  "type": "user_intent",
  "text": "Pick up the red cube and place it on the shelf",
  "meta": {"source": "stt"}
}
```

### Action Request
```json
{
  "version": "1.0",
  "id": "uuid4",
  "type": "move",
  "payload": {"object": "red_cube", "target": "shelf"},
  "meta": {"certainty": 0.93}
}
```

### Robot Feedback
```json
{
  "status": "completed",
  "timestamp": 1730000000
}
```

---

## 🧱 Folder Structure

```
RoboThinker/
│
├── stt_tts/
│   ├── stt.py
│   ├── tts.py
│   └── app.py
│
├── vlm_server/
│   ├── vlm.py
│   ├── vlm_app.py
│   └── frame_buffer.py
│
├── policy_server/
│   ├── policy_server.py
│   ├── robot_client.py
│   └── adapter_lerobot.py
│
└── common/
    ├── event_router.py
    ├── schemas.py
    └── utils.py
```

---

## 🧪 Development Model

- **Test-first principle**: write mocks & tests before implementing functions
- **Async by default**: all models and communication use `asyncio`
- **Parallel coding**: each service is independently runnable and testable

---

## 🧱 Future Extensions

- Add **World Model (WM)** for environment understanding
- Add **dialogue memory** across sessions
- Expand PolicyServer for multi-robot control

---

# IMPLEMENTATION_PLAN.md

# ⚙️ Implementation Plan: RoboThinker Distributed Framework

This document defines the **updated technical plan**, **module responsibilities**, and **ticketing system** for parallel development.

---

## 1. System Design

- Each subsystem is an **independent async FastAPI app**.
- Communication via WebSocket or REST between services.
- Robot communication handled via LeRobot’s async `PolicyServer` / `RobotClient` pair.

---

## 2. Updated Module Specifications

### 2.1 STT/TTS (`stt_tts/`)
```python
class SpeechInterface:
    async def transcribe(audio_chunk: bytes) -> str: ...
    async def synthesize(text: str) -> bytes: ...
```
- Uses SpeechT5 for both STT and TTS
- Outputs base64-encoded PCM audio

### 2.2 VLM Server (`vlm_server/`)
```python
class VLM:
    async def describe(frame_batch, prompt: str) -> dict: ...
```
- Accepts POST `/prompt` with STT text + optional frame context
- Returns response text to TTS
- Sends action JSON to PolicyServer when confidence > threshold

### 2.3 PolicyServer (`policy_server/`)
```python
from lerobot.policy_server import PolicyServer

class RoboPolicyServer(PolicyServer):
    async def on_message(self, prompt: dict): ...
```
- Receives task prompt from VLM
- Converts into structured robot commands
- Streams commands asynchronously to RobotClient

### 2.4 RobotClient (`policy_server/robot_client.py`)
```python
from lerobot.robot_client import RobotClient

class RoboClient(RobotClient):
    async def observe(self): ...
```
- Connects to PolicyServer
- Executes actions
- Streams status feedback

### 2.5 EventRouter (`common/event_router.py`)
- Local pub/sub using `asyncio.Queue`
- Scoped per service, not global

### 2.6 FrameBuffer (`vlm_server/frame_buffer.py`)
- Holds last 10s of frames with **time-weighted sampling**

---

## 3. API Schema Summary

| Service | Endpoint | Direction | Description |
|----------|-----------|------------|--------------|
| STT/TTS | `/audio` | Client → Server | Audio input / speech output |
| VLM | `/prompt` | STT → VLM | Process user prompt |
| VLM | `/action` | VLM → PolicyServer | Send structured action JSON |
| PolicyServer | `/status` | RobotClient → PolicyServer | Report robot state |

---

## 4. Testing Strategy

- **Unit tests:** mock inputs, validate function I/O contracts
- **Integration tests:** simulate full network pipeline between services
- **System tests:** deploy local services and simulate robot interaction

---

## 5. Developer Tickets

### 👩‍💻 Developer A – STT/TTS Subsystem
- Implement `stt.py`, `tts.py`, and FastAPI app endpoints.
- Write test mocks for SpeechT5 input/output.
- Ensure async base64 audio streaming.

### 🧑‍💻 Developer B – VLM Server
- Implement `vlm_app.py` with `/prompt` and `/action` routes.
- Integrate vision model (OpenVLM/Qwen-Vision).
- Connect FrameBuffer for causal temporal reasoning.
- Send structured JSON action prompts to PolicyServer.

### 👨‍💻 Developer C – PolicyServer
- Merge logic from `VLA.py` and `robot.py`.
- Implement `RoboPolicyServer` subclass with async reasoning.
- Parse VLM action JSON and stream to RobotClient.
- Use LeRobot SDK.

### 👩‍💻 Developer D – RobotClient & Integration Tests
- Implement `RoboClient` subclass with mock sensors.
- Connect to PolicyServer and execute dummy commands.
- Build integration test harness simulating end-to-end flow.

---

## 6. Next Milestone Checklist

- [ ] STT/TTS FastAPI prototype
- [ ] VLM server integration with FrameBuffer
- [ ] PolicyServer and RobotClient connected via LeRobot async API
- [ ] Full pipeline test from speech → reasoning → robot action
- [ ] Logging and metrics dashboard

---

## 7. Future Considerations

- Distributed orchestration (Kubernetes / Docker Compose)
- Cross-service health monitoring and logging
- Optional multimodal world model (text + scene graph)

---

