# RoboHack
Repo for the Aaltoes robotic + ai hackathon 24.-26.10.2025.

## Webcam streaming via SSH tunnel

This repository now includes `stream_webcam_ssh.py`, a small script to capture webcam frames
and stream them to a remote endpoint over either TCP or HTTP. It's written to be used together
with an SSH port forwarding (local or reverse) so the remote server can receive the stream.

Install dependencies (recommended in a virtualenv):

```powershell
python -m pip install --upgrade pip
pip install opencv-python requests
```

Example usage (TCP mode):

```powershell
python stream_webcam_ssh.py --mode tcp --host localhost --port 9000
```

Example usage (HTTP mode):

```powershell
python stream_webcam_ssh.py --mode http --url http://localhost:8000/frames --interval 0.1
```

SSH tunnel examples:

- Reverse tunnel (expose a local service to the remote host):

```powershell
ssh -R 9001:localhost:9000 user@remote_host
# remote host can now connect to localhost:9001 (on remote) and it will be forwarded to your local 9000
```

- Local-forward (send to remote HTTP endpoint via a forwarded local port):

```powershell
ssh -L 8000:localhost:80 user@remote_host
# Then send HTTP requests to http://localhost:8000 which are forwarded to remote_host:80
```

Adapt the commands to your environment and security policies. If you provide a specific HTTP
request format, the script's `http` mode can be adjusted to match it (headers, payload, auth, etc.).

