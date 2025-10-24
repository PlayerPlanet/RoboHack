"""Stream webcam frames to a remote server via an SSH-forwarded tunnel.

Supports two simple transport modes:
- tcp: connect to a host:port and send framed JPEG images (4-byte big-endian length + bytes)
- http: POST each frame as multipart/form-data to a provided URL

This file is intentionally dependency-light; it uses OpenCV for capture and `requests` for HTTP.

Usage examples (basic):
  python stream_webcam_ssh.py --mode tcp --host localhost --port 9000
  python stream_webcam_ssh.py --mode http --url http://localhost:8000/frames

Notes about SSH tunnels:
- To forward a local TCP sender to be reachable from the remote server (remote accepts connections
  and connects back to you), you can create a reverse tunnel from the remote host:

  # Open a reverse tunnel on the remote host so remote_port on remote maps to your local_port
  ssh -R remote_port:localhost:local_port user@remote_host

  Example: expose local TCP sender (listening on local_port) to remote host on port 9001:
  ssh -R 9001:localhost:9000 user@remote_host

- To send frames directly to an HTTP endpoint on the remote server via an SSH local-forward,
  forward a local port to the remote HTTP server and POST to the local forwarded port:

  ssh -L 8000:localhost:80 user@remote_host
  # then send to http://localhost:8000/your-endpoint

Replace the exact SSH command with one that matches your access and desired direction.

The user promised to provide specific HTTP request format later; this script implements a
conventional POST-per-frame approach which is easy to adapt to a specific request format.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import struct
import sys
import threading
import time
from typing import Optional

try:
    import cv2
except Exception as e:  # pragma: no cover - runtime dependency
    print("Missing dependency: opencv-python is required. Install with: pip install opencv-python", file=sys.stderr)
    raise

try:
    import requests
except Exception:  # pragma: no cover - runtime dependency
    requests = None

LOG = logging.getLogger("stream_webcam_ssh")


class FrameStreamer:
    def __init__(self, src: int = 0, width: Optional[int] = None, height: Optional[int] = None, fps: float = 10.0):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None
        self.running = False

    def open(self) -> None:
        LOG.info("Opening video capture %s", self.src)
        self.capture = cv2.VideoCapture(self.src, cv2.CAP_ANY)
        if not self.capture.isOpened():
            # Try alternate index if 0 failed
            LOG.warning("Primary capture failed, trying index 1")
            self.capture = cv2.VideoCapture(1, cv2.CAP_ANY)
        if not self.capture.isOpened():
            raise RuntimeError("Unable to open any webcam device")

        if self.width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps:
            self.capture.set(cv2.CAP_PROP_FPS, float(self.fps))

    def read_frame(self) -> Optional[bytes]:
        if self.capture is None:
            return None
        ret, frame = self.capture.read()
        if not ret:
            LOG.debug("Frame read returned False")
            return None
        # encode to JPEG
        ret2, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret2:
            LOG.warning("Failed to encode frame to JPEG")
            return None
        return buf.tobytes()

    def close(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass


def tcp_send_loop(host: str, port: int, streamer: FrameStreamer, reconnect: bool = True) -> None:
    sock: Optional[socket.socket] = None
    try:
        while True:
            if sock is None:
                LOG.info("Connecting to %s:%d", host, port)
                sock = socket.create_connection((host, port), timeout=10)
                LOG.info("Connected to %s:%d", host, port)

            frame = streamer.read_frame()
            if frame is None:
                LOG.debug("No frame received; sleeping briefly")
                time.sleep(0.01)
                continue

            # send length-prefixed frame
            try:
                header = struct.pack('!I', len(frame))
                sock.sendall(header + frame)
            except Exception as exc:
                LOG.warning("TCP send failed: %s", exc)
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
                if not reconnect:
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass


def http_send_loop(url: str, streamer: FrameStreamer, interval: float = 0.0, headers: Optional[dict] = None) -> None:
    if requests is None:
        raise RuntimeError("requests is required for http mode. Install with: pip install requests")

    try:
        while True:
            frame = streamer.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            files = {'frame': ('frame.jpg', frame, 'image/jpeg')}
            try:
                resp = requests.post(url, files=files, headers=headers or {}, timeout=10)
                LOG.debug("POST %s -> %d", url, getattr(resp, 'status_code', None))
            except Exception as exc:
                LOG.warning("HTTP POST failed: %s", exc)
                time.sleep(1)

            if interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stream webcam over an SSH-forwarded tunnel")
    sub = parser.add_argument_group('transport')
    sub.add_argument('--mode', choices=['tcp', 'http'], default='tcp', help='transport mode')
    sub.add_argument('--host', default='localhost', help='target host for TCP')
    sub.add_argument('--port', type=int, default=9000, help='target port for TCP')
    sub.add_argument('--url', help='target URL for HTTP POST mode')

    parser.add_argument('--src', type=int, default=0, help='video capture source index')
    parser.add_argument('--width', type=int, default=None, help='requested capture width')
    parser.add_argument('--height', type=int, default=None, help='requested capture height')
    parser.add_argument('--fps', type=float, default=10.0, help='capture FPS target')
    parser.add_argument('--interval', type=float, default=0.0, help='minimum seconds between HTTP posts (http mode)')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    args = parser.parse_args(argv)

    log_level = logging.WARNING
    if args.verbose >= 1:
        log_level = logging.INFO
    if args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')

    streamer = FrameStreamer(src=args.src, width=args.width, height=args.height, fps=args.fps)
    try:
        streamer.open()
    except Exception as exc:
        LOG.error("Failed to open webcam: %s", exc)
        sys.exit(2)

    # shutdown event
    stop_event = threading.Event()

    def handle_sigint(sig, frame):
        LOG.info("Signal received, stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        if args.mode == 'tcp':
            LOG.info("Starting TCP send to %s:%d", args.host, args.port)
            tcp_send_loop(args.host, args.port, streamer, reconnect=True)
        elif args.mode == 'http':
            if not args.url:
                LOG.error("--url is required in http mode")
                sys.exit(2)
            LOG.info("Starting HTTP POST to %s (interval=%s)", args.url, args.interval)
            http_send_loop(args.url, streamer, interval=args.interval)
    finally:
        LOG.info("Cleaning up")
        streamer.close()


if __name__ == '__main__':
    main()
