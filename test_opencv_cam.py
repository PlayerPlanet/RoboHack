from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import cv2

config = OpenCVCameraConfig(
    index_or_path=0,
    fps=15,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

camera = OpenCVCamera(config)

# --- Patch the backend directly ---
def connect_with_dshow(self):
    import cv2
    cap = cv2.VideoCapture(self.config.index_or_path, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise ConnectionError(f"Failed to open camera {self.config.index_or_path} via DSHOW")
    self.cap = cap
    print("✅ Camera connected via DSHOW")

camera.connect = connect_with_dshow.__get__(camera, OpenCVCamera)

# --- Test capture ---
camera.connect()
ret, frame = camera.cap.read()
print("Frame OK:", ret, frame.shape if ret else None)
camera.cap.release()
