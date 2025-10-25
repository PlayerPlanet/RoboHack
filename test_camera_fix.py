"""Test that the camera fix works."""

# Import the fix FIRST
from RoboHack.client.camera_fix import patch_opencv_backend

from lerobot.cameras.opencv import OpenCVCameraConfig, OpenCVCamera

print("\n=== Testing LeRobot OpenCVCamera with DSHOW patch ===")

try:
    cam_cfg = OpenCVCameraConfig(
        index_or_path=0,
        width=640,
        height=480,
        fps=30,
    )
    print(f"Config created: {cam_cfg}")
    
    cam = OpenCVCamera(cam_cfg)
    print("Camera object created, attempting connect...")
    cam.connect()
    print("✅ LeRobot OpenCVCamera connected successfully!")
    
    # Try to capture a frame
    frame = cam.read()
    print(f"✅ Frame captured: shape={frame.shape}, dtype={frame.dtype}")
    
    cam.disconnect()
    print("✅ Camera disconnected cleanly")
    
    print("\n🎉 SUCCESS! The camera fix works!")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
