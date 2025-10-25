"""Deep diagnostic to understand lerobot's OpenCV camera initialization."""
import cv2
import numpy as np

print("=== Testing different OpenCV initialization patterns ===\n")

# Test 1: Basic VideoCapture (what works)
print("1. Basic VideoCapture(0):")
cap = cv2.VideoCapture(0)
print(f"   isOpened: {cap.isOpened()}")
if cap.isOpened():
    cap.release()

# Test 2: With DSHOW backend (what also works)
print("\n2. VideoCapture(0, CAP_DSHOW):")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"   isOpened: {cap.isOpened()}")
if cap.isOpened():
    cap.release()

# Test 3: Set properties AFTER opening (mimicking lerobot's approach)
print("\n3. Set properties after opening:")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    print(f"   Initial isOpened: True")
    
    # Try setting properties like lerobot might
    ret1 = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    ret2 = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret3 = cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"   Set width: {ret1}")
    print(f"   Set height: {ret2}")
    print(f"   Set FPS: {ret3}")
    
    # Verify actual values
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Actual dimensions: {actual_w}x{actual_h} @ {actual_fps} fps")
    
    # Try reading a frame
    ret, frame = cap.read()
    print(f"   Can read frame: {ret}, shape={frame.shape if ret else 'N/A'}")
    
    cap.release()
else:
    print("   Failed to open")

# Test 4: Check what backend lerobot might be using
print("\n4. Check available backends:")
backends = []
for backend in dir(cv2):
    if backend.startswith('CAP_'):
        backends.append(backend)
print(f"   Found {len(backends)} backends")

# Test 5: Try opening without backend specification (lerobot might not specify it)
print("\n5. VideoCapture(0) without backend, then set properties:")
cap = cv2.VideoCapture(0)  # No backend specified
if cap.isOpened():
    print(f"   Opened: True")
    backend_name = cap.getBackendName()
    print(f"   Backend: {backend_name}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    ret, frame = cap.read()
    print(f"   Can read: {ret}")
    cap.release()