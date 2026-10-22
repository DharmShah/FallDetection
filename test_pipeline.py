"""
Verification test script for Fall Detection pipeline and callbacks.
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# 1. Test syntax / imports
print("Step 1: Testing imports...")
try:
    from fall_detection import load_pose_model, process_video, analyze_posture, STATES
    print("  SUCCESS: fall_detection module imported successfully.")
except Exception as e:
    print(f"  FAILED: Import error: {e}")
    sys.exit(1)

# 2. Test model loading
print("\nStep 2: Testing model loading...")
checkpoint = "models/yolo11n_pose_pretrained.pt"
try:
    model = load_pose_model(checkpoint)
    print(f"  SUCCESS: Model loaded successfully from {checkpoint}.")
    print(f"  Task: {model.task}, Classes: {model.names}")
except Exception as e:
    print(f"  FAILED: Model loading error: {e}")
    sys.exit(1)

# 3. Test progress callback contract
print("\nStep 3: Testing progress callback contract...")
callback_2arg_calls = []
def callback_2arg(current, total):
    callback_2arg_calls.append((current, total))

callback_1arg_calls = []
def callback_1arg(fraction):
    callback_1arg_calls.append(fraction)

# Create a sample test video with 10 frames
test_input_path = "test_sample_video.mp4"
test_output_path = "outputs/test_pipeline_output.mp4"

frame_h, frame_w = 480, 640
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(test_input_path, fourcc, 10.0, (frame_w, frame_h))

# Draw synthetic human silhouette or basic frame
for i in range(10):
    img = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    cv2.putText(img, f"Test Frame {i+1}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    writer.write(img)
writer.release()

print(f"  Created test video: {test_input_path} (10 frames)")

# Test process_video with 2-arg callback
try:
    result_2arg = process_video(
        model=model,
        input_path=test_input_path,
        output_path=test_output_path,
        confidence=0.35,
        fallen_frames=3,
        show_pose=True,
        progress_callback=callback_2arg,
    )
    print("  SUCCESS: process_video completed with 2-argument callback.")
    print("  Result dictionary:", result_2arg)
    print(f"  Callback called {len(callback_2arg_calls)} times. First: {callback_2arg_calls[0]}, Last: {callback_2arg_calls[-1]}")
    
    # Assertions
    assert result_2arg["total_frames"] == 10, f"Expected 10 frames, got {result_2arg['total_frames']}"
    assert sum(result_2arg["state_counts"].values()) == 10, "Sum of state counts must equal total_frames"
    for state in STATES:
        assert state in result_2arg["state_counts"], f"Missing state {state}"
    assert Path(test_output_path).exists() and Path(test_output_path).stat().st_size > 0, "Output video missing or empty"
    print("  SUCCESS: All return dictionary assertions passed!")
except Exception as e:
    print(f"  FAILED: Error during 2-arg callback test: {e}")
    sys.exit(1)

# Test process_video with 1-arg callback
test_output_1arg = "outputs/test_1arg_output.mp4"
try:
    result_1arg = process_video(
        model=model,
        input_path=test_input_path,
        output_path=test_output_1arg,
        confidence=0.35,
        fallen_frames=3,
        show_pose=False,
        progress_callback=callback_1arg,
    )
    print("  SUCCESS: process_video completed with 1-argument callback fallback.")
    print(f"  Callback called {len(callback_1arg_calls)} times. Last ratio: {callback_1arg_calls[-1]}")
except Exception as e:
    print(f"  FAILED: Error during 1-arg callback test: {e}")
    sys.exit(1)

# Clean up temporary test files
if os.path.exists(test_input_path):
    os.remove(test_input_path)
if os.path.exists(test_output_path):
    os.remove(test_output_path)
if os.path.exists(test_output_1arg):
    os.remove(test_output_1arg)

print("\nStep 4: Compiling app.py syntax check...")
import py_compile
py_compile.compile("app.py", doraise=True)
py_compile.compile("fall_detection.py", doraise=True)
print("  SUCCESS: Both app.py and fall_detection.py compiled without syntax errors!")

print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
