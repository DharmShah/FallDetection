# 🛡️ AI Person Fall Detection Application

A Streamlit-based computer vision application for detecting human posture and analyzing fall events in videos using YOLO11 Pose estimation and temporal state classification.

---

## 📌 Architecture Overview

This project separates **visual feature extraction** from **temporal state classification**:

1. **Pose Estimation Model (`models/yolo11n_pose_pretrained.pt`)**:
   - Uses the pretrained Ultralytics YOLO11 Nano Pose model (`yolo11n-pose`).
   - Extracts 17 COCO human body keypoints (shoulders, hips, knees, ankles, etc.) and bounding boxes.
   - **Important Note:** The `.pt` file contains weights strictly for person pose detection. It is **not** a custom-trained end-to-end fall classifier.

2. **Temporal State Classifier (`fall_detection.py`)**:
   - Implements rule-based temporal posture analysis in Python.
   - Computes torso angle from shoulder-to-hip vector (`arctan2(abs(dx), abs(dy))`) and bounding box aspect ratio ($W/H$).
   - Maintains temporal tracking across consecutive frames to distinguish an active fall transition from static horizontal postures.

---

## 🏷️ State Definitions

- **`STANDING`**: Person is upright (`torso_angle < 35°` and `aspect_ratio < 0.85`).
- **`FALLING`**: Person is transitioning from an upright posture toward horizontal within a temporal sliding window ($2 \times \text{FPS}$).
- **`FALLEN`**: Person maintains a horizontal posture (`torso_angle > 55°` and `aspect_ratio > 0.80`) for at least the configured threshold of consecutive frames (default: 5 frames).
- **`UNCERTAIN`**: Keypoints are occluded, confidence is below threshold (< 0.25), no person is detected, or the posture is intermediate without a clear transition.

> **Fall Transition Rule:** A fall event is only recorded when an observed transition occurs from a recent `STANDING` state into `FALLING` and sustained `FALLEN`. If a person is lying down from the beginning without an observed fall transition, no fall event is triggered.

---

## 📁 Project Directory Structure

```text
FallDetection/
├── app.py                         # Streamlit web interface
├── fall_detection.py              # Pose inference & temporal state classification logic
├── requirements.txt               # Project Python dependencies
├── README.md                      # Documentation and usage guide
├── models/
│   ├── yolo11n_pose_pretrained.pt # Pretrained YOLO11 pose checkpoint
│   └── yolo11n-pose.pt            # Compatibility symlink/copy
└── outputs/                       # Destination folder for processed annotated videos
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10 – 3.13
- Virtual environment recommended

### 2. Activate Virtual Environment (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
```
*(Or create a new environment: `python -m venv venv`)*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 💻 Running the Application

Launch the Streamlit web dashboard:

```powershell
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Usage Steps:
1. Upload a video file (`.mp4`, `.avi`, `.mov`, `.mkv`).
2. Adjust detection settings in the sidebar if needed:
   - **Detection confidence**: Confidence threshold for YOLO person detection (default: 0.35).
   - **Fallen frames threshold**: Minimum consecutive horizontal frames required to declare `FALLEN` (default: 5).
   - **Display pose keypoints**: Toggle skeleton rendering overlay.
3. Click **"Analyze Video"**.
4. View the real-time progress bar, original video preview, annotated output video, state counts, measured inference metrics, and download the annotated video.

---

## ⚠️ System Limitations & Custom Training

- **Lighting & Camera Angle:** Works best with overhead or wall-mounted camera views where torso and hip keypoints are visible.
- **Occlusions:** When keypoints are obscured, detections safely fall back to `UNCERTAIN`.
- **Pretrained Pose vs. Custom Classifier:** The current system uses heuristic temporal thresholds on top of pose keypoints. To train a custom real-world fall classification model:
  - Collect a labeled dataset with real temporal sequences (e.g., UR Fall Detection Dataset, Multiple Cameras Fall Dataset, Le2i Fall Detection Dataset).
  - Extract temporal keypoint sequences or 3D bounding boxes.
  - Train an action-recognition model (e.g., ST-GCN, PoseConv3D, or LSTM) on real-world fall sequences.
