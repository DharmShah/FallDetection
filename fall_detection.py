
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


STATES = ("STANDING", "FALLING", "FALLEN", "UNCERTAIN")


def load_pose_model(checkpoint_path):
    """Load the pretrained YOLO pose-estimation model."""
    checkpoint = Path(checkpoint_path)

    if not checkpoint.exists():
        # Check common fallback locations in project
        alternatives = [
            Path("models") / checkpoint.name,
            Path("models") / "yolo11n_pose_pretrained.pt",
            Path("models") / "yolo11n-pose.pt",
            Path(checkpoint.name),
            Path("yolo11n_pose_pretrained.pt"),
            Path("yolo11n-pose.pt"),
        ]
        for alt in alternatives:
            if alt.exists():
                checkpoint = alt
                break
        else:
            raise FileNotFoundError(
                f"Model file not found: {checkpoint.resolve()}\n"
                "Please verify the model path in app.py or place the model file in 'models/'."
            )

    return YOLO(str(checkpoint))


def _notify_progress(callback, current, total):
    """
    Safely invoke progress callback with (current, total).
    Supports 1-arg callbacks (ratio) as a fallback, and prevents UI exceptions
    from crashing the core video processing loop.
    """
    if callback is None:
        return
    try:
        callback(current, total)
    except TypeError:
        try:
            ratio = (float(current) / float(total)) if total > 0 else 0.0
            callback(min(max(ratio, 0.0), 1.0))
        except Exception:
            pass
    except Exception:
        pass


def midpoint(point_a, point_b):
    """Calculate the midpoint between two points."""
    return (
        (float(point_a[0]) + float(point_b[0])) / 2.0,
        (float(point_a[1]) + float(point_b[1])) / 2.0,
    )


def analyze_posture(keypoints, keypoint_confidence, box):
    """
    Estimate posture using YOLO COCO keypoints.

    Keypoint indices:
        5, 6   = shoulders
        11, 12 = hips

    Returns:
        posture, torso_angle, bounding_box_aspect_ratio
    """
    if keypoints is None or len(keypoints) < 13:
        return "UNCERTAIN", None, None

    required_indices = [5, 6, 11, 12]

    if keypoint_confidence is not None:
        for index in required_indices:
            if (
                index >= len(keypoint_confidence)
                or float(keypoint_confidence[index]) < 0.25
            ):
                return "UNCERTAIN", None, None

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    # Approximately 0 degrees = upright torso.
    # Approximately 90 degrees = horizontal torso.
    torso_angle = float(
        np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
    )

    x1, y1, x2, y2 = [float(value) for value in box]

    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    aspect_ratio = box_width / box_height

    if torso_angle < 35 and aspect_ratio < 0.85:
        posture = "UPRIGHT"
    elif torso_angle > 55 and aspect_ratio > 0.80:
        posture = "HORIZONTAL"
    else:
        posture = "INTERMEDIATE"

    return posture, torso_angle, aspect_ratio


def process_video(
    model,
    input_path,
    output_path=None,
    confidence=0.35,
    fallen_frames=5,
    show_pose=True,
    progress_callback=None,
    frame_callback=None,
):
    """
    Process a video, optionally stream frames on screen, and save an annotated result.

    progress_callback accepts:
        progress_callback(current_frame, total_frames)
    frame_callback accepts:
        frame_callback(annotated_frame_bgr)

    Returns:
        total_frames
        state_counts
        fall_events
        inference_seconds
    """
    input_path = str(input_path)
    writer = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(input_path)

    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not np.isfinite(fps):
        fps = 25.0

    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video has invalid dimensions.")

    if output_path is not None:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not create output video: {output_path}"
            )

    state_counts = {state: 0 for state in STATES}

    frame_index = 0
    horizontal_streak = 0
    recent_upright_frame = None
    fall_events = 0
    event_already_counted = False

    recent_upright_window = max(int(fps * 2), 1)
    start_time = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_index += 1
            annotated = frame.copy()

            # Perform actual YOLO inference. No synthetic detections are used.
            results = model.predict(
                source=frame,
                conf=confidence,
                verbose=False,
            )

            current_state = "UNCERTAIN"
            selected_box = None
            selected_posture = "UNKNOWN"
            torso_angle = None

            if results:
                result = results[0]
                boxes = result.boxes
                keypoints_object = result.keypoints

                if boxes is not None and len(boxes) > 0:
                    candidates = []

                    for detection_index in range(len(boxes)):
                        class_id = int(
                            boxes.cls[detection_index].item()
                        )

                        # COCO class 0 is person.
                        if class_id != 0:
                            continue

                        coordinates = (
                            boxes.xyxy[detection_index]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        x1, y1, x2, y2 = coordinates
                        area = max(x2 - x1, 0) * max(y2 - y1, 0)
                        detection_confidence = float(
                            boxes.conf[detection_index].item()
                        )

                        candidates.append(
                            (
                                area,
                                detection_confidence,
                                detection_index,
                                coordinates,
                            )
                        )

                    if candidates:
                        # Simple primary-person strategy: choose the largest
                        # visible person in each frame.
                        candidates.sort(
                            key=lambda item: item[0],
                            reverse=True,
                        )

                        _, _, selected_index, selected_box = candidates[0]

                        points = None
                        point_confidence = None

                        if (
                            keypoints_object is not None
                            and len(keypoints_object.xy) > selected_index
                        ):
                            points = (
                                keypoints_object.xy[selected_index]
                                .detach()
                                .cpu()
                                .numpy()
                            )

                            if keypoints_object.conf is not None:
                                point_confidence = (
                                    keypoints_object.conf[selected_index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )

                        (
                            selected_posture,
                            torso_angle,
                            aspect_ratio,
                        ) = analyze_posture(
                            points,
                            point_confidence,
                            selected_box,
                        )

                        if selected_posture == "UPRIGHT":
                            horizontal_streak = 0
                            recent_upright_frame = frame_index
                            current_state = "STANDING"
                            event_already_counted = False

                        elif selected_posture == "HORIZONTAL":
                            horizontal_streak += 1

                            recently_upright = (
                                recent_upright_frame is not None
                                and frame_index - recent_upright_frame
                                <= recent_upright_window
                            )

                            if horizontal_streak >= max(
                                int(fallen_frames), 1
                            ):
                                current_state = "FALLEN"

                                # Count an event only if upright posture was
                                # observed recently before the horizontal state.
                                if (
                                    recently_upright
                                    and not event_already_counted
                                ):
                                    fall_events += 1
                                    event_already_counted = True

                            elif recently_upright:
                                current_state = "FALLING"
                            else:
                                current_state = "UNCERTAIN"

                        else:
                            horizontal_streak = 0
                            current_state = "UNCERTAIN"

                        if show_pose:
                            annotated = result.plot()

                        x1, y1, x2, y2 = [
                            int(value) for value in selected_box
                        ]

                        colors = {
                            "STANDING": (0, 180, 0),
                            "FALLING": (0, 140, 255),
                            "FALLEN": (0, 0, 230),
                            "UNCERTAIN": (150, 150, 150),
                        }

                        color = colors[current_state]

                        # Thicker bounding box for critical fall events
                        box_thickness = 4 if current_state in ("FALLING", "FALLEN") else 2

                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            box_thickness,
                        )

                        # Big font styling for FALLING and FALLEN
                        if current_state in ("FALLING", "FALLEN"):
                            font_scale = 0.95
                            font_thickness = 2
                            font_face = cv2.FONT_HERSHEY_DUPLEX
                        else:
                            font_scale = 0.65
                            font_thickness = 2
                            font_face = cv2.FONT_HERSHEY_SIMPLEX

                        label = current_state
                        if torso_angle is not None:
                            label += f" | {torso_angle:.0f} deg"

                        # Draw solid background badge behind text for crystal-clear readability
                        (label_w, label_h), baseline = cv2.getTextSize(
                            label, font_face, font_scale, font_thickness
                        )
                        label_y = max(y1 - 10, label_h + 12)
                        label_x = max(x1, 5)

                        cv2.rectangle(
                            annotated,
                            (label_x - 2, label_y - label_h - 4),
                            (label_x + label_w + 6, label_y + baseline + 2),
                            color,
                            -1,
                        )
                        cv2.putText(
                            annotated,
                            label,
                            (label_x + 2, label_y),
                            font_face,
                            font_scale,
                            (255, 255, 255),
                            font_thickness,
                            cv2.LINE_AA,
                        )
                    else:
                        horizontal_streak = 0
                else:
                    horizontal_streak = 0
            else:
                horizontal_streak = 0

            # Count one state per frame.
            state_counts[current_state] += 1

            # Prominent top status banner with BIG fonts for FALLING and FALLEN
            if current_state == "FALLEN":
                # High-visibility red alert banner across top
                cv2.rectangle(annotated, (0, 0), (width, 56), (0, 0, 210), -1)
                cv2.putText(
                    annotated,
                    "🚨 ALERT: FALLEN DETECTED!",
                    (15, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.1,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"Frame {frame_index}",
                    (max(width - 150, 10), 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif current_state == "FALLING":
                # High-visibility orange warning banner across top
                cv2.rectangle(annotated, (0, 0), (width, 56), (0, 130, 255), -1)
                cv2.putText(
                    annotated,
                    "⚠️ WARNING: FALLING IN PROGRESS!",
                    (15, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"Frame {frame_index}",
                    (max(width - 150, 10), 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                # Standard info banner
                cv2.rectangle(annotated, (0, 0), (width, 36), (40, 40, 40), -1)
                cv2.putText(
                    annotated,
                    f"Frame: {frame_index} | State: {current_state}",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if frame_callback is not None:
                try:
                    frame_callback(annotated, current_state)
                except TypeError:
                    try:
                        frame_callback(annotated)
                    except Exception:
                        pass
                except Exception:
                    pass

            if writer is not None:
                writer.write(annotated)

            # Safely report progress using (current, total)
            total_for_progress = (
                reported_frames if reported_frames > 0 else 0
            )
            _notify_progress(progress_callback, frame_index, total_for_progress)

        # Report final completion
        _notify_progress(progress_callback, frame_index, max(frame_index, 1))

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    inference_seconds = time.perf_counter() - start_time

    if frame_index == 0:
        raise ValueError("No readable frames were found in the video.")

    if output_path is not None and (not output_path.exists() or output_path.stat().st_size == 0):
        raise RuntimeError("The annotated output video was not created.")

    return {
        "total_frames": frame_index,
        "state_counts": state_counts,
        "fall_events": fall_events,
        "inference_seconds": round(inference_seconds, 3),
    }
