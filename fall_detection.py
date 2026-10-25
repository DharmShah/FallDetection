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


class VirtualLineCounter:
    """
    Virtual vertical line-crossing people counter.
    Tracks people across a vertical virtual line with hysteresis deadband
    to prevent duplicate counts caused by jitter, detection loss, or standing near the line.
    """

    def __init__(
        self,
        line_ratio=0.5,
        initial_count=0,
        allow_negative=False,
        deadband_pixels=25,
    ):
        self.line_ratio = float(line_ratio)
        self.right_side_count = int(initial_count)
        self.allow_negative = bool(allow_negative)
        self.deadband_pixels = int(deadband_pixels)

        self.total_left_to_right = 0
        self.total_right_to_left = 0

        # Persistent track state: track_id -> "LEFT" or "RIGHT"
        self.established_sides = {}
        # Persistent positions: track_id -> (cx, cy)
        self.last_positions = {}
        # Active crossing alerts for on-screen direction indicators: track_id -> dict
        self.active_cross_alerts = {}

    def get_line_x(self, frame_width):
        return int(frame_width * self.line_ratio)

    def get_deadband(self, frame_width):
        return max(self.deadband_pixels, int(frame_width * 0.025))

    def update(self, frame_width, frame_index, detections):
        """
        Update tracking and detect crossings.
        detections: list of tuples (track_id, (x1, y1, x2, y2))
        """
        line_x = self.get_line_x(frame_width)
        margin = self.get_deadband(frame_width)

        new_crossings = []

        for track_id, box in detections:
            if track_id is None:
                continue

            x1, y1, x2, y2 = box
            cx = (float(x1) + float(x2)) / 2.0
            cy = float(y2)
            self.last_positions[track_id] = (cx, cy)

            # Determine spatial zone relative to the line and deadband
            if cx < line_x - margin:
                zone = "LEFT"
            elif cx > line_x + margin:
                zone = "RIGHT"
            else:
                zone = "DEADBAND"

            prev_side = self.established_sides.get(track_id)

            if prev_side is None:
                # First time tracking this ID
                if zone in ("LEFT", "RIGHT"):
                    self.established_sides[track_id] = zone
                else:
                    self.established_sides[track_id] = "LEFT" if cx < line_x else "RIGHT"
            elif prev_side == "LEFT" and zone == "RIGHT":
                # Genuine Left to Right crossing
                self.total_left_to_right += 1
                self.right_side_count += 1
                self.established_sides[track_id] = "RIGHT"
                event = {
                    "track_id": track_id,
                    "direction": "L->R",
                    "frame": frame_index,
                    "point": (cx, cy),
                }
                self.active_cross_alerts[track_id] = event
                new_crossings.append(event)
            elif prev_side == "RIGHT" and zone == "LEFT":
                # Genuine Right to Left crossing
                self.total_right_to_left += 1
                self.right_side_count -= 1
                if not self.allow_negative:
                    self.right_side_count = max(0, self.right_side_count)
                self.established_sides[track_id] = "LEFT"
                event = {
                    "track_id": track_id,
                    "direction": "R->L",
                    "frame": frame_index,
                    "point": (cx, cy),
                }
                self.active_cross_alerts[track_id] = event
                new_crossings.append(event)

        # Clear expired crossing alerts older than 25 frames
        expired = [
            tid
            for tid, ev in self.active_cross_alerts.items()
            if frame_index - ev["frame"] > 25
        ]
        for tid in expired:
            del self.active_cross_alerts[tid]

        return new_crossings

    def draw_overlay(self, frame, frame_width, frame_height, has_fall_alert=False):
        """Draw the virtual line, deadband boundaries, and people counter HUD."""
        line_x = self.get_line_x(frame_width)
        margin = self.get_deadband(frame_width)

        # Draw virtual vertical line
        cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 255, 255), 2, cv2.LINE_AA)

        # Draw subtle deadband buffer lines
        cv2.line(frame, (line_x - margin, 0), (line_x - margin, frame_height), (80, 140, 140), 1, cv2.LINE_AA)
        cv2.line(frame, (line_x + margin, 0), (line_x + margin, frame_height), (80, 140, 140), 1, cv2.LINE_AA)

        hud_y = 62 if has_fall_alert else 42

        # Virtual line marker badge
        label_text = f"LINE {int(self.line_ratio * 100)}%"
        cv2.rectangle(frame, (line_x - 42, hud_y), (line_x + 42, hud_y + 18), (0, 200, 200), -1)
        cv2.putText(frame, label_text, (line_x - 38, hud_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # People Counter HUD on top right
        hud_w, hud_h = 230, 75
        hud_x = max(frame_width - hud_w - 15, 10)

        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"RIGHT-SIDE: {self.right_side_count}",
            (hud_x + 10, hud_y + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"IN  (L->R): {self.total_left_to_right}",
            (hud_x + 10, hud_y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"OUT (R->L): {self.total_right_to_left}",
            (hud_x + 10, hud_y + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )


def process_video(
    model,
    input_path,
    output_path=None,
    confidence=0.35,
    fallen_frames=5,
    show_pose=True,
    progress_callback=None,
    frame_callback=None,
    enable_line_counter=True,
    line_ratio=0.5,
    initial_count=0,
    allow_negative=False,
):
    """
    Process a video, perform pose estimation, temporal fall detection,
    and virtual line-crossing people counting.

    progress_callback accepts:
        progress_callback(current_frame, total_frames)
    frame_callback accepts:
        frame_callback(annotated_frame_bgr, current_state)

    Returns:
        total_frames
        state_counts
        fall_events
        inference_seconds
        line_crossing: {
            "right_side_count": int,
            "left_to_right": int,
            "right_to_left": int,
        }
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

    counter = None
    if enable_line_counter:
        counter = VirtualLineCounter(
            line_ratio=line_ratio,
            initial_count=initial_count,
            allow_negative=allow_negative,
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

            # Perform YOLO tracking with ByteTrack to assign persistent IDs
            try:
                results = model.track(
                    source=frame,
                    conf=confidence,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            except Exception:
                results = model.predict(
                    source=frame,
                    conf=confidence,
                    verbose=False,
                )

            current_state = "UNCERTAIN"
            selected_box = None
            selected_track_id = None
            selected_posture = "UNKNOWN"
            torso_angle = None

            all_person_detections = []

            if results:
                result = results[0]
                boxes = result.boxes
                keypoints_object = result.keypoints

                if boxes is not None and len(boxes) > 0:
                    for detection_index in range(len(boxes)):
                        class_id = int(boxes.cls[detection_index].item())

                        # COCO class 0 is person
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

                        track_id = None
                        if (
                            boxes.id is not None
                            and detection_index < len(boxes.id)
                            and boxes.id[detection_index] is not None
                        ):
                            try:
                                track_id = int(boxes.id[detection_index].item())
                            except Exception:
                                track_id = None

                        all_person_detections.append(
                            (
                                area,
                                detection_confidence,
                                detection_index,
                                coordinates,
                                track_id,
                            )
                        )

                    # Update Virtual Line-Crossing Counter with all detected/tracked people
                    if counter is not None and all_person_detections:
                        counter_inputs = [
                            (item[4], item[3]) for item in all_person_detections
                        ]
                        counter.update(width, frame_index, counter_inputs)

                    if all_person_detections:
                        # Primary-person strategy for fall detection: largest visible person
                        all_person_detections.sort(
                            key=lambda item: item[0],
                            reverse=True,
                        )

                        (
                            _,
                            _,
                            selected_index,
                            selected_box,
                            selected_track_id,
                        ) = all_person_detections[0]

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

                        # 1. Draw non-primary tracked people with bounding boxes and IDs
                        for (
                            _,
                            _,
                            idx,
                            coords,
                            tid,
                        ) in all_person_detections:
                            bx1, by1, bx2, by2 = [int(v) for v in coords]
                            bcx = int((bx1 + bx2) / 2.0)
                            bcy = int(by2)

                            # Bottom-center tracking marker
                            cv2.circle(
                                annotated, (bcx, bcy), 5, (0, 255, 255), -1
                            )

                            # Direction indicator if a crossing just occurred
                            if (
                                counter is not None
                                and tid in counter.active_cross_alerts
                            ):
                                ev = counter.active_cross_alerts[tid]
                                arrow_text = (
                                    "--> L to R (+1)"
                                    if ev["direction"] == "L->R"
                                    else "<-- R to L (-1)"
                                )
                                arrow_color = (
                                    (0, 255, 0)
                                    if ev["direction"] == "L->R"
                                    else (0, 140, 255)
                                )
                                cv2.putText(
                                    annotated,
                                    arrow_text,
                                    (max(bx1, 5), max(by1 - 35, 25)),
                                    cv2.FONT_HERSHEY_DUPLEX,
                                    0.65,
                                    arrow_color,
                                    2,
                                    cv2.LINE_AA,
                                )

                            if idx != selected_index:
                                cv2.rectangle(
                                    annotated,
                                    (bx1, by1),
                                    (bx2, by2),
                                    (200, 200, 0),
                                    2,
                                )
                                id_label = (
                                    f"ID: {tid}"
                                    if tid is not None
                                    else "Person"
                                )
                                cv2.putText(
                                    annotated,
                                    id_label,
                                    (bx1 + 2, max(by1 - 6, 16)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.52,
                                    (200, 200, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

                        # 2. Draw primary fall-detection person box and label
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
                        box_thickness = (
                            4 if current_state in ("FALLING", "FALLEN") else 2
                        )

                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            box_thickness,
                        )

                        if current_state in ("FALLING", "FALLEN"):
                            font_scale = 0.95
                            font_thickness = 2
                            font_face = cv2.FONT_HERSHEY_DUPLEX
                        else:
                            font_scale = 0.65
                            font_thickness = 2
                            font_face = cv2.FONT_HERSHEY_SIMPLEX

                        id_prefix = (
                            f"ID: {selected_track_id} | "
                            if selected_track_id is not None
                            else ""
                        )
                        label = f"{id_prefix}{current_state}"
                        if torso_angle is not None:
                            label += f" | {torso_angle:.0f} deg"

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

            # Count one posture state per frame
            state_counts[current_state] += 1

            has_fall_alert = current_state in ("FALLING", "FALLEN")

            # Top alert banner for fall states
            if current_state == "FALLEN":
                cv2.rectangle(annotated, (0, 0), (width, 56), (0, 0, 210), -1)
                cv2.putText(
                    annotated,
                    "ALERT: FALLEN DETECTED!",
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
                cv2.rectangle(annotated, (0, 0), (width, 56), (0, 130, 255), -1)
                cv2.putText(
                    annotated,
                    "WARNING: FALLING IN PROGRESS!",
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

            # Draw virtual line, deadband, and People Counter HUD
            if counter is not None:
                counter.draw_overlay(
                    annotated, width, height, has_fall_alert=has_fall_alert
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

            total_for_progress = (
                reported_frames if reported_frames > 0 else 0
            )
            _notify_progress(progress_callback, frame_index, total_for_progress)

        _notify_progress(progress_callback, frame_index, max(frame_index, 1))

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    inference_seconds = time.perf_counter() - start_time

    if frame_index == 0:
        raise ValueError("No readable frames were found in the video.")

    if output_path is not None and (
        not output_path.exists() or output_path.stat().st_size == 0
    ):
        raise RuntimeError("The annotated output video was not created.")

    return {
        "total_frames": frame_index,
        "state_counts": state_counts,
        "fall_events": fall_events,
        "inference_seconds": round(inference_seconds, 3),
        "line_crossing": {
            "right_side_count": (
                counter.right_side_count if counter is not None else initial_count
            ),
            "left_to_right": (
                counter.total_left_to_right if counter is not None else 0
            ),
            "right_to_left": (
                counter.total_right_to_left if counter is not None else 0
            ),
        },
    }
