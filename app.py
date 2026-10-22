
import streamlit as st
import tempfile
import time
import os
from pathlib import Path

import cv2

from fall_detection import process_video, load_pose_model


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Fall Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI Person Fall Detection")
st.write(
    "Upload a video to analyze human posture and detect possible "
    "fall events using pose estimation and temporal analysis."
)

st.info(
    "This prototype uses pretrained pose-estimation weights and "
    "rule-based temporal logic. It is not a medically validated "
    "fall-detection system."
)


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------
st.sidebar.header("Detection Settings")


model_path = st.sidebar.text_input(
    "Model checkpoint",
    value="models/yolo11n_pose_pretrained.pt",
)


confidence = st.sidebar.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
)

fallen_frames = st.sidebar.slider(
    "Frames required to confirm fallen posture",
    min_value=2,
    max_value=30,
    value=5,
    step=1,
)

show_pose = st.sidebar.checkbox(
    "Display pose keypoints",
    value=True,
)

st.sidebar.caption(
    "Thresholds are heuristic. Validate them using real videos."
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------
@st.cache_resource
def get_model(checkpoint_path):
    return load_pose_model(checkpoint_path)


# --------------------------------------------------
# Video upload
# --------------------------------------------------
uploaded_video = st.file_uploader(
    "Upload a human activity video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is not None:
    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.subheader("📹 Original Video")
        st.video(uploaded_video)

    with col_annot:
        st.subheader("🛡️ Annotated Video (On-Screen View)")
        live_state_badge = st.empty()
        live_frame_view = st.empty()
        live_frame_view.info("The live annotated video feed will stream right here as frames are processed.")

    def render_state_banner(container, state):
        if state == "FALLEN":
            container.markdown(
                """
                <div style="background-color: #d32f2f; color: white; padding: 12px 18px; border-radius: 8px;
                            font-size: 26px; font-weight: 800; text-align: center; letter-spacing: 1.5px;
                            box-shadow: 0 4px 12px rgba(211,47,47,0.5); margin-bottom: 8px;">
                    🚨 ALERT: FALLEN DETECTED
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif state == "FALLING":
            container.markdown(
                """
                <div style="background-color: #f57c00; color: white; padding: 12px 18px; border-radius: 8px;
                            font-size: 24px; font-weight: 800; text-align: center; letter-spacing: 1.5px;
                            box-shadow: 0 4px 12px rgba(245,124,0,0.5); margin-bottom: 8px;">
                    ⚠️ WARNING: FALLING IN PROGRESS
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif state == "STANDING":
            container.markdown(
                """
                <div style="background-color: #2e7d32; color: white; padding: 8px 12px; border-radius: 6px;
                            font-size: 18px; font-weight: 600; text-align: center; margin-bottom: 8px;">
                    🟢 Status: STANDING
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            container.markdown(
                """
                <div style="background-color: #546e7a; color: white; padding: 8px 12px; border-radius: 6px;
                            font-size: 16px; font-weight: 500; text-align: center; margin-bottom: 8px;">
                    ⚪ Status: UNCERTAIN
                </div>
                """,
                unsafe_allow_html=True,
            )

    start_processing = st.button(
        "Analyze Video",
        type="primary",
        use_container_width=True,
    )

    if start_processing:
        temp_input_path = None
        temp_output_path = None

        try:
            # ------------------------------------------
            # Save uploaded video temporarily
            # ------------------------------------------
            suffix = Path(uploaded_video.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_input:
                temp_input.write(uploaded_video.getvalue())
                temp_input_path = temp_input.name

            output_dir = Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
                dir=str(output_dir),
            )
            temp_output_path = output_file.name
            output_file.close()

            # ------------------------------------------
            # Load model
            # ------------------------------------------
            checkpoint_file = Path(model_path)
            if not checkpoint_file.is_file():
                # Check fallback paths in project
                fallbacks = [
                    Path("models") / checkpoint_file.name,
                    Path("models") / "yolo11n_pose_pretrained.pt",
                    Path("models") / "yolo11n-pose.pt",
                    Path(checkpoint_file.name),
                    Path("yolo11n_pose_pretrained.pt"),
                ]
                for fb in fallbacks:
                    if fb.is_file():
                        checkpoint_file = fb
                        break
                else:
                    st.error(
                        f"Model checkpoint not found: {model_path}. "
                        "Place the verified .pt file at this path "
                        "or update the model path in the sidebar."
                    )
                    st.stop()

            with st.spinner("Loading pose-estimation model..."):
                model = get_model(str(checkpoint_file))

            # ------------------------------------------
            # Process video & Stream on Screen
            # ------------------------------------------
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            start_time = time.perf_counter()

            status_text.info("Analyzing video frames...")

            recorded_display_frames = []

            def update_progress(current, total):
                if total <= 0:
                    progress_bar.progress(0.0)
                    status_text.text(f"Processing frame {current}...")
                    return

                progress_ratio = min(max(float(current) / float(total), 0.0), 1.0)
                progress_bar.progress(progress_ratio)
                status_text.text(
                    f"Processing frame {current} of {total} ({int(progress_ratio * 100)}%)"
                )

            def handle_live_frame(frame_bgr, state="UNCERTAIN"):
                # Render state banner in big fonts + stream frame live
                render_state_banner(live_state_badge, state)
                live_frame_view.image(frame_bgr, channels="BGR", use_container_width=True)
                if len(recorded_display_frames) < 300:
                    recorded_display_frames.append((frame_bgr, state))

            result = process_video(
                model=model,
                input_path=temp_input_path,
                output_path=temp_output_path,
                confidence=confidence,
                fallen_frames=fallen_frames,
                show_pose=show_pose,
                progress_callback=update_progress,
                frame_callback=handle_live_frame,
            )

            elapsed = time.perf_counter() - start_time

            progress_bar.progress(1.0)
            status_text.success("Video processing completed successfully!")

            # ------------------------------------------
            # On-Screen Video Controls
            # ------------------------------------------
            if recorded_display_frames:
                st.subheader("🔍 On-Screen Video Controls")
                scrub_col, replay_col = st.columns([3, 1])
                with scrub_col:
                    scrub_idx = st.slider(
                        "Scrub through analyzed frames on screen",
                        min_value=1,
                        max_value=len(recorded_display_frames),
                        value=len(recorded_display_frames),
                    )
                    f_bgr, f_state = recorded_display_frames[scrub_idx - 1]
                    render_state_banner(live_state_badge, f_state)
                    live_frame_view.image(
                        f_bgr,
                        channels="BGR",
                        use_container_width=True,
                    )
                with replay_col:
                    st.write("")
                    st.write("")
                    if st.button("🔁 Replay on Screen", use_container_width=True):
                        for f_bgr, f_state in recorded_display_frames:
                            render_state_banner(live_state_badge, f_state)
                            live_frame_view.image(f_bgr, channels="BGR", use_container_width=True)
                            time.sleep(0.04)

            # Export / Download Option
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                with st.expander("📥 Download Annotated Video File", expanded=False):
                    with open(temp_output_path, "rb") as output_file:
                        st.download_button(
                            label="Download Annotated Video (.mp4)",
                            data=output_file.read(),
                            file_name="fall_detection_output.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )

            # ------------------------------------------
            # Display metrics
            # ------------------------------------------
            st.subheader("Analysis Summary")

            if not isinstance(result, dict):
                result = {}

            total_frames = result.get("total_frames", 0)
            counts = result.get("state_counts", {})
            fall_events = result.get("fall_events", 0)
            inference_seconds = result.get("inference_seconds")

            # Fall warning indicator
            if fall_events > 0:
                st.error(
                    f"🚨 **Fall Alert:** {fall_events} fall event(s) detected! "
                    "An observed transition from upright to fallen posture was recorded."
                )
            elif counts.get("FALLEN", 0) > 0:
                st.warning(
                    f"⚠️ **Notice:** {counts.get('FALLEN', 0)} frame(s) with horizontal/fallen posture detected, "
                    "but no upright-to-fallen transition was observed."
                )
            else:
                st.success("✅ **Status Normal:** No fall events detected in the video.")

            col1, col2, col3 = st.columns(3)

            col1.metric("Frames Processed", total_frames)
            col2.metric("Fall Events", fall_events)
            col3.metric("Total Processing Time", f"{elapsed:.2f} s")

            if inference_seconds is not None:
                st.metric(
                    "Measured Inference Time",
                    f"{inference_seconds:.3f} s",
                )

            if total_frames and elapsed > 0:
                st.metric(
                    "Average Processing FPS",
                    f"{total_frames / elapsed:.2f}",
                )

            st.subheader("Posture State Counts")

            states = ["STANDING", "FALLING", "FALLEN", "UNCERTAIN"]

            metric_cols = st.columns(4)

            for col, state_name in zip(metric_cols, states):
                col.metric(
                    state_name,
                    counts.get(state_name, 0),
                )

            st.caption(
                "A detected fallen posture is not necessarily proof "
                "that a fall occurred. Interpret results in context."
            )

        except Exception as exc:
            st.error(f"An error occurred during processing: {exc}")
            st.exception(exc)

        finally:
            # Clean up the temporary input.
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.remove(temp_input_path)
                except OSError:
                    pass

            # Keep the output video available for download/playback.
            # Do not delete temp_output_path here.
else:
    st.warning("Upload a video to begin analysis.")

st.divider()
st.caption(
    "AI Fall Detection Prototype | YOLO Pose + Temporal Analysis"
)
