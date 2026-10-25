
import streamlit as st
import tempfile
import time
import os
import subprocess
from pathlib import Path

import cv2

from fall_detection import process_video, load_pose_model


def prepare_downloadable_video(input_video_path: str) -> bytes:
    """
    Ensure the annotated video is encoded in standard H.264 MP4 format with yuv420p
    for universal browser and video-player playback, returning raw bytes for download.
    Falls back to direct binary reading if ffmpeg conversion fails.
    """
    output_h264 = str(Path(input_video_path).with_suffix(".h264.mp4"))
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_video_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "fast",
            output_h264,
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if res.returncode == 0 and os.path.exists(output_h264) and os.path.getsize(output_h264) > 0:
            with open(output_h264, "rb") as f:
                data = f.read()
            try:
                os.remove(output_h264)
            except OSError:
                pass
            return data
    except Exception:
        pass

    with open(input_video_path, "rb") as f:
        return f.read()


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

st.sidebar.markdown("---")
st.sidebar.header("Virtual Line Counter")

enable_line_counter = st.sidebar.checkbox(
    "Enable line-crossing people counter",
    value=True,
)

line_position_pct = st.sidebar.slider(
    "Virtual line position (% of width)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
)

initial_count = st.sidebar.number_input(
    "Initial right-side count",
    min_value=0,
    value=0,
    step=1,
)

allow_negative = st.sidebar.checkbox(
    "Allow negative count",
    value=False,
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
    # Track video identity in session_state to persist results across reruns
    video_key = f"{uploaded_video.name}_{uploaded_video.size}"
    if st.session_state.get("active_video_key") != video_key:
        st.session_state["active_video_key"] = video_key
        st.session_state["analysis_result"] = None
        st.session_state["annotated_video_bytes"] = None
        st.session_state["annotated_video_filename"] = f"annotated_{Path(uploaded_video.name).stem}.mp4"
        st.session_state["recorded_display_frames"] = []
        st.session_state["analysis_elapsed"] = 0.0

    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.subheader("📹 Original Video")
        st.video(uploaded_video)

    with col_annot:
        st.subheader("🛡️ Annotated Video (On-Screen View)")
        live_state_badge = st.empty()
        live_frame_view = st.empty()

        # If already analyzed, show immediate download button directly in video column
        if st.session_state.get("annotated_video_bytes") is not None:
            st.download_button(
                label="📥 Download Annotated Video (.mp4)",
                data=st.session_state["annotated_video_bytes"],
                file_name=st.session_state.get("annotated_video_filename", "annotated_video.mp4"),
                mime="video/mp4",
                type="primary",
                use_container_width=True,
                key="download_btn_col_annot",
            )
        else:
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
                enable_line_counter=enable_line_counter,
                line_ratio=line_position_pct / 100.0,
                initial_count=initial_count,
                allow_negative=allow_negative,
            )

            elapsed = time.perf_counter() - start_time
            progress_bar.progress(1.0)
            status_text.success("Video processing completed successfully!")

            # Prepare downloadable video bytes (encoded via H.264 when ffmpeg is available)
            with st.spinner("Preparing downloadable video format..."):
                video_bytes = prepare_downloadable_video(temp_output_path)

            # Store results in session state so user interactions (scrubbing, replay) don't wipe them
            st.session_state["analysis_result"] = result
            st.session_state["annotated_video_bytes"] = video_bytes
            st.session_state["annotated_video_filename"] = f"annotated_{Path(uploaded_video.name).stem}.mp4"
            st.session_state["recorded_display_frames"] = recorded_display_frames
            st.session_state["analysis_elapsed"] = elapsed

            # Clean up temp file
            if os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except OSError:
                    pass

            st.rerun()

        except Exception as exc:
            st.error(f"An error occurred during processing: {exc}")
            st.exception(exc)

        finally:
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.remove(temp_input_path)
                except OSError:
                    pass

    # --------------------------------------------------
    # Persistent Results & Download View
    # --------------------------------------------------
    if st.session_state.get("analysis_result") is not None:
        result = st.session_state["analysis_result"]
        video_bytes = st.session_state.get("annotated_video_bytes")
        video_filename = st.session_state.get("annotated_video_filename", "annotated_video.mp4")
        elapsed = st.session_state.get("analysis_elapsed", 0.0)
        frames_list = st.session_state.get("recorded_display_frames", [])

        # Prominent Download Action Banner
        if video_bytes:
            st.markdown("---")
            st.markdown(
                """
                <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px 24px; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #38bdf8;">📥 Download Annotated Video</h3>
                    <p style="color: #94a3b8; margin-bottom: 12px;">
                        The full video with bounding boxes, tracking IDs, posture labels, fall alert banners, and people counter HUD has been rendered and is ready for export.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            dl_col1, dl_col2 = st.columns([2, 1])
            with dl_col1:
                size_mb = len(video_bytes) / (1024 * 1024)
                st.write(f"📁 **Filename:** `{video_filename}`")
                st.write(f"📊 **File Size:** `{size_mb:.2f} MB` | **Format:** MP4 (Universal H.264)")
            with dl_col2:
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=video_bytes,
                    file_name=video_filename,
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                    key="download_btn_action_card",
                )

        # ------------------------------------------
        # On-Screen Video Controls
        # ------------------------------------------
        if frames_list:
            st.subheader("🔍 On-Screen Video Controls")
            scrub_col, replay_col = st.columns([3, 1])
            with scrub_col:
                scrub_idx = st.slider(
                    "Scrub through analyzed frames on screen",
                    min_value=1,
                    max_value=len(frames_list),
                    value=len(frames_list),
                    key="scrub_slider",
                )
                f_bgr, f_state = frames_list[scrub_idx - 1]
                render_state_banner(live_state_badge, f_state)
                live_frame_view.image(
                    f_bgr,
                    channels="BGR",
                    use_container_width=True,
                )
            with replay_col:
                st.write("")
                st.write("")
                if st.button("🔁 Replay on Screen", use_container_width=True, key="replay_button"):
                    for f_bgr, f_state in frames_list:
                        render_state_banner(live_state_badge, f_state)
                        live_frame_view.image(f_bgr, channels="BGR", use_container_width=True)
                        time.sleep(0.04)

        # ------------------------------------------
        # Display metrics
        # ------------------------------------------
        st.subheader("Analysis Summary")

        if not isinstance(result, dict):
            result = {}

        if enable_line_counter:
            st.subheader("👥 People Counter (Right-Side)")
            line_data = result.get("line_crossing", {})
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Current Right-Side People", line_data.get("right_side_count", 0))
            p_col2.metric("Total In (Left -> Right)", line_data.get("left_to_right", 0))
            p_col3.metric("Total Out (Right -> Left)", line_data.get("right_to_left", 0))

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

else:
    st.warning("Upload a video to begin analysis.")

st.divider()
st.caption(
    "AI Fall Detection Prototype | YOLO Pose + Temporal Analysis"
)

