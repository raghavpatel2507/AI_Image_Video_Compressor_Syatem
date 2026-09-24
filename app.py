"""
app.py
Streamlit demo UI with session_state caching and robust exception handling.
"""

import tempfile
from pathlib import Path
import streamlit as st

from src.baseline import baseline_compress_image, baseline_compress_video
from src.quality_metrics import evaluate_image_quality, evaluate_video_quality, VMAF_SUPPORTED
from src.pipeline import compress_image_adaptive, compress_video_adaptive

st.set_page_config(page_title="AI Media Compression System", layout="wide", page_icon="⚡")


def format_size(size_bytes: int) -> str:
    """Formats file size into readable MB or KB."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / 1024:.1f} KB"


st.title("⚡ AI-Based Image & Video Compression System")
st.markdown(
    "Upload any image or video below. Our **AI Machine Learning Model** will analyze the media content "
    "(motion, texture, resolution) to pick the best compression settings — giving you **maximum file size reduction** "
    "while maintaining **high visual quality**!"
)

if not VMAF_SUPPORTED:
    st.caption("ℹ️ *Video quality is evaluated using SSIM/PSNR standard visual similarity metrics.*")

uploaded = st.file_uploader(
    "📁 Select or drop an Image (.jpg, .png) or Video (.mp4, .mov):",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "mov", "mkv", "avi"],
)

if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)

    # Check if we already processed this file to avoid re-running compression on download click
    if st.session_state.get("file_id") != file_id:
        try:
            suffix = Path(uploaded.name).suffix.lower()
            is_video = suffix in {".mp4", ".mov", ".mkv", ".avi"}

            cache_dir = Path(tempfile.gettempdir()) / "ai_compression_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            in_path = cache_dir / f"orig_{uploaded.name}"
            in_path.write_bytes(uploaded.getvalue())

            with st.spinner("🔄 Running Traditional Fixed Compression..."):
                if is_video:
                    base_out = cache_dir / f"baseline_{uploaded.name}"
                    base_comp = baseline_compress_video(str(in_path), str(base_out))
                    base_qual = evaluate_video_quality(str(in_path), str(base_out))
                else:
                    base_out = cache_dir / f"baseline_{Path(uploaded.name).stem}.webp"
                    base_comp = baseline_compress_image(str(in_path), str(base_out))
                    base_qual = evaluate_image_quality(str(in_path), str(base_out))

            with st.spinner("🤖 AI Machine Learning Compression in progress..."):
                if is_video:
                    ai_out = cache_dir / f"ai_{uploaded.name}"
                    ai_result = compress_video_adaptive(str(in_path), str(ai_out))
                else:
                    ai_out = cache_dir / f"ai_{Path(uploaded.name).stem}.webp"
                    ai_result = compress_image_adaptive(str(in_path), str(ai_out))

            # Store in session state so re-runs (like clicking download) don't re-compress
            st.session_state["file_id"] = file_id
            st.session_state["is_video"] = is_video
            st.session_state["suffix"] = suffix
            st.session_state["in_path"] = str(in_path)
            st.session_state["in_size"] = in_path.stat().st_size
            st.session_state["base_out"] = str(base_out)
            st.session_state["base_comp"] = base_comp
            st.session_state["base_qual"] = base_qual
            st.session_state["ai_out"] = str(ai_out)
            st.session_state["ai_result"] = ai_result
            st.session_state["ai_bytes"] = Path(ai_out).read_bytes()

        except Exception as e:
            st.session_state.pop("file_id", None)
            st.error(f"❌ **Error processing file ({uploaded.name}):** {str(e)}")
            st.warning("⚠️ Please verify that your system has FFmpeg installed and that the uploaded file is valid.")
            st.stop()

    # Load from session_state (instant loading on re-runs)
    if "ai_result" in st.session_state:
        is_video = st.session_state["is_video"]
        suffix = st.session_state["suffix"]
        in_path = st.session_state["in_path"]
        in_size = st.session_state["in_size"]
        base_comp = st.session_state["base_comp"]
        base_qual = st.session_state["base_qual"]
        ai_out = st.session_state["ai_out"]
        ai_result = st.session_state["ai_result"]
        ai_bytes = st.session_state["ai_bytes"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Original Media")
            if is_video:
                st.video(in_path)
            else:
                st.image(in_path)
            st.metric("Original File Size", format_size(in_size))

        with col2:
            st.subheader("✨ AI-Compressed Result")
            if is_video:
                st.video(ai_out)
            else:
                st.image(ai_out)

            st.metric(
                label="AI Compressed File Size",
                value=format_size(ai_result["compressed_size_bytes"]),
                delta=f"-{ai_result['size_reduction_pct']}% Storage Saved",
            )

            download_name = f"ai_compressed_{Path(uploaded.name).stem}" + (".webp" if not is_video else suffix)
            mime_type = "image/webp" if not is_video else f"video/{suffix.lstrip('.')}"

            st.download_button(
                label="⬇️ Download AI-Compressed File",
                data=ai_bytes,
                file_name=download_name,
                mime=mime_type,
                type="primary",
                use_container_width=True,
            )

        st.divider()
        st.subheader("📊 Comparison: Traditional Method vs AI Smart Method")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            label="Traditional Size (Fixed)",
            value=format_size(base_comp["output_size_bytes"]),
            help="File size produced by standard fixed compression settings.",
        )
        m2.metric(
            label="AI Compressed Size",
            value=format_size(ai_result["compressed_size_bytes"]),
            help="File size produced by our content-aware AI model.",
        )
        m3.metric(
            label="Traditional Quality (SSIM)",
            value=f"{base_qual['ssim']:.4f}",
            help="Visual similarity score (1.0 = identical to original).",
        )
        m4.metric(
            label="AI Quality (SSIM)",
            value=f"{ai_result['ssim']:.4f}",
            help="Target quality maintained by AI (0.95+ is excellent visual quality).",
        )

        source_display = (
            "🧠 Trained Machine Learning Model (RandomForest)"
            if ai_result["param_source"] == "ml_model"
            else "💡 Rule-Based Heuristic Strategy"
        )
        st.success(f"**Decision Engine:** {source_display}")

        if is_video and ai_result.get("vmaf") is not None:
            v1, v2 = st.columns(2)
            v1.metric(
                label="Traditional VMAF Score (0-100)",
                value=f"{base_qual.get('vmaf'):.2f}" if base_qual.get("vmaf") is not None else "N/A",
                help="Netflix perceptual VMAF score for traditional method.",
            )
            v2.metric(
                label="AI VMAF Score (0-100)",
                value=f"{ai_result['vmaf']:.2f}",
                help="Netflix perceptual VMAF score for AI method (90+ is excellent).",
            )

        st.info(
            f"⏱️ **Processing Speed:** {ai_result['processing_time_sec']} seconds  |  "
            f"🔄 **Optimization Retries:** {ai_result['iterations']} step(s)"
        )

        with st.expander("🔍 AI Content Analysis (What features the AI analyzed)"):
            st.write(
                "Below are the exact visual properties extracted from your media by OpenCV that fed the ML model decision:"
            )
            st.json(ai_result["features"])

        with st.expander("📈 Step-by-Step Optimization History"):
            st.write("Shows how the AI checked quality and fine-tuned settings:")
            st.table(ai_result["history"])
else:
    # Clear session state if file is removed
    st.session_state.pop("file_id", None)
    st.info("💡 Upload an image or video file above to see the AI compression results in real-time!")
