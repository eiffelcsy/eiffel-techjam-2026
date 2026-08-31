"""Minimal Streamlit dashboard for scripts/main/predict.py.

Run from the repo root:

    streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import torch
from torch.utils.data import DataLoader

from eval.config import load_detector_config
from eval.detectors import build_detector
from preprocessing.dataset import ImageFolderDataset, collate

DETECTOR_DIR = ROOT / "eval" / "configs" / "detectors"
DETECTOR_CONFIGS = sorted(str(p) for p in DETECTOR_DIR.glob("*.yaml"))


@st.cache_resource(show_spinner="Loading detector (first run may download weights)...")
def load_detector(cfg_path: str, device: str):
    cfg = load_detector_config(cfg_path)
    cfg.device = device
    detector = build_detector(cfg)
    n_params = sum(p.numel() for p in detector.parameters())
    return detector, n_params


def score_dir(detector, image_dir, batch_size: int = 16, num_workers: int = 0):
    dataset = ImageFolderDataset(
        image_dir, preprocess=detector.preprocess_fn(), aux=detector.aux_fn()
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=collate,
    )
    device = next(detector.parameters()).device
    rows = []
    with torch.no_grad():
        for batch, metas in loader:
            preds = detector.score(batch.to(device)).float().cpu().numpy()
            rows.extend(
                (Path(m["image_path"]).name, float(p)) for m, p in zip(metas, preds)
            )
    return rows


def main():
    st.set_page_config(page_title="RESONANCE", page_icon="🔍")
    st.title("RESONANCE — AI-generated image detector")

    with st.sidebar:
        st.header("Model")
        cfg_labels = {Path(p).name: p for p in DETECTOR_CONFIGS}
        cfg_path = cfg_labels[st.selectbox(
            "Detector", sorted(cfg_labels),
            index=sorted(cfg_labels).index("dinov3-crop200+grace.yaml")
            if "dinov3-crop200+grace.yaml" in cfg_labels else 0,
        )]
        device = st.selectbox("Device", ["auto", "cpu", "cuda"])
        batch_size = st.slider("Batch size", 1, 64, 16)
        detector, n_params = load_detector(cfg_path, device)
        st.metric("Parameters (end-to-end)", f"{n_params:,}")

    source = st.radio("Images from", ["Upload", "Folder path"])
    image_dir = None
    if source == "Upload":
        uploads = st.file_uploader(
            "Upload one or more images", type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
        )
        if uploads:
            image_dir = Path(tempfile.mkdtemp(prefix="st_upload_"))
            for u in uploads:
                (image_dir / u.name).write_bytes(u.getvalue())
    else:
        path = st.text_input("Folder path", placeholder="/path/to/images")
        if path:
            image_dir = Path(path).expanduser()

    if st.button("Run inference", type="primary", disabled=image_dir is None):
        if image_dir is None:
            st.warning("Provide images to score.")
            return
        try:
            detector, _ = load_detector(cfg_path, device)
            with st.spinner("Scoring..."):
                rows = score_dir(detector, image_dir, batch_size=batch_size)
        except Exception as e:
            st.error(f"Failed: {e}")
            return

        st.subheader("Results")
        if not rows:
            st.info("No images found.")
            return
        st.dataframe(
            [
                {"image": name, "P(AI-generated)": p, "verdict": "AI" if p > 0.5 else "Real"}
                for name, p in rows
            ],
            hide_index=True,
        )
        for name, p in rows:
            st.caption(f"**{name}**")
            st.progress(p, text=f"P(AI-generated) = {p:.3f}")


if __name__ == "__main__":
    main()
