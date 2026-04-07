#!/usr/bin/env python3
"""
Create a COM .mat whose sampleID matches Label3D labelData sampleIDs.

Why:
  - Full-length COM from com-predict often uses sampleID = frame * (1000/fps) + 1
    (e.g. 1, 34, 67, ...) while Label3D labelData sampleIDs can be irregular
    timestamps (e.g. 88934, 159301, ...).
  - DANNCE training matches COM to labels by sampleID. If they don't match,
    many labeled frames get dropped.

This script builds a new COM file for training:
  - Reads labeled frames (data_frame, data_sampleID) from label3d_file (labelData)
  - Reads full-length COM positions (com) from com_full_file
  - Uses data_frame as index into full COM to pick the COM for each labeled frame
  - Saves:
      com: (N,3)
      sampleID: (1,N)  (exactly labelData data_sampleID)

Usage:
  python make_com3d_for_labeled_frames.py \
    --label3d-file /home/haonan/proj/dannce-release_dev2/demo/mouse2/mouse_dannce.mat \
    --com-full-file /home/haonan/proj/dannce-release_dev2/demo/mouse2/COM/predict_results/com3d.mat \
    --output /home/haonan/proj/dannce-release_dev2/demo/mouse2/COM/predict_results/com3d_labeled.mat
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import scipy.io as sio


def _load_label_frames_and_sampleids(repo_root: str, label3d_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      frames: (N,) int
      sampleID: (N,) int
    """
    sys.path.insert(0, repo_root)
    from dannce.engine import io  # noqa: E402

    labels = io.load_labels(label3d_file)
    df = np.asarray(labels[0]["data_frame"]).reshape(-1).astype(int)
    sid = np.asarray(labels[0]["data_sampleID"]).reshape(-1).astype(int)
    if df.size != sid.size:
        raise ValueError(f"labelData data_frame ({df.size}) != data_sampleID ({sid.size})")
    return df, sid


def _load_full_com(com_full_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      com: (M,3) float
      sampleID: (M,) int  (from file, not used for matching here)
    """
    m = sio.loadmat(com_full_file)
    keys = [k for k in m.keys() if not k.startswith("__")]
    if "com" not in m or "sampleID" not in m:
        raise KeyError(f"{com_full_file} must contain keys com + sampleID. Got keys: {keys}")
    com = np.asarray(m["com"])
    sid = np.asarray(m["sampleID"]).reshape(-1).astype(int)
    if com.ndim != 2 or com.shape[1] != 3:
        raise ValueError(f"Expected com shape (M,3), got {com.shape}")
    if sid.size != com.shape[0]:
        # tolerate mismatch if sampleID is (1,M) etc handled by reshape; otherwise error
        raise ValueError(f"sampleID length ({sid.size}) != com rows ({com.shape[0]})")
    return com, sid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None, help="Repo root for importing dannce (defaults to auto)")
    ap.add_argument("--label3d-file", required=True, help="Label3D dannce mat with labelData")
    ap.add_argument("--com-full-file", required=True, help="Full-length COM .mat (from com-predict)")
    ap.add_argument("--output", required=True, help="Output COM .mat (sampleID matches labelData)")
    args = ap.parse_args()

    repo_root = args.repo_root
    if repo_root is None:
        # script is under <repo>/Video_Preprocess/
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    frames, sid_label = _load_label_frames_and_sampleids(repo_root, args.label3d_file)
    com_full, _sid_full = _load_full_com(args.com_full_file)

    if np.any(frames < 0) or np.any(frames >= com_full.shape[0]):
        bad = frames[(frames < 0) | (frames >= com_full.shape[0])]
        raise IndexError(
            f"Some labeled frames are out of range for full COM length {com_full.shape[0]}. "
            f"Bad frame indices examples: {bad[:10]}"
        )

    com_labeled = com_full[frames, :]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sio.savemat(
        args.output,
        {
            "com": com_labeled.astype(np.float64),
            "sampleID": sid_label.reshape(1, -1).astype(np.int64),
            "metadata": {
                "label3d_file": os.path.abspath(args.label3d_file),
                "com_full_file": os.path.abspath(args.com_full_file),
                "note": "com rows indexed by labelData data_frame; sampleID copied from labelData data_sampleID",
            },
        },
        do_compression=True,
    )

    print(args.output)
    print(f"Saved com for labeled frames: N={com_labeled.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

