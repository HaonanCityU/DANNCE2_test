"""
Add 3D COM predictions into a Label3D/*dannce.mat file.

This is useful when you already ran COM prediction (e.g. produced com3d0.mat)
and want to embed the COM into the Label3D mat so downstream steps can load it
from the label3d file directly.

Expected COM file layouts:
- MATLAB struct `com` with fields `com3d` and `sampleID`
- or top-level keys `com3d` and `sampleID`
- or (common in this repo) top-level keys `com` (Nx3) and `sampleID`
"""

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np
import scipy.io as sio

try:
    import mat73
except Exception:  # pragma: no cover
    mat73 = None

from dannce.engine import io as dannce_io


def _load_label3d_mat(path: str) -> Dict[str, Any]:
    """
    Load Label3D mat. Prefer scipy (v7.2 and earlier); fall back to mat73 (v7.3).
    """
    try:
        return sio.loadmat(path)
    except NotImplementedError as e:
        if mat73 is None:
            raise e
        return mat73.loadmat(path)


def _extract_com_from_mat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (com3d, sampleID) from a COM mat using dannce's tolerant loader.
    """
    d = dannce_io.load_com(path)
    com3d = np.asarray(d["com3d"])
    sampleID = np.asarray(d["sampleID"]).reshape(-1)
    return com3d, sampleID


def add_com_to_label3d(label3d_file: str, com_file: str, out_file: str) -> None:
    rr = _load_label3d_mat(label3d_file)
    com3d, sampleID = _extract_com_from_mat(com_file)

    com_struct: Dict[str, Any] = {
        "com3d": com3d,
        "sampleID": sampleID,
    }

    rr["com"] = com_struct

    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Note: this writes a MATLAB v5/v7.2 style .mat (scipy savemat).
    # If your original label3d_file was v7.3, this will produce a new non-v7.3 output.
    sio.savemat(out_file, rr, do_compression=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Embed COM (com3d/sampleID) into a Label3D/*dannce.mat file."
    )
    ap.add_argument(
        "--label3d-file",
        required=True,
        help="Path to Label3D/*dannce.mat (e.g. mouse_dannce.mat).",
    )
    ap.add_argument(
        "--com-file",
        required=True,
        help="Path to COM .mat (e.g. COM/predict_results/com3d0.mat).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output .mat path. If omitted, overwrites --label3d-file in place.",
    )
    args = ap.parse_args()

    out_file = args.out if args.out is not None else args.label3d_file
    add_com_to_label3d(args.label3d_file, args.com_file, out_file)
    print(f"Wrote COM into: {out_file}")


if __name__ == "__main__":
    main()

