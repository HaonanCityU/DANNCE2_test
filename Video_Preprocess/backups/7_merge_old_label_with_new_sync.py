"""
Merge old DANNCE labels/camera parameters with newly generated per-camera sync .mat files.

What this script does:
1. Load an existing mouse_dannce.mat (or similar) that already contains labelData/params/camnames.
2. Load new sync files from a folder, e.g. sync/Camera1_sync.mat ... Camera6_sync.mat.
3. Replace only the `sync` field in the old mat.
4. Keep `camnames`, `params`, `labelData`, and optionally `com` from the old mat.

Why:
- Useful when labels were made on the same videos, but sync needs to be regenerated.
- Video file rename (e.g. 2.mp4 -> 0.mp4) itself is not stored in the mat content,
  but regenerated sync files are.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


def _extract_camnames(camnames_cell: np.ndarray):
    """Convert MATLAB camnames cell array to Python list of strings."""
    out = []
    for i in range(camnames_cell.shape[1]):
        val = camnames_cell[0, i]
        if isinstance(val, np.ndarray):
            out.append(str(val[0]))
        else:
            out.append(str(val))
    return out


def _load_sync_as_struct(sync_mat_path: Path):
    """Load one CameraX_sync.mat and return a 1x1 MATLAB-style struct ndarray."""
    d = sio.loadmat(str(sync_mat_path), squeeze_me=False, struct_as_record=False)

    for key in ["data_frame", "data_sampleID", "data_2d", "data_3d"]:
        if key not in d:
            raise KeyError(f"{sync_mat_path} missing key: {key}")

    data_frame = d["data_frame"]
    data_sampleID = d["data_sampleID"]
    data_2d = d["data_2d"]
    data_3d = d["data_3d"]

    # Keep old mouse_dannce_tic style: data_frame as row vector (1, N)
    if isinstance(data_frame, np.ndarray) and data_frame.ndim == 2 and data_frame.shape[1] == 1:
        data_frame = data_frame.T.astype("float64")
    else:
        data_frame = np.array(data_frame, dtype="float64")

    data_sampleID = np.array(data_sampleID, dtype="float64")
    data_2d = np.array(data_2d, dtype="float64")
    data_3d = np.array(data_3d, dtype="float64")

    sync_struct = np.array(
        [(data_frame, data_sampleID, data_2d, data_3d)],
        dtype=[
            ("data_frame", "O"),
            ("data_sampleID", "O"),
            ("data_2d", "O"),
            ("data_3d", "O"),
        ],
    ).reshape(1, 1)
    return sync_struct


def _unwrap_mat_struct_like(x):
    """Unwrap nested object arrays until a scipy mat_struct-like object is found."""
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flat[0]
    return x


def _rebuild_labeldata_from_old(old_label_data: np.ndarray):
    """
    Rebuild labelData as MATLAB-style cell(nCam,1) where each cell is a 1x1 struct.
    This avoids extra object nesting that can break dannce.engine.io.load_label3d_data.
    """
    n_cams = old_label_data.shape[0]
    out_cells = np.empty((n_cams, 1), dtype=object)

    for cam_idx in range(n_cams):
        raw = old_label_data[cam_idx, 0]
        st = _unwrap_mat_struct_like(raw)
        fieldnames = getattr(st, "_fieldnames", None)
        if fieldnames is None:
            raise TypeError(
                f"labelData[{cam_idx}] is not a MATLAB struct-like object after unwrapping."
            )
        for need in ["data_2d", "data_3d", "data_frame", "data_sampleID"]:
            if need not in fieldnames:
                raise KeyError(f"labelData[{cam_idx}] missing field: {need}")

        data_2d = np.array(getattr(st, "data_2d"), dtype="float64")
        data_3d = np.array(getattr(st, "data_3d"), dtype="float64")
        data_frame = np.array(getattr(st, "data_frame"), dtype="float64")
        data_sampleID = np.array(getattr(st, "data_sampleID"), dtype="float64")

        label_struct = np.array(
            [(data_2d, data_3d, data_frame, data_sampleID)],
            dtype=[
                ("data_2d", "O"),
                ("data_3d", "O"),
                ("data_frame", "O"),
                ("data_sampleID", "O"),
            ],
        ).reshape(1, 1)
        out_cells[cam_idx, 0] = label_struct

    return out_cells


def _rebuild_params_from_old(old_params: np.ndarray):
    """
    Rebuild params as MATLAB-style cell(nCam,1) with 1x1 struct per camera.
    Expected fields: K, RDistort, TDistort, r, t
    """
    n_cams = old_params.shape[0]
    out_cells = np.empty((n_cams, 1), dtype=object)

    for cam_idx in range(n_cams):
        raw = old_params[cam_idx, 0]
        st = _unwrap_mat_struct_like(raw)
        fieldnames = getattr(st, "_fieldnames", None)
        if fieldnames is None:
            raise TypeError(
                f"params[{cam_idx}] is not a MATLAB struct-like object after unwrapping."
            )
        for need in ["K", "RDistort", "TDistort", "r", "t"]:
            if need not in fieldnames:
                raise KeyError(f"params[{cam_idx}] missing field: {need}")

        K = np.array(getattr(st, "K"), dtype="float64")
        RDistort = np.array(getattr(st, "RDistort"), dtype="float64")
        TDistort = np.array(getattr(st, "TDistort"), dtype="float64")
        r = np.array(getattr(st, "r"), dtype="float64")
        t = np.array(getattr(st, "t"), dtype="float64")

        p_struct = np.array(
            [(K, RDistort, TDistort, r, t)],
            dtype=[
                ("K", "O"),
                ("RDistort", "O"),
                ("TDistort", "O"),
                ("r", "O"),
                ("t", "O"),
            ],
        ).reshape(1, 1)
        out_cells[cam_idx, 0] = p_struct

    return out_cells


def merge_old_label_with_new_sync(old_mat_path: Path, sync_dir: Path, output_mat_path: Path):
    old = sio.loadmat(str(old_mat_path), squeeze_me=False, struct_as_record=False)

    if "camnames" not in old:
        raise KeyError(f"{old_mat_path} missing key: camnames")
    for key in ["params", "labelData"]:
        if key not in old:
            raise KeyError(f"{old_mat_path} missing key: {key}")

    camnames = _extract_camnames(old["camnames"])
    sync_cells = np.empty((len(camnames), 1), dtype=object)

    for idx, cam in enumerate(camnames):
        sync_path = sync_dir / f"{cam}_sync.mat"
        if not sync_path.exists():
            raise FileNotFoundError(f"Missing sync file: {sync_path}")
        sync_cells[idx, 0] = _load_sync_as_struct(sync_path)

    params = _rebuild_params_from_old(old["params"])
    label_data = _rebuild_labeldata_from_old(old["labelData"])

    new_data = {
        "camnames": old["camnames"],
        "params": params,
        "labelData": label_data,
        "sync": sync_cells,
    }
    if "com" in old:
        new_data["com"] = old["com"]

    output_mat_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(output_mat_path), new_data, do_compression=True)


def main():
    # --- Edit these paths ---
    old_mat_path = Path(
        "/home/haonan/proj/dannce-release_dev2/demo/backups/hanshu_20251205_exp1_rat1/backups/mouse_dannce_tic.mat"
    )
    sync_dir = Path(
        "/home/haonan/proj/dannce-release_dev2/demo/multi_tic/sync"
    )
    output_mat_path = Path(
        "/home/haonan/proj/dannce-release_dev2/demo/multi_tic/mouse_dannce_tic.mat"
    )
    # --- End ---

    print(f"Old mat: {old_mat_path}")
    print(f"Sync dir: {sync_dir}")
    print(f"Output : {output_mat_path}")

    try:
        merge_old_label_with_new_sync(old_mat_path, sync_dir, output_mat_path)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Done. New merged mat saved.")


if __name__ == "__main__":
    main()
