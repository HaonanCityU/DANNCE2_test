"""
Build a multi-camera montage *video* from extracted frames (3_extract_frames.py layout).

Files are named row_XXXXX_frame_YYYYYYY.jpg (CSV row + that camera's video frame).

Selection uses the reference camera's video frame number in [start_frame, end_frame].
For each hit, the matching CSV row is used to load the same sync row in all Camera folders.

Speed notes:
  - One-time per-camera index (csv_row -> jpg name); avoids scanning huge folders on every frame.
  - Optional parallel JPEG decode across cameras (ThreadPoolExecutor).
  - Faster resize (INTER_LINEAR) and lighter text (LINE_8) by default.

Output is a single MP4. Progress uses tqdm.

Dependencies: opencv-python, numpy, tqdm

Expected layout::

    frames_root/
    ├── Camera1/
    │   ├── row_00001_frame_0001000.jpg
    │   └── ...
    ├── Camera2/
    └── ...
"""

from __future__ import annotations

import concurrent.futures
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    print("OpenCV required: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1) from e

try:
    from tqdm import tqdm
except ImportError as e:
    print("tqdm required: pip install tqdm", file=sys.stderr)
    raise SystemExit(1) from e

_ROW_FRAME_RE = re.compile(r"^row_(\d+)_frame_(\d+)\.jpg$", re.IGNORECASE)


def _parse_row_frame_jpg(name: str) -> Optional[Tuple[int, int]]:
    m = _ROW_FRAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_per_camera_row_index(
    frames_root: Path,
    num_cameras: int,
    show_progress: bool = True,
) -> List[Dict[int, str]]:
    """
    One directory listing per camera: csv_row -> one jpg basename (same rule as before: sorted first).
    """
    indexes: List[Dict[int, str]] = []
    cam_iter = range(1, num_cameras + 1)
    if show_progress:
        cam_iter = tqdm(cam_iter, desc="Index camera folders", unit="cam")

    for cam in cam_iter:
        cam_dir = frames_root / f"Camera{cam}"
        row_to_names: Dict[int, List[str]] = defaultdict(list)
        if cam_dir.is_dir():
            for f in cam_dir.iterdir():
                if not f.is_file() or f.suffix.lower() != ".jpg":
                    continue
                parsed = _parse_row_frame_jpg(f.name)
                if parsed is None:
                    continue
                csv_row, _vid = parsed
                row_to_names[csv_row].append(f.name)
        m: Dict[int, str] = {
            r: sorted(names)[0] for r, names in row_to_names.items()
        }
        indexes.append(m)
    return indexes


def map_ref_video_frames_from_index(
    row_index: List[Dict[int, str]],
    reference_camera: int,
    start_frame: int,
    end_frame: int,
    frames_root: Path,
) -> Dict[int, int]:
    """
    From the prebuilt reference-camera index (one jpg per csv_row), build
    video_frame -> csv_row for ref frames in [start_frame, end_frame].

    If multiple csv_rows map to the same ref video frame, keep the smallest csv_row.
    (Same rule as a full directory scan when there is only one file per csv_row.)
    """
    if not (1 <= reference_camera <= len(row_index)):
        raise FileNotFoundError(
            f"reference_camera {reference_camera} out of range for index length {len(row_index)}"
        )
    cam_dir = frames_root / f"Camera{reference_camera}"
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"Reference camera folder not found: {cam_dir}")

    ref_map = row_index[reference_camera - 1]
    out: Dict[int, int] = {}
    for csv_row, name in ref_map.items():
        parsed = _parse_row_frame_jpg(name)
        if parsed is None:
            continue
        _, vid_frame = parsed
        if start_frame <= vid_frame <= end_frame:
            if vid_frame not in out or csv_row < out[vid_frame]:
                out[vid_frame] = csv_row
    return out


def _imread_cam(args: Tuple[Path, Optional[str]]) -> Optional[np.ndarray]:
    cam_dir, basename = args
    if not basename:
        return None
    path = cam_dir / basename
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def build_montage_for_row_indexed(
    frames_root: Path,
    csv_row: int,
    row_index: List[Dict[int, str]],
    num_cameras: int,
    cols: int,
    rows_grid: int,
    resize_interpolation: int,
    executor: Optional[concurrent.futures.ThreadPoolExecutor],
    draw_camera_labels: bool = True,
) -> np.ndarray:
    """Build one montage using prebuilt row_index; optional parallel imread."""
    if cols * rows_grid < num_cameras:
        raise ValueError("cols * rows_grid must be >= num_cameras")

    cam_dirs = [frames_root / f"Camera{i}" for i in range(1, num_cameras + 1)]
    tasks = [
        (cam_dirs[i], row_index[i].get(csv_row)) for i in range(num_cameras)
    ]

    if executor is not None:
        tiles_bgr: List[Optional[np.ndarray]] = list(executor.map(_imread_cam, tasks))
    else:
        tiles_bgr = [_imread_cam(t) for t in tasks]

    valid = [t for t in tiles_bgr if t is not None]
    if not valid:
        raise FileNotFoundError(f"csv row {csv_row}: no jpg found for any camera")

    min_w = min(t.shape[1] for t in valid)
    min_h = min(t.shape[0] for t in valid)

    resized: List[np.ndarray] = []
    for i, t in enumerate(tiles_bgr):
        if t is None:
            tile = np.full((min_h, min_w, 3), 24, dtype=np.uint8)
        else:
            tile = cv2.resize(
                t, (min_w, min_h), interpolation=resize_interpolation
            )
        if draw_camera_labels:
            cv2.putText(
                tile,
                f"Camera{i + 1}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (80, 255, 80),
                2,
                cv2.LINE_8,
            )
        resized.append(tile)

    w, h = min_w, min_h
    row_stripes = []
    for r in range(rows_grid):
        row_tiles = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(resized):
                row_tiles.append(resized[idx])
            else:
                row_tiles.append(np.zeros((h, w, 3), dtype=np.uint8))
        row_stripes.append(np.hstack(row_tiles))
    grid = np.vstack(row_stripes)
    return grid


def main() -> None:
    # --- Edit these parameters ---
    frames_root = Path(
        "/home/haonan/proj/dannce-release_dev2/demo/hanshu_20250923_pre1/vessel_1/frames/0.mp4/all"
    )
    output_video = Path(
        "/home/haonan/proj/dannce-release_dev2/demo/hanshu_20250923_pre1/vessel_1/frames/check/montage_refCam1.mp4"
    )
    reference_camera = 1
    start_frame = 17172
    end_frame = 17472
    num_cameras = 6
    cols = 3
    rows_grid = 2
    fps = 60.0
    video_fourcc = "mp4v"
    # Speed: True = parallel JPEG decode per frame (usually faster on SSD/NVMe).
    parallel_imread = True
    imread_workers = 6
    # cv2.INTER_LINEAR is faster than INTER_AREA; use INTER_AREA if you need sharper downscale.
    resize_interpolation = cv2.INTER_LINEAR
    # Optional: scale final montage before VideoWriter (e.g. 0.5 = half size, faster encode).
    output_scale: Optional[float] = None
    draw_camera_labels = True
    # --- end ---

    if end_frame < start_frame:
        print("Error: end_frame must be >= start_frame", file=sys.stderr)
        sys.exit(2)

    try:
        row_index = build_per_camera_row_index(frames_root, num_cameras, show_progress=True)
        frame_to_csv_row = map_ref_video_frames_from_index(
            row_index,
            reference_camera,
            start_frame,
            end_frame,
            frames_root,
        )
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        sys.exit(2)

    if not frame_to_csv_row:
        print(
            f"No frames in range [{start_frame}, {end_frame}] under "
            f"Camera{reference_camera} in {frames_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    ordered = sorted(frame_to_csv_row.items(), key=lambda x: x[0])
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[cv2.VideoWriter] = None
    out_size: Optional[Tuple[int, int]] = None
    frames_written = 0

    def run_encode_loop(executor: Optional[concurrent.futures.ThreadPoolExecutor]) -> None:
        nonlocal writer, out_size, frames_written
        for vid_frame, csv_row in tqdm(
            ordered,
            desc="Montage video",
            unit="frm",
            total=len(ordered),
        ):
            try:
                grid_bgr = build_montage_for_row_indexed(
                    frames_root,
                    csv_row,
                    row_index,
                    num_cameras=num_cameras,
                    cols=cols,
                    rows_grid=rows_grid,
                    resize_interpolation=resize_interpolation,
                    executor=executor,
                    draw_camera_labels=draw_camera_labels,
                )
            except FileNotFoundError as err:
                tqdm.write(f"Skip ref_frame {vid_frame} (csv_row {csv_row}): {err}")
                continue

            if output_scale is not None and 0 < output_scale < 1.0:
                grid_bgr = cv2.resize(
                    grid_bgr,
                    (
                        max(1, int(grid_bgr.shape[1] * output_scale)),
                        max(1, int(grid_bgr.shape[0] * output_scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            h, w = grid_bgr.shape[:2]
            if writer is None:
                out_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*video_fourcc)
                writer = cv2.VideoWriter(
                    str(output_video),
                    fourcc,
                    float(fps),
                    out_size,
                )
                if not writer.isOpened():
                    print(
                        f"Failed to open VideoWriter for {output_video}",
                        file=sys.stderr,
                    )
                    sys.exit(2)

            assert out_size is not None
            if grid_bgr.shape[1] != out_size[0] or grid_bgr.shape[0] != out_size[1]:
                grid_bgr = cv2.resize(
                    grid_bgr,
                    out_size,
                    interpolation=resize_interpolation,
                )
            writer.write(grid_bgr)
            frames_written += 1

    try:
        if parallel_imread:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=imread_workers
            ) as executor:
                run_encode_loop(executor)
        else:
            run_encode_loop(None)
    finally:
        if writer is not None:
            writer.release()

    if frames_written == 0:
        if output_video.is_file():
            try:
                output_video.unlink()
            except OSError:
                pass
        print("No frames written; removed empty output if present.", file=sys.stderr)
        sys.exit(1)

    print(f"Done: {frames_written} frame(s) -> {output_video.resolve()} @ {fps} fps")


if __name__ == "__main__":
    main()
