import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

BASE_DIR = "/home/haonan/proj/dannce-release_dev2/demo/NIPS/mice7/videos"
CAMERA_DIR_PREFIX = "camera"  # 匹配 Camera1/Camera2/...（大小写不敏感）
EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}  # 需要的话可增减
DRY_RUN = False  # 先 True 预览；确认无误再改为 False 真正重命名

# 帧数统计策略：
# - "metadata": 优先读 nb_frames（很快，但少数文件可能缺失/不准）
# - "count_frames": 强制逐帧统计（最准，但可能非常慢）
FRAME_COUNT_MODE = "metadata"

# ffprobe 超时（秒）。逐帧统计时很容易超时；超时会回退到其它策略/报错。
FFPROBE_TIMEOUT_S = 120


def natural_key(p: Path):
    # 优先按纯数字文件名排序：0.mp4,1.mp4,...；否则按名称自然排序
    stem = p.stem
    if re.fullmatch(r"\d+", stem):
        return (0, int(stem))
    parts = re.split(r"(\d+)", p.name)
    parts = [int(x) if x.isdigit() else x.lower() for x in parts]
    return (1, parts)


def ffprobe_nb_frames(path: Path) -> int | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found")

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.check_output(
        cmd, text=True, stderr=subprocess.STDOUT, timeout=FFPROBE_TIMEOUT_S
    ).strip()
    if not out or out.upper() == "N/A":
        return None
    try:
        n = int(out)
    except ValueError:
        return None
    return n if n > 0 else None


def ffprobe_count_frames(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found")

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.check_output(
        cmd, text=True, stderr=subprocess.STDOUT, timeout=FFPROBE_TIMEOUT_S
    ).strip()
    if not out or out.upper() == "N/A":
        raise RuntimeError("ffprobe returned empty nb_read_frames")
    n = int(out)
    if n <= 0:
        raise RuntimeError("ffprobe returned non-positive nb_read_frames")
    return n


def opencv_frames(path: Path) -> int:
    try:
        import cv2  # pip install opencv-python
    except Exception as e:
        raise RuntimeError("OpenCV not available. Install: pip install opencv-python") from e

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frames <= 0:
        raise RuntimeError("OpenCV returned non-positive frame count")
    return frames


def get_frames(path: Path) -> int:
    # 1) 快速：ffprobe 元数据 nb_frames
    if FRAME_COUNT_MODE == "metadata":
        try:
            n = ffprobe_nb_frames(path)
            if n is not None:
                return n
        except Exception:
            pass

        # 2) 回退：OpenCV 元数据
        try:
            return opencv_frames(path)
        except Exception:
            pass

        # 3) 最后：逐帧统计（可能很慢）
        return ffprobe_count_frames(path)

    # 强制逐帧统计
    if FRAME_COUNT_MODE == "count_frames":
        return ffprobe_count_frames(path)

    raise ValueError(f"Unknown FRAME_COUNT_MODE: {FRAME_COUNT_MODE}")


def process_one_camera_dir(d: Path) -> None:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if not files:
        print(f"No video files found in {d}")
        return

    files.sort(key=natural_key)

    # 计算每个视频帧数
    frame_counts = {}
    for p in files:
        n = get_frames(p)
        frame_counts[p] = n
        print(f"Frames: {n:>10}  File: {p.name}")

    # 生成目标名：累计帧数(重命名为 <cum>.ext)
    mapping = []
    cum = 0
    used_targets = set()
    for p in files:
        target = p.with_name(f"{cum}{p.suffix.lower()}")
        if target in used_targets:
            raise RuntimeError(f"Target name collision: {target.name}")
        used_targets.add(target)
        mapping.append((p, target, frame_counts[p], cum))
        cum += frame_counts[p]

    print("\nPlanned renames:")
    for src, dst, n, start in mapping:
        print(f"{src.name}  ->  {dst.name}   (start={start}, frames={n})")

    if DRY_RUN:
        print("\nDRY_RUN=True: no files renamed. Set DRY_RUN=False to apply.")
        return

    # 两阶段重命名避免冲突：先改成临时名，再改成最终名
    temp_map = []
    for src, dst, *_ in mapping:
        tmp = src.with_name(f".__tmp__{uuid.uuid4().hex}{src.suffix.lower()}")
        os.replace(src, tmp)
        temp_map.append((tmp, dst))

    for tmp, dst in temp_map:
        os.replace(tmp, dst)

    print("\nDone.")


def main():
    base = Path(BASE_DIR)
    if not base.exists():
        raise FileNotFoundError(f"BASE_DIR not found: {BASE_DIR}")
    if not base.is_dir():
        raise NotADirectoryError(f"BASE_DIR is not a directory: {BASE_DIR}")

    camera_dirs = [
        p
        for p in base.iterdir()
        if p.is_dir() and p.name.lower().startswith(CAMERA_DIR_PREFIX)
    ]
    camera_dirs.sort(key=natural_key)

    if not camera_dirs:
        print(f"No camera dirs found in {BASE_DIR} (prefix={CAMERA_DIR_PREFIX})")
        return

    for d in camera_dirs:
        print(f"\n=== Processing {d.name} ===")
        process_one_camera_dir(d)


if __name__ == "__main__":
    main()