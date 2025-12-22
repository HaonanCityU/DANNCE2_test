import os
import csv
import numpy as np
from tqdm import tqdm


def read_csv_file(file_path):
    """读取CSV文件，返回帧号和时间戳的NumPy数组（预转换为数组，避免重复操作）"""
    print(f"读取文件: {file_path}")
    frame_numbers = []
    timestamps = []
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            for row in reader:
                frame_numbers.append(int(row[0]))
                timestamps.append(float(row[1]))
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    # 转换为NumPy数组（提前转换，避免后续重复操作）

    return np.array(frame_numbers, dtype=int), np.array(timestamps, dtype=float), header


def find_closest_timestamp_fast(target_timestamp, timestamps, frame_numbers, max_allowed_diff=10.0):
    """利用二分查找快速找到最接近的时间戳（仅适用于有序时间戳）"""
    if len(timestamps) == 0:
        return None
    
    # 二分查找插入位置（利用时间戳单调递增特性）
    idx = np.searchsorted(timestamps, target_timestamp, side='left')
    
    # 候选索引：idx-1（小于等于目标）和idx（大于等于目标）
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(timestamps):
        candidates.append(idx)
    
    # 找到最小差值的索引
    min_diff = float('inf')
    best_idx = -1
    timestamp = -1
    for i in candidates:
        diff = abs(timestamps[i] - target_timestamp)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
            timestamp = timestamps[i]
    
    # 检查是否超过阈值
    if min_diff > max_allowed_diff:
        return None, None
    return frame_numbers[best_idx], timestamp


def process(current_dir, offsets, video_name):
    all_files = os.listdir(current_dir)
    print(f"当前目录: {current_dir}")
    csv_files = [f for f in all_files if f in offsets and f.endswith('.csv') and f.startswith('processed_')]

    if not csv_files:
        print("未找到在offsets字典中定义的CSV文件.")
        return

    # 选择基准文件
    base_file = csv_files[0]
    print(f"已选择 {base_file} 作为基准文件")

    offsets_copy = offsets.copy()
    print("设置的时间偏置值:")
    for file, offset in offsets_copy.items():
        print(f"{file}: {offset} 毫秒")

    # 读取所有CSV文件的数据（预转换为NumPy数组）
    data = {}
    headers = {}
    for file in csv_files:
        file_path = os.path.join(current_dir, file)
        frame_numbers, timestamps, header = read_csv_file(file_path)
        data[file] = (frame_numbers, timestamps)
        headers[file] = header

    # 读取基准文件数据（直接使用NumPy数组）
    base_frame_numbers, base_timestamps = data[base_file]
    total_frames = len(base_frame_numbers)
    if total_frames == 0:
        print("基准文件无数据，退出处理")
        return

    # 准备输出
    output_file = f'synchronized_frames_with_offsets_{video_name}.csv'
    output_file = os.path.join(current_dir, output_file)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 构建表头
        header = [base_file.split('_')[1]]  # 基准文件的帧号列
        for file in csv_files:
            if file != base_file:
                header.append(f"{file.split('_')[1]}")
        writer.writerow(header)

        successful_matches = 0

        # 逐行处理基准文件（使用优化的二分查找）
        for base_frame, base_ts in tqdm(
            zip(base_frame_numbers, base_timestamps),
            total=total_frames,
            desc="处理进度"
        ):
            row = [base_frame]
            all_found = True
            
            for file in csv_files:
                if file == base_file:
                    continue
                # 计算目标时间戳（基准时间戳 - 偏置）
                target_ts = base_ts - offsets_copy[file]
                # 获取当前文件的时间戳和帧号（已预转换为数组）
                frames, timestamps = data[file]
                # 快速查找
                closest_frame, ts = find_closest_timestamp_fast(
                    target_ts, timestamps, frames, max_allowed_diff=10.0
                )
                if closest_frame is None:
                    all_found = False
                    break
                row.append(closest_frame)
            
            if all_found:
                writer.writerow(row)
                successful_matches += 1

        # 输出匹配结果
        match_ratio = successful_matches / total_frames if total_frames > 0 else 0
        print(f"成功匹配帧数: {successful_matches}/{total_frames} ({match_ratio:.2%})")

    print(f"同步结果已保存至 {output_file}")


if __name__ == '__main__':
    print("===== 帧同步工具（优化版） =====")
    dir = "/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/videos"
    video_name = "0"

    offsets = {
        f'processed_Camera1_{video_name}_results.csv': 0,
        f'processed_Camera2_{video_name}_results.csv': -116,
        f'processed_Camera3_{video_name}_results.csv': 13,
        f'processed_Camera4_{video_name}_results.csv': 150,
        f'processed_Camera5_{video_name}_results.csv': 115,
        f'processed_Camera6_{video_name}_results.csv': 266,
    }
    
    process(dir, offsets, video_name)




