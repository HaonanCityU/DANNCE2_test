import os
import csv
import numpy as np
from tqdm import tqdm


def read_csv_file(file_path):
    """读取CSV文件，返回帧号和时间戳的NumPy数组"""
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
        return None, None, None
    
    return np.array(frame_numbers, dtype=int), np.array(timestamps, dtype=float), header


def find_closest_timestamp_fast(target_timestamp, timestamps, frame_numbers, max_allowed_diff=10.0):
    """利用二分查找快速找到最接近的时间戳（仅适用于有序时间戳）"""
    if len(timestamps) == 0:
        return None, None
    
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


def process_with_sync_points(current_dir, sync_points, video_name, base_camera=1):
    """
    基于已知同步帧点来同步多个视频的帧
    该方法从预处理文件中读取时间戳（cap_msec，相对于第一帧的时间差），
    然后从同步帧点计算时间偏移量，使用时间戳匹配来同步
    
    Args:
        current_dir: CSV文件所在目录
        sync_points: 字典，格式为 {camera_num: sync_frame_num}
                    例如: {1: 1000, 2: 1050, 3: 980, 4: 1100, 5: 1020, 6: 1080}
                    表示Camera1的第1000帧、Camera2的第1050帧等是同步的
        video_name: 视频名称（用于输出文件名）
        base_camera: 基准摄像头编号（默认为1）
    """
    all_files = os.listdir(current_dir)
    print(f"当前目录: {current_dir}")
    
    # 构建CSV文件名列表
    csv_files = {}
    for cam_num in sync_points.keys():
        file_name = f'Camera{cam_num}_{video_name}_results.csv'
        if file_name in all_files:
            csv_files[cam_num] = file_name
        else:
            print(f"警告：未找到文件 {file_name}")
    
    if not csv_files:
        print("未找到任何CSV文件.")
        return
    
    if base_camera not in csv_files:
        print(f"错误：基准摄像头 {base_camera} 的文件不存在")
        return
    
    print(f"已选择 Camera{base_camera} 作为基准摄像头")
    print(f"同步帧点信息:")
    for cam_num, sync_frame in sync_points.items():
        print(f"  Camera{cam_num}: 第 {sync_frame} 帧")
    
    # 读取所有CSV文件的数据（包含帧号和时间戳）
    data = {}
    headers = {}
    for cam_num, file_name in csv_files.items():
        file_path = os.path.join(current_dir, file_name)
        frame_numbers, timestamps, header = read_csv_file(file_path)
        if frame_numbers is None:
            print(f"跳过文件 {file_name}")
            continue
        data[cam_num] = (frame_numbers, timestamps)
        headers[cam_num] = header
    
    if base_camera not in data:
        print("基准摄像头数据读取失败")
        return
    
    # 从同步帧点读取对应的时间戳，并计算时间偏移量
    base_sync_frame = sync_points[base_camera]
    base_frames, base_timestamps_data = data[base_camera]
    
    # 找到基准摄像头同步帧对应的时间戳
    base_sync_idx = np.where(base_frames == base_sync_frame)[0]
    if len(base_sync_idx) == 0:
        print(f"错误：在基准摄像头中未找到同步帧 {base_sync_frame}")
        return
    base_sync_ts = base_timestamps_data[base_sync_idx[0]]
    print(f"基准摄像头同步帧 {base_sync_frame} 对应的时间戳: {base_sync_ts} 毫秒")
    
    # 计算每个摄像头相对于基准的时间偏移量（毫秒）
    time_offsets = {}
    time_offsets[base_camera] = 0.0
    
    for cam_num, sync_frame in sync_points.items():
        if cam_num == base_camera:
            continue
        
        frames, timestamps = data[cam_num]
        sync_idx = np.where(frames == sync_frame)[0]
        if len(sync_idx) == 0:
            print(f"警告：在Camera{cam_num}中未找到同步帧 {sync_frame}，跳过")
            continue
        
        sync_ts = timestamps[sync_idx[0]]
        # 时间偏移量 = 基准同步时间戳 - 当前摄像头同步时间戳
        # 如果Camera2的同步帧时间戳比基准大，说明Camera2延迟了，偏移量为负
        time_offset = base_sync_ts - sync_ts
        time_offsets[cam_num] = time_offset
        print(f"  Camera{cam_num} 同步帧 {sync_frame} 对应时间戳: {sync_ts} 毫秒")
        print(f"  Camera{cam_num} 相对于基准的时间偏移: {time_offset:.2f} 毫秒")
    
    # 读取基准文件数据
    base_frame_numbers, base_timestamps = data[base_camera]
    total_frames = len(base_frame_numbers)
    if total_frames == 0:
        print("基准文件无数据，退出处理")
        return
    
    # 准备输出
    output_file = f'synchronized_frames_with_offset_{video_name}.csv'
    output_file = os.path.join(current_dir, output_file)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 构建表头
        header = [f'Camera{base_camera}']
        for cam_num in sorted(csv_files.keys()):
            if cam_num != base_camera:
                header.append(f"Camera{cam_num}")
        writer.writerow(header)
        
        successful_matches = 0
        
        # 逐行处理基准文件（使用时间戳匹配，类似原代码）
        for base_frame, base_ts in tqdm(
            zip(base_frame_numbers, base_timestamps),
            total=total_frames,
            desc="处理进度"
        ):
            row = [base_frame]
            all_found = True
            
            for cam_num in sorted(csv_files.keys()):
                if cam_num == base_camera:
                    continue
                
                # 计算目标时间戳（基准时间戳 - 时间偏移量）
                target_ts = base_ts - time_offsets.get(cam_num, 0.0)
                
                # 获取当前文件的时间戳和帧号
                frames, timestamps = data[cam_num]
                
                # 快速查找最接近的时间戳
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
    print("===== 基于同步帧点的帧同步工具 =====")
    dir = "/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/videos"
    video_name = "0"
    
    # 定义同步帧点：每个摄像头在哪个帧号时是同步的
    # 格式：{摄像头编号: 同步帧号}
    # 例如：如果Camera1的第1000帧、Camera2的第1050帧、Camera3的第980帧等是同步的
    sync_points = {
        1: 315,  # Camera1的第1000帧
        2: 298,  # Camera2的第1050帧（与Camera1的第1000帧同步）
        3: 287,   # Camera3的第980帧（与Camera1的第1000帧同步）
        4: 292,  # Camera4的第1100帧（与Camera1的第1000帧同步）
        5: 297,  # Camera5的第1020帧（与Camera1的第1000帧同步）
        6: 289,  # Camera6的第1080帧（与Camera1的第1000帧同步）
    }
    
    # 基准摄像头（默认为Camera1）
    base_camera = 1
    
    process_with_sync_points(dir, sync_points, video_name, base_camera)

