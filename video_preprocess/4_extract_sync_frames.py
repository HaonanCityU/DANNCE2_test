import cv2
import csv
import os
import subprocess
import threading
from tqdm import tqdm

def create_dir_if_not_exist(dir_path):
    """创建目录（如果不存在）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def extract_frames_ffmpeg_individual(video_path, camera_name, frame_data_list, output_root, pbar):
    """
    使用ffmpeg逐个提取帧（最准确的方法）
    
    Args:
        frame_data_list: [(csv_row, frame_num), ...] 列表
    """
    output_dir = os.path.join(output_root, camera_name)
    create_dir_if_not_exist(output_dir)
    
    for csv_row, frame_num in frame_data_list:
        # 输出文件名
        frame_filename = f"row_{csv_row:05d}_frame_{frame_num:07d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        
        # 使用ffmpeg精确提取单帧
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_path,
            '-vf', f"select='eq(n\\,{frame_num})'",
            '-frames:v', '1',
            '-update', '1',
            frame_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"⚠️  摄像头 {camera_name} 帧号 {frame_num} (CSV行{csv_row}) 提取超时")
        except subprocess.CalledProcessError:
            # 静默失败，继续处理下一帧
            pass
        
        pbar.update(1)

def extract_frames_opencv_accurate(video_path, camera_name, frame_data_list, output_root, pbar):
    """
    使用OpenCV精确提取帧（准确版：逐帧读取，定期重新打开视频以优化性能）
    
    Args:
        frame_data_list: [(csv_row, frame_num), ...] 列表，按帧号排序
    """
    output_dir = os.path.join(output_root, camera_name)
    create_dir_if_not_exist(output_dir)
    
    # 按帧号排序，便于顺序读取
    sorted_frames = sorted(frame_data_list, key=lambda x: x[1])
    
    current_frame_idx = 0  # 当前视频位置
    cap = None
    frames_processed_since_reopen = 0
    reopen_interval = 5000  # 每处理5000帧重新打开一次视频（优化性能）
    
    for csv_row, target_frame_num in sorted_frames:
        # 如果需要从头开始，或者处理了很多帧，重新打开视频（优化性能）
        if (target_frame_num < current_frame_idx or 
            frames_processed_since_reopen >= reopen_interval or
            cap is None):
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 无法打开视频：{video_path}")
                pbar.update(len(frame_data_list) - sorted_frames.index((csv_row, target_frame_num)))
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame_idx = 0
            frames_processed_since_reopen = 0
        
        # 从当前位置逐帧读取到目标帧（确保准确）
        while current_frame_idx < target_frame_num:
            ret, _ = cap.read()
            if not ret:
                # 视频结束，无法到达目标帧
                print(f"⚠️  摄像头 {camera_name} 帧号 {target_frame_num} (CSV行{csv_row}) 超出视频范围")
                pbar.update(1)
                return
            current_frame_idx += 1
        
        # 现在current_frame_idx应该等于target_frame_num，读取目标帧
        ret, frame = cap.read()
        if ret:
            frame_filename = f"row_{csv_row:05d}_frame_{target_frame_num:07d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            current_frame_idx += 1  # 移动到下一帧
            frames_processed_since_reopen += 1
        else:
            print(f"⚠️  摄像头 {camera_name} 帧号 {target_frame_num} (CSV行{csv_row}) 读取失败")
        
        pbar.update(1)
    
    if cap is not None:
        cap.release()

def extract_synced_frames(sync_csv_path, video_root, output_root, start_row, end_row, video_name):
    """基于同步 CSV 抽取指定区间的同步帧
    
    Args:
        start_row: 开始行数（CSV表头下第一行为1）。如果为None，则从第一行开始
        end_row: 结束行数（包含该行）。如果为None，则到文件末尾
        如果start_row和end_row都为None，则提取所有帧
    """
    # 1. 读取同步 CSV，截取目标区间
    sync_frame_map = {}  # key: 摄像头名称, value: 目标帧号列表
    camera_names = []    # 存储摄像头顺序（与 CSV 列一致）

    with open(sync_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # 第一行是摄像头名称（Camera1, Camera2...）
        camera_names = header

        # 初始化每个摄像头的帧号列表
        for cam in camera_names:
            sync_frame_map[cam] = []

        # 读取数据行，按起止行数截取（row_idx 从 1 开始，对应 CSV 实际行数）
        row_idx = 0
        for row in reader:
            row_idx += 1
            # 如果start_row和end_row都为None，读取所有行
            if start_row is None and end_row is None:
                # 读取所有行，不做过滤
                for cam_idx, cam_name in enumerate(camera_names):
                    frame_num = int(row[cam_idx])
                    sync_frame_map[cam_name].append((row_idx, frame_num))  # (CSV行号, 帧号)
            else:
                # 只保留 [start_row, end_row] 区间的行（闭区间）
                if start_row is not None and row_idx < start_row:
                    continue
                if end_row is not None and row_idx > end_row:
                    break
                # 把当前行的帧号存入对应摄像头（同时保存CSV行号）
                for cam_idx, cam_name in enumerate(camera_names):
                    frame_num = int(row[cam_idx])
                    sync_frame_map[cam_name].append((row_idx, frame_num))  # (CSV行号, 帧号)

    # 校验目标帧数（所有摄像头的帧号数量应一致）
    target_frame_count = len(sync_frame_map[camera_names[0]]) if camera_names else 0
    
    # 格式化输出信息
    if start_row is None and end_row is None:
        row_range_str = "全部行（从开始到结束）"
    elif start_row is None:
        row_range_str = f"第1-{end_row}行"
    elif end_row is None:
        row_range_str = f"第{start_row}行到结束"
    else:
        row_range_str = f"第{start_row}-{end_row}行"
    
    print(f"📊 目标抽取帧数：{target_frame_count} 帧（CSV {row_range_str}）")
    print(f"📹 涉及摄像头：{camera_names}")
    
    # 检查是否有重复帧号（可能会导致文件覆盖）
    print(f"\n🔍 检查CSV中的帧号重复情况：")
    for cam_name in camera_names:
        frame_data_list = sync_frame_map[cam_name]
        frame_numbers = [fn for _, fn in frame_data_list]
        unique_count = len(set(frame_numbers))
        if len(frame_numbers) != unique_count:
            duplicates = len(frame_numbers) - unique_count
            print(f"  ⚠️  {cam_name}: {duplicates} 个重复帧号（总{len(frame_numbers)}个，唯一{unique_count}个）")
        else:
            print(f"  ✅ {cam_name}: 无重复帧号")

    if target_frame_count == 0:
        print("❌ 未获取到目标帧号（可能起止行数超出 CSV 数据范围）")
        return

    # 2. 准备视频路径（假设原视频路径为：video_root/CameraX/1.mp4，可根据实际修改）
    video_paths = {}
    for cam_name in camera_names:
        video_path = os.path.join(video_root, cam_name, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            print(f"❌ 未找到视频文件：{video_path}")
            return
        video_paths[cam_name] = video_path

    # 3. 选择抽帧方法
    # 方法选择：'ffmpeg' = 使用ffmpeg逐个提取（最准确，较慢）
    #          'opencv' = 使用改进的OpenCV方法（从当前位置逐帧读取，准确且较快）
    extract_method = 'opencv'  # 可以改为 'ffmpeg' 如果需要最高准确度
    
    print(f"\n🔧 使用抽帧方法: {extract_method}")
    if extract_method == 'ffmpeg':
        print("   (ffmpeg逐个提取 - 最准确但较慢)")
    else:
        print("   (OpenCV逐帧读取 - 准确且较快)")
    
    # 4. 多线程抽取帧（每个摄像头一个线程）
    threads = []
    total_tasks = target_frame_count * len(camera_names)
    pbar = tqdm(total=total_tasks, desc="📸 抽帧进度", unit="帧")

    for cam_name in camera_names:
        frame_data_list = sync_frame_map[cam_name]
        video_path = video_paths[cam_name]
        
        # 根据选择的方法调用不同的函数
        if extract_method == 'ffmpeg':
            extract_func = extract_frames_ffmpeg_individual
            args = (video_path, cam_name, frame_data_list, output_root, pbar)
        else:
            extract_func = extract_frames_opencv_accurate
            args = (video_path, cam_name, frame_data_list, output_root, pbar)
        
        # 创建线程
        thread = threading.Thread(
            target=extract_func,
            args=args,
            name=f"Thread-{cam_name}"
        )
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()
    pbar.close()

    print(f"\n✅ 抽帧完成！所有同步帧已保存至：{output_root}")
    print(f"📋 统计：共抽取 {target_frame_count} 组同步帧，涉及 {len(camera_names)} 个摄像头")
    
    # 统计所有输出文件夹的文件数
    print(f"\n📁 各文件夹文件数统计：")
    if os.path.exists(output_root):
        # 获取所有子文件夹并排序
        subdirs = sorted([d for d in os.listdir(output_root) 
                         if os.path.isdir(os.path.join(output_root, d))])
        if subdirs:
            for subdir in subdirs:
                subdir_path = os.path.join(output_root, subdir)
                file_count = len([f for f in os.listdir(subdir_path) 
                                 if os.path.isfile(os.path.join(subdir_path, f))])
                print(f"  {subdir}: {file_count} 个文件")
        else:
            print(f"  ⚠️  输出目录下没有子文件夹")
    else:
        print(f"  ⚠️  输出目录不存在：{output_root}")

if __name__ == "__main__":
    # ===================== 直接在这里定义参数 =====================
    sync_csv_path = "/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/videos/synchronized_frames_with_offset_0.csv"  # 同步CSV文件路径
    video_root = "/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/videos"  # 原视频根目录（包含Camera1/Camera2等文件夹）
    start_row = 3600  # 开始行数（CSV表头下第一行为1，根据需求修改）
    end_row = 6000  # 结束行数（包含该行，10000帧就设为10000，根据需求修改）
    if start_row is None and end_row is None:
        output_root = f"/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/frames/test_all"
    else:
        output_root = f"/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/frames/test_{start_row}_{end_row}"  # 抽帧输出根目录（自动创建CameraX子文件夹）

    video_name = "0"
    # ==============================================================

    # 执行抽帧
    extract_synced_frames(
        sync_csv_path=sync_csv_path,
        video_root=video_root,
        output_root=output_root,
        start_row=start_row,
        end_row=end_row,
        video_name=video_name
    )