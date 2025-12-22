import cv2
import os
import sys
from tqdm import tqdm

def frames_to_video(frame_dir, output_path, start_frame, end_frame, fps=60):
    """
    将指定范围内的图片组合成视频
    
    Args:
        frame_dir: 图片所在目录（包含 row_XXXXX_frame_YYYYYYY.jpg 格式的文件）
        output_path: 输出视频路径
        start_frame: 起始CSV行号（文件名中的第一个数字）
        end_frame: 结束CSV行号（文件名中的第一个数字）
        fps: 输出视频的帧率
    """
    # 扫描目录中所有符合条件的图片文件
    # 文件名格式：row_08000_frame_008445.jpg
    print(f"正在扫描 {frame_dir} 中的图片文件...")
    valid_frame_files = []  # 存储完整的文件名
    
    for filename in os.listdir(frame_dir):
        if filename.endswith('.jpg') and filename.startswith('row_'):
            try:
                # 解析文件名：row_08000_frame_008445.jpg
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) >= 4 and parts[0] == 'row' and parts[2] == 'frame':
                    csv_row = int(parts[1])  # CSV行号
                    frame_num = int(parts[3])  # 帧号
                    # 检查是否在指定范围内（根据CSV行号或帧号判断，这里用CSV行号）
                    if start_frame <= csv_row <= end_frame:
                        valid_frame_files.append((csv_row, frame_num, filename))
            except (ValueError, IndexError):
                continue
    
    # 按CSV行号排序
    valid_frame_files.sort(key=lambda x: x[0])
    
    if not valid_frame_files:
        print(f"错误：在 {frame_dir} 中找不到 {start_frame}~{end_frame} 范围内的有效图片")
        return False
    
    print(f"找到 {len(valid_frame_files)} 个有效图片帧")
    
    # 获取第一张图片以确定尺寸
    first_frame_path = os.path.join(frame_dir, valid_frame_files[0][2])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape
    
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"错误：无法创建视频文件 {output_path}")
        return False
    
    # 处理每一帧
    print(f"开始处理 {frame_dir} 中的有效帧")
    success_count = 0
    
    with tqdm(total=len(valid_frame_files), desc=f"生成视频 {os.path.basename(output_path)}") as pbar:
        for csv_row, frame_num, filename in valid_frame_files:
            # 使用完整的文件名
            frame_path = os.path.join(frame_dir, filename)
            
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
                success_count += 1
            else:
                print(f"警告：无法读取帧 {frame_path}")
            
            pbar.update(1)
    
    # 释放资源
    out.release()
    
    if success_count == 0:
        print(f"错误：没有成功处理任何帧")
        # 删除空文件
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    
    print(f"成功生成视频：{output_path}")
    print(f"成功处理 {success_count} 帧")
    return True

def process_cameras(frames_root, output_root, start_frame, end_frame, fps=60, output_name="sync.mp4"):
    """
    处理所有Camera文件夹
    """
    # 确保输出根目录存在
    os.makedirs(output_root, exist_ok=True)
    
    # 处理Camera1到Camera6
    for camera_num in range(1, 7):
        camera_name = f"Camera{camera_num}"
        
        # 输入帧目录
        frame_dir = os.path.join(frames_root, camera_name)
        if not os.path.isdir(frame_dir):
            print(f"警告：找不到目录 {frame_dir}")
            continue
        
        # 输出视频目录
        output_dir = os.path.join(output_root, camera_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 输出视频路径
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\n处理 {camera_name}...")
        success = frames_to_video(frame_dir, output_path, start_frame, end_frame, fps)
        
        if success:
            print(f"{camera_name} 处理完成！")
        else:
            print(f"{camera_name} 处理失败！")

def main():
    # 配置参数
    FRAMES_ROOT = "/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/videos/frames/test_all"
    OUTPUT_ROOT = "/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/videos"
    START_FRAME = 3593
    END_FRAME = 7972
    FPS = 60
    output_name = "calibration.mp4"
    
    # 打印配置信息
    print("===== 图片转视频工具 =====")
    print(f"帧目录：{FRAMES_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"帧范围：{START_FRAME}~{END_FRAME}")
    print(f"输出帧率：{FPS} fps")
    print(f"输出文件名：{output_name}")
    print("==========================")
    
    # 开始处理
    process_cameras(FRAMES_ROOT, OUTPUT_ROOT, START_FRAME, END_FRAME, FPS, output_name)
    
    print("\n所有处理完成！")

if __name__ == "__main__":
    main()