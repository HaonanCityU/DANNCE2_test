import cv2
import csv
import os
import threading
from tqdm import tqdm

def process_camera_folder(folder_path, root_path, video):
    """处理单个摄像头文件夹中的所有视频"""
    folder_name = os.path.basename(folder_path)
    # 遍历文件夹中的视频文件
    for video_file in os.listdir(folder_path):
        if video_file.endswith(f'{video}.mp4'):
            video_path = os.path.join(folder_path, video_file)
            # print(f'线程 {threading.current_thread().name} 处理视频: {video_path}')
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f'无法打开视频: {video_path}')
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            # print(f'视频总帧数: {frame_count}, 帧率: {fps:.2f}')

            # 生成CSV文件名
            video_name = os.path.splitext(video_file)[0]
            csv_filename = f"{folder_name}_{video_name}_results.csv"
            csv_path = os.path.join(root_path, csv_filename)

            # 创建进度条（总帧数包含第0帧，所以+1）
            pbar = tqdm(total=frame_count, desc=f'{folder_name}_{video_name}', unit='帧')

            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['frame_number', 'cap_msec'])
                

                # 第二步：从第1帧开始读取视频并记录
                frame_number = 0  # 从1开始计数
                while frame_number < frame_count:  # 总帧数不变，避免多写
                    # 读取当前帧（第frame_number帧）
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 获取当前帧的时间戳
                    cap_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    csv_writer.writerow([frame_number, round(cap_msec)])

                    pbar.update(1)
                    pbar.set_postfix({
                        '当前帧': frame_number, 
                        '进度': f'{((frame_number + 1)/frame_count)*100:.2f}%'  # +1是因为包含第0帧
                    })

                    frame_number += 1

            pbar.close()
            cap.release()
            # print(f'CSV文件已保存到: {csv_path}')

def process_videos(root_path, video_name):
    """多线程处理根目录下的所有摄像头文件夹"""
    threads = []
    # 遍历根文件夹下的所有子文件夹
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        # 只处理以Camera开头的文件夹
        if os.path.isdir(folder_path) and folder_name.startswith('Camera'):
            # 为每个摄像头文件夹创建一个线程
            thread = threading.Thread(
                target=process_camera_folder,
                args=(folder_path, root_path, video_name),
                name=folder_name 
            )
            threads.append(thread)
            thread.start()
            print(f'启动线程处理 {folder_name}')

    # 等待所有线程完成
    for thread in threads:
        thread.join()
    print(f'根目录 {root_path} 下所有视频处理完成!')

if __name__ == "__main__":
    for root_path in ["/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/videos"]:
        video_name = '0'
        process_videos(root_path, video_name)
    print('所有根目录处理完成!')