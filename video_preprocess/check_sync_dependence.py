# this script is used to check the dependence of the synchronized frames
import csv
col = 3   # Camera4
with open('/home/haonan/proj/dannce/demo/vessel_4/videos/synchronized_frames_with_offsets_0.csv', encoding='utf-8') as f:
    reader = list(csv.reader(f))[1:]  # 跳过表头
vals = [int(row[col]) for row in reader[8000:18001]]  # 注意18002,因slice上界不含
print('总行数:', len(vals))   # 10002
print('唯一帧数:', len(set(vals)))  # 8172（应等于输出张数）