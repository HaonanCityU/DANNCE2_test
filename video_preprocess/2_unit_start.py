import os
import csv

def process_csv_files(current_dir,video_name,n):
    # 获取当前目录
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # 遍历当前目录下的所有文件
    for file_name in os.listdir(current_dir):
        if file_name.endswith(f'{video_name}_results.csv') and file_name.startswith('Camera'):
            file_path = os.path.join(current_dir, file_name)
            output_file_path = os.path.join(current_dir, f'processed_{file_name}')
            
            with open(file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
                csv_reader = csv.reader(infile)
                csv_writer = csv.writer(outfile)
                
                # 读取并写入表头
                header = next(csv_reader)
                csv_writer.writerow(header)
                
                # 跳过前n行数据（因为已经读取了表头）
                for _ in range(n+1):
                    try:
                        next(csv_reader)
                    except StopIteration:
                        break
                
                # 写入剩余行，并处理第二列数据
                first_row_value = None
                for i, row in enumerate(csv_reader):
                    if i == 0 and len(row) > 1:
                        # 保存第二列的第一个值作为基准
                        try:
                            first_row_value = float(row[1])
                        except ValueError:
                            first_row_value = 0  # 如果无法转换为数字，使用0作为基准
                    # 处理第二列的值
                    if len(row) > 1 and first_row_value is not None:
                        try:
                            row[1] = str(float(row[1]) - first_row_value)
                        except ValueError:
                            # 如果无法转换为数字，保持原值
                            pass
                    csv_writer.writerow(row)
            
            print(f'已处理文件: {file_name}，结果保存至: {output_file_path}')

if __name__ == '__main__':
    dir = "../sh_test2/videos"
    video_name = "0"
    n = 0  # 从第几行开始读取数据
    process_csv_files(dir, video_name, n)
