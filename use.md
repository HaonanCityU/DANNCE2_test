# DANNCE 使用指南

## 多视频文件预测

### 视频文件组织

多个视频文件（如 `0.mp4`, `1.mp4`）可以放在同一个相机文件夹中：

```
videos/
  Camera1/
    0.mp4
    1.mp4
  Camera2/
    0.mp4
    1.mp4
  ...
```

代码会自动扫描并识别所有视频文件，创建 chunks 字典，例如：
```python
chunks = {
    'Camera1': [0, 1],  # 0.mp4 和 1.mp4
    'Camera2': [0, 1],
    ...
}
```

### 生成 Sync 文件

当有多个视频文件时，运行 `makeSyncFiles.py` 会：
- 自动扫描每个相机文件夹中的所有视频文件（`0.mp4`, `1.mp4`, ...）
- 计算所有视频的总帧数（`0.mp4` 的帧数 + `1.mp4` 的帧数）
- 为每个相机生成一个 `CameraX_sync.mat` 文件，包含所有视频的帧信息

**命令：**
```bash
python dannce/utils/makeSyncFiles.py \
  /path/to/videos \
  60 \
  22
```

**注意：** 所有相机的视频总帧数必须相同，否则会报错。

### 预测所有视频

如果 `label3d_file` 包含所有视频的帧信息，直接运行预测命令即可：

```bash
dannce-predict /path/to/config.yaml
```

最终的预测文件（`save_data_AVG.mat` 或 `save_data_MAX.mat`）会包含所有视频的预测结果，通过 `sampleID` 区分来自哪个视频。

### 只预测特定视频（如 1.mp4）

如果已经完成了 `0.mp4` 的预测，现在只想预测 `1.mp4`，可以使用以下方法：

#### 方法 1：使用命令行参数（推荐）

1. **确定 0.mp4 的帧数：**
   ```bash
   python -c "import imageio; v = imageio.get_reader('path/to/Camera1/0.mp4'); print(f'0.mp4 has {v.count_frames()} frames'); v.close()"
   ```

2. **计算 1.mp4 的起始 sampleID：**
   - 如果 `sampleID` 是连续的帧索引：`start_sample = 0.mp4的帧数`
   - 如果 `sampleID` 基于时间戳：`start_sample = frames_0mp4 * (1000.0 / fps) + 1`

3. **运行预测：**
   ```bash
   dannce-predict /path/to/config.yaml \
     --start-sample <1.mp4的起始sampleID> \
     --max-num-samples <1.mp4的帧数>
   ```

   **示例：** 如果 `0.mp4` 有 1000 帧，`1.mp4` 有 800 帧：
   ```bash
   dannce-predict config.yaml \
     --start-sample 1000 \
     --max-num-samples 800
   ```

   或者如果不知道 `1.mp4` 的帧数，可以使用 `max`：
   ```bash
   dannce-predict config.yaml \
     --start-sample 1000 \
     --max-num-samples max
   ```

#### 方法 2：检查现有预测结果

如果已经预测了 `0.mp4`，可以查看预测文件来确定范围：

```python
import scipy.io as sio
import numpy as np

# 加载已有的预测结果
data = sio.loadmat('DANNCE/predict_results/save_data_AVG.mat')
sampleIDs = data['sampleID'].flatten()

print(f"Last sampleID from 0.mp4: {sampleIDs[-1]}")
print(f"Start sampleID for 1.mp4 should be: {sampleIDs[-1] + 1}")

# 或者查看 sync 文件
sync = sio.loadmat('sync/Camera1_sync.mat')
frames = sync['data_frame'].flatten()
print(f"Total frames in sync: {len(frames)}")
```

#### 方法 3：使用配置文件

在 `io.yaml` 或主配置文件中设置：

```yaml
# io.yaml 或 dannce_mouse_config.yaml
start_sample: 1000  # 1.mp4的起始sampleID
max_num_samples: 800  # 1.mp4的帧数
```

然后在命令行运行：
```bash
dannce-predict /path/to/config.yaml
```

### 重要提示

1. **sampleID vs data_frame：**
   - `start_sample` 和 `max_num_samples` 基于 `sampleID`，不是 `data_frame`
   - `sampleID` 通常按时间戳计算：`sampleID = frame_index * frame_period_ms + 1`
   - 如果帧率是 60 fps，`frame_period_ms = 1000/60 ≈ 16.67 ms`

2. **帧索引映射：**
   - Generator 会根据 `sampleID` 和 `chunks` 自动确定从哪个视频文件加载帧
   - 如果 `sampleID` 对应的帧在 `0.mp4` 范围内 → 从 `0.mp4` 加载
   - 如果 `sampleID` 对应的帧在 `1.mp4` 范围内 → 从 `1.mp4` 加载

3. **预测结果保存：**
   - 如果指定了 `start_sample`，结果可能保存为 `save_data_AVG<start_sample>.mat`
   - 如果没有指定，会保存为 `save_data_AVG.mat`（可能覆盖已有文件）

### 推荐工作流程

1. **首次预测所有视频：**
   ```bash
   # 生成 sync 文件（包含所有视频）
   python dannce/utils/makeSyncFiles.py videos/ 60 22
   
   # 预测所有视频
   dannce-predict config.yaml
   ```

2. **只预测新视频（如 1.mp4）：**
   ```bash
   # 检查 0.mp4 的帧数或已预测的最后一个 sampleID
   # 然后运行
   dannce-predict config.yaml \
     --start-sample <1.mp4的起始sampleID> \
     --max-num-samples <1.mp4的帧数或max>
   ```

3. **合并预测结果：**
   如果需要将多个预测结果合并，可以使用 `makeStructuredData.py` 或手动合并 `.mat` 文件。
