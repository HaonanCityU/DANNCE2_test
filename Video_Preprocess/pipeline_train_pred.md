# DANNCE/COM 全流程（视频→sync→label/gt3d→merge→COM→对齐→DANNCE 训练）

本文档把一次完整的数据处理与训练流程串起来，覆盖你提到的步骤：

1) 得到视频（多相机、分段 chunk）  
2) 生成全长 `sync`（预测用）  
3) 生成/准备 `label`（训练用 `labelData`）  
4) `gt3d`/`gt3d_local` 处理（需要 local 的场景）  
5) merge 多份 mat（可选）  
6) COM 预测（全长）  
7) 生成“与 labelData sampleID 对齐”的 COM（训练用）  
8) DANNCE 训练（含 joint training 多实验）

> 约定：下面示例以 `demo/mouse2/` 为例。请根据你的实际目录替换。

---

## 0. 目录与关键文件速查

- **视频目录**：`demo/mouse2/videos/Camera1/*.mp4 ... Camera6/*.mp4`  
  chunk 文件名一般是整数起点：`0.mp4, 92185.mp4, 191851.mp4, ...`
- **训练 mat（含 labelData）**：`demo/mouse2/mouse_dannce.mat`
- **全长预测 mat（含 sync）**：`demo/mouse2/mouse_fullsync_dannce.mat`
- **全长 COM 输出**：`demo/mouse2/COM/predict_results/com3d.mat`
- **训练用对齐 COM 输出**：`demo/mouse2/COM/predict_results/com3d_labeled.mat`（本文新增）

---

## 1. 从视频生成全长 `sync`（预测用）

目的：让 `sync` 覆盖整段视频（所有 chunk 拼起来），供 `com-predict` / `dannce-predict` 全长推理使用。

在 `demo/mouse2` 下运行：

```bash
python ../../dannce/utils/regenerate_dannce_mat.py \
  --old-mat ./mouse_dannce.mat \
  --viddir ./videos \
  --output ./mouse_fullsync_dannce.mat
```

自检（期望 `data_frame` 形状接近 `(1, N)`，且 N 为总帧数，例如 490287）：

```bash
python3 -c "import scipy.io as s, numpy as np; m=s.loadmat('mouse_fullsync_dannce.mat'); df=m['sync'][0,0]['data_frame'][0,0]; print(np.asarray(df).shape)"
```

---

## 2. 生成/准备训练用 `label`（Label3D 导出 `labelData`）

训练时 DANNCE 读的是 mat 里的 `labelData`（稀疏标注帧），不是全长 `sync`。

常见做法：

- 用 Label3D 标注并导出 `*_dannce.mat`（包含 `labelData`）
- 或用你已有脚本 merge/修复后得到 `mouse_dannce.mat`

自检（仓库内已有脚本，能检查基本结构/一致性）：

```bash
python Label3D/validate_merged_dannce_mat.py demo/mouse2/mouse_dannce.mat
```

---

## 3. `gt3d` 改成 `local`（当你需要 local 坐标时）

说明：

- DANNCE 训练主要使用 `labelData`（3D/2D + frame/sampleID）  
- `gt3d / gt3d_local / gt3d_global` 更多是 **Label3D/后处理**侧的记录

如果你的工作流需要把某份 `gt3d` 转成 local（例如以 COM 为原点/或以某参考点为原点），建议：

- 明确你的 local 定义（减去哪一个 reference，单位是否 mm）
- 对 `gt3d_local` 写入 mat，并保留原始 `gt3d_global` 以便回溯

（本仓库中与 merge/gt3d 相关逻辑见 `Label3D/merge.py`，以及校验脚本 `Label3D/validate_merged_dannce_mat.py`。）

---

## 4. merge 多份 mat（可选）

当你有多个标注 mat（多天/多段/多只鼠）需要合并训练时：

- 使用 `Label3D/merge.py` 合并 `labelData` / `gt3d*` 等
- **注意 sampleID 唯一性**：若不同来源的 `data_sampleID` 会重复，需要在 merge 前处理，否则会导致字典覆盖/样本丢失。

合并后再跑一次校验：

```bash
python Label3D/validate_merged_dannce_mat.py /path/to/merged_dannce.mat
```

---

## 5. 全长 COM 预测（com-predict）

目的：生成覆盖整段视频的 `com3d.mat`（用于全长 `dannce-predict`，也能作为训练 COM 的来源）。

关键点：

- `com-predict` 预测时读 **`sync`**（全长），所以应使用 `mouse_fullsync_dannce.mat`
- `io_predict.yaml` 里 `exp[0].label3d_file` 指向全长 mat

示例（你的主配置在 `configs/com_mouse_config.yaml`，其 `io_config` 指向 `io_predict.yaml`）：

```bash
com-predict /home/haonan/proj/dannce-release_dev2/configs/com_mouse_config.yaml
```

跑完自检（期望 `com` 行数 = 全长帧数）：

```bash
python3 -c "import scipy.io as s, numpy as np; m=s.loadmat('demo/mouse2/COM/predict_results/com3d.mat'); print(np.asarray(m['com']).shape, np.asarray(m['sampleID']).shape)"
```

---

## 6. 为什么训练时 COM 会“丢样本”（sampleID 对不上）

训练时 DANNCE 会用 `sampleID` 把 `labelData` 与 `com3d` 对齐。

常见情况：

- 全长 COM 的 `sampleID` 是均匀序列：`1, 34, 67, ...`（fps=30 对应 `frame*33.33ms + 1`）
- 但 Label3D 导出的 `labelData.data_sampleID` 可能是不规则的大整数时间戳（例如 `88934, 159301, ...`）

这会导致“交集很小”，训练时大量标注帧被过滤掉（不是分了 test，而是对不上被剔除）。

---

## 7. 生成“与 labelData sampleID 对齐”的训练 COM（推荐）

目的：为训练生成一个新的 COM 文件，使：

- `sampleID` **严格等于** `labelData.data_sampleID`（N 个标注帧）
- `com` 来自全长 COM，按 `labelData.data_frame`（帧号）索引得到对应帧的 COM

脚本（已添加到仓库）：

- `Video_Preprocess/make_com3d_for_labeled_frames.py`

示例：

```bash
/home/haonan/miniconda3/envs/dannce/bin/python \
  Video_Preprocess/make_com3d_for_labeled_frames.py \
  --label3d-file demo/mouse2/mouse_dannce.mat \
  --com-full-file demo/mouse2/COM/predict_results/com3d.mat \
  --output demo/mouse2/COM/predict_results/com3d_labeled.mat
```

然后在训练用的 io（例如 `demo/mouse2/io_train.yaml` 或 joint training 的 `demo/multi_sum/io_train.yaml`）里，把对应 experiment 的 `com_file` 指向 `com3d_labeled.mat`。

> 重要：`com3d_labeled.mat` **只用于训练（标注子集）**；全长预测仍用全长 `com3d.mat`。

---

## 8. DANNCE 训练（单实验 / 多实验）

### 8.1 单实验（mouse2）

确保 `demo/mouse2/io_train.yaml` 的 `exp[0]` 至少包含：

```yaml
exp:
  - label3d_file: ./mouse_dannce.mat
    com_file: ./COM/predict_results/com3d_labeled.mat
    viddir: ./videos/
```

然后运行：

```bash
dannce-train /home/haonan/proj/dannce-release_dev2/configs/dannce_mouse_config.yaml
```

### 8.2 多实验 joint training（mouse1 + mouse2）

在 `demo/multi_sum/io_train.yaml` 中写两个 experiment（注意 YAML 缩进要正确）：

```yaml
exp:
  - label3d_file: /abs/path/mouse1/mouse_dannce.mat
    com_file:      /abs/path/mouse1/COM/.../com3d_labeled.mat
    viddir:        /abs/path/mouse1/videos
  - label3d_file: /abs/path/mouse2/mouse_dannce.mat
    com_file:      /abs/path/mouse2/COM/.../com3d_labeled.mat
    viddir:        /abs/path/mouse2/videos
```

运行 `dannce-train` 时，确保 base config 的 `io_config` 指向该 io（或在对应目录运行并让路径正确）。

---

## 9. 检查 train/val split 是否真的来自多实验

训练会在结果目录写：

- `train_samples.pickle`
- `val_samples.pickle`

你可以用脚本（已添加）检查各 experiment 的样本数量（看 sampleID 前缀 `0_`/`1_`）：

- `Video_Preprocess/check_data.py`

示例：

```bash
python Video_Preprocess/check_data.py \
  --train-pickle demo/multi_sum/DANNCE/train_results/<RUN>/train_samples.pickle \
  --val-pickle   demo/multi_sum/DANNCE/train_results/<RUN>/val_samples.pickle \
  --io-yaml       demo/multi_sum/io_train.yaml
```

---

## 10. 常见坑总结

- **`exp:` 的 YAML 写法**：`label3d_file` 与 `viddir` 必须在同一个 `-` 条目下，否则会被解析成两个 experiment，导致 KeyError。  
- **训练用 COM 与 labelData 的 sampleID 对齐**：不对齐会导致大量样本被过滤（表现为某个 exp 的 train/val 样本数异常少）。  
- **全长预测与训练不是同一套 COM**：  
  - 全长预测：`sync` + 全长 `com3d.mat`  
  - 训练：`labelData` + `com3d_labeled.mat`（对齐 sampleID）  

