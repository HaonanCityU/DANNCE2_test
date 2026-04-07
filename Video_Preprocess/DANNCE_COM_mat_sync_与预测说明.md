# DANNCE / COM：`mouse_dannce.mat`、sync、训练与全片预测 — 说明备忘

本文档整理自同一项目下的讨论，便于日后查阅。路径以仓库内 `demo/` 为例，可按实际目录替换。

---

## 1. 训练时的「样本过滤」到底靠什么？

- 代码里虽有 `serve_data_DANNCE.remove_samples(..., mode="clean"|"liberal")`，**当前训练流程未调用**，不必按 clean/liberal 理解。
- **DANNCE 训练**主要读 **`labelData`**（`data_frame`、`data_sampleID`、`data_3d`、`data_2d`），样本列表来自 **`data_sampleID`** 等；损失常用 **`mask_nan_keep_loss`**，NaN 关节可被 mask，未必整帧丢弃。
- **COM 加载**阶段会调用 **`remove_samples_com`**：去掉 COM 字典里没有的 sample、COM 为 NaN 的样本；若配置了 **`cthresh`** 还会按 3D 范围再筛。
- 合并多段 mat 时注意 **`data_sampleID` 是否全局唯一**；`merge.py` 默认只做拼接，**不会**自动给每段加偏移，重复 ID 会导致字典覆盖。

---

## 2. 多相机视频：拼图、随机片段、前/后 N 帧

- **六路拼图**：`demo/NIPS/mice2/merge_6cam_grid.py`（`xstack` 2×3）。合并前若各路时长不一致，脚本会 **裁到最短公共时长并统一 fps**，避免「有的先停、有的还在播」的假不同步。
- **随机 5 秒片段**：`extract_random_5s_clips.py`；默认 **重编码 + CFR** 比纯 `-c copy` 更利于各路时长一致。
- **前 N / 后 N 帧拼图**：`first_n_frames_6cam_grid.py`；**末尾 N 帧**用 **`-sseof`** 避免对大 `start_frame` 做 trim 导致整文件解码过慢。

---

## 3. `mouse_dannce.mat`：训练是否「合格」？

可用仓库内校验脚本：

```bash
python Label3D/validate_merged_dannce_mat.py /path/to/mouse_dannce.mat
```

要点：

- **`dannce.engine.io`** 能 **`load_labels` / `load_sync` / `load_camera_params`**。
- 各路 **`labelData` 行数一致**；**Camera1 的 `data_sampleID` 无重复**。
- 曾出现 **`sync` 长度 80、`labelData` 72**：训练主路径吃 **`labelData`**，**不要求**二者等长；若希望与 Label3D 子集导出习惯完全一致，可再整理 `sync`。

**训练读的是 `labelData`，不是 `gt3d`。**  
`gt3d` / `gt3d_global` / `gt3d_local` 多为 MATLAB/Label3D 侧记录；`gt3d` 为 72 与 72 行标注一致时，仅表示导出记录一致。

---

## 4. `Label3D/merge.py` 与全局 `gt3d`

合并多个分片 `*_dannce.mat` 时，除 `labelData`、`sync` 外，若各分片均含 **`gt3d_global`**（或 **`gt3d`** 1-based），会 **纵向拼接** 并写入：

- `gt3d_global`（0-based）、`gt3d`（1-based）、`gt3d_local`（0…N−1 行序）。

若分片缺少顶层 `gt3d*`，会尝试用 **合并后 Camera1 的 `data_frame`** 推断。  
可用 **`--no-gt3d`** 关闭写入。

---

## 5. COM 配置：`crop_height` / `crop_width`

- 注释中「U-Net 维度为 32 的倍数」在 **`downfac: 2`** 下，常等价于 **`crop` 高宽宜为 64 的倍数**（因网络侧约为 `crop/downfac`）。
- **宽 1920** 可用满；**高 1080** 除以 2 得 540，**不是** 32 的倍数，可能不理想。常见做法是高度用 **1024** 或 **960** 等 ≤1080 且满足整除的取值，而不是强行 `[0,1080]`。

---

## 6. `com-train` / 配置合并报错

- **`AttributeError: 'NoneType' object has no attribute 'keys'`**：多为 **`io.yaml` 为空、路径不对，或 YAML 解析为 `null`**。应在 **含 `io.yaml` 的目录**运行，或把 **`io_config` 写成绝对路径**，或让 `io_config` 相对 **主 yaml 所在目录** 解析（若已改代码）。
- **`com-predict` + `max_num_samples: 'max'`（字符串）**：切片需要整数或 `None`；已在 `serve_data_DANNCE.prepare_data` 中将 **`'max'` 视为不截断（`None`）**。

调试输出：**`check_unrecognized_params` 内误留的 `print` 已移除**；`inherit_config` 的 fallback 提示改为 **`logging.debug`**，减少刷屏。

---

## 7. 为何 `sync` 只有 80 帧，而视频很长？

- **`demo/.../sync/` 或全量 CSV** 往往对应 **整段录制**。
- **`data_devide.m` 分片标注**时会对 **全局 `sync` 按 `gt3d` 取子集**，得到 **`sync_sub`**，故 **单个 `*_dannce.mat` 里 `sync` 可能只有几十行**。

因此：

- **`dannce-train`**：主要看 **`labelData`**，**不要求** `sync` 覆盖全片。
- **`com-predict` / `dannce-predict`**：预测步数大致由 **`sync` 行数**决定；若只有 80 行，**只会预测 80 个时间点**（除非再截断）。

---

## 8. 全片预测：`regenerate_dannce_mat.py` 与 `makeSync` 的 sync

**`regenerate_dannce_mat.py`**：

- 根据 **`--viddir` 下各 chunk 视频**统计帧数，生成 **每路相同的全局帧索引 `0 … N−1`**（与 `makeSyncFiles.py` 一类假设一致：**多相机同长、同一本地时间轴**）。
- **不会**自动读取 `sync/` 目录里已有 mat；需自行拼进 `mouse_dannce.mat` 的 **`sync` cell** 才能被 `load_sync` 使用。

**`makeSync` 生成的 `Camera*_sync.mat`**：

- 每个文件含 `data_frame`、`data_sampleID` 等，需 **按 DANNCE 期望的 cell 结构** 合并进顶层 `label3d` 文件。

若实际采集必须用 **CSV 那种「每行六路帧号不同」** 的同步，**不能**用简单的 `0…N−1` 替代，需与 **CSV / 全量对齐逻辑** 一致。

---

## 9. 命名与自动发现 `*dannce.mat`

- `processing.grab_predict_label3d_file` 会在目录里找文件名包含 **`dannce.mat`** 的文件。
- **最稳妥**：在 **`io.yaml` 的 `exp` 里显式写 `label3d_file`**，不依赖自动发现。

---

## 10. 推荐工作流（训练 vs 全片预测）

| 用途 | 建议使用的 `label3d_file` |
|------|---------------------------|
| **训练** | 带真实 **`labelData`** 的 mat（如合并后的 `mouse_dannce.mat`，子集 `sync` 亦可） |
| **全片 COM / DANNCE 预测** | 用 **`regenerate_dannce_mat.py`** 或 **全量 sync** 生成的 **`mouse_fullsync_dannce.mat`**（占位 `labelData`），并在 **`io_predict.yaml`** 中指向它 |

保持两份 **`io_train.yaml` / `io_predict.yaml`**，避免同一文件来回改。

---

## 11. 相关脚本路径（仓库内）

| 说明 | 路径 |
|------|------|
| 合并分片 mat + `gt3d` | `Label3D/merge.py` |
| 校验合并 mat | `Label3D/validate_merged_dannce_mat.py` |
| 重生成全量 sync 占位 mat | `dannce/utils/regenerate_dannce_mat.py` |
| 简易 makeSync | `dannce/utils/makeSyncFiles.py` |

---

*文档为讨论纪要，不替代官方 README；若与代码更新冲突，以当前源码为准。*
