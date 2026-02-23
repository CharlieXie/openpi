# Waypoint VLA — openpi 实现设计文档

> 最后更新: 2026-02-22 (rev 4 — LIBERO 评测管线完善)
> 基于 Pi0.5 (PyTorch) 在 openpi 项目中实现两段式 Waypoint VLA 系统

---

## 一、整体架构

```
VLA = VLM (自回归预测稀疏 waypoint 轨迹) + Action Expert (填充 waypoint 间动作)
```

### 两段式推理流程

```
1. 获取当前观测 (images, proprio)
2. VLM 自回归生成 M=7 个 waypoint: [(proprio_1, d_1), ..., (proprio_7, d_7)]
   - Constrained decoding: <wp>/<dur> 强制注入，模型只预测 proprio tokens 和 duration tokens
3. 对每个 waypoint pair (WP_i-1, WP_i):
   a. d_i = 0  → 停止 (episode 结束)
   b. d_i ≤ 32 → Action Expert 推理:
      - 输入: images, instruction, start_proprio, end_proprio, duration
      - 输出: 32 步 action chunk (取前 d_i 步执行)
4. 执行完所有 WP 后，用新观测重新调用 VLM (循环重规划)
```

---

## 二、文件结构

```
openpi/
├── src/openpi/waypoint/
│   ├── __init__.py
│   ├── robot_config.py   # Robot 配置 (cameras, dims, gripper norm)
│   ├── normalize.py      # q99/z-score 归一化工具
│   ├── tokenizer.py      # ProprioTokenizer + WaypointTokenizer
│   ├── ae_dataset.py     # WaypointAEDataset (RLDS → AE training samples)
│   ├── ae_model.py       # PI0WaypointAE (flow matching + duration AdaRMSNorm)
│   ├── vlm_dataset.py    # WaypointVLMDataset (RLDS → VLM token sequences)
│   ├── vlm_model.py      # PI0WaypointVLM (PaliGemma AR + CE loss)
│   └── eval_libero.py    # 两段式 LIBERO 评测管线
├── scripts/
│   └── train_waypoint.py # 统一训练入口 (--mode ae|vlm)
└── configs/
    ├── waypoint_ae_libero.yaml    # LIBERO AE 训练配置
    ├── waypoint_vlm_libero.yaml   # LIBERO VLM 训练配置
    ├── waypoint_ae_r1lite.yaml    # R1 Lite AE 训练配置
    └── eval_waypoint_libero.yaml  # LIBERO 评测配置
```

---

## 三、数据资产路径

### LIBERO

| 用途 | 路径 |
|------|------|
| 原始 RLDS (AE 训练) | `/workspace/data/libero/libero_object_no_noops/` |
| Waypoint-filtered RLDS (VLM 训练) | `/workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/` |
| Waypoint indices JSON (AE 训练) | `/workspace/data/libero/libero_object_wp_001/waypoint_indices.json` |
| 归一化统计量 (AE) | `/workspace/data/libero_object_no_noops/1.0.0/dataset_statistics_*.json` |
| 归一化统计量 (VLM) | `/workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json` |

**LIBERO 数据统计:**
- 原始: 454 个 episode, 66,984 步
- Waypoint 压缩后: 8,863 步 (~13.2% 保留率)
- Waypoint pair 数: ~7,000+ (duration ≤ 32 的有效 pair)

### R1 Lite

| 用途 | 路径 |
|------|------|
| Waypoint-filtered RLDS (VLM 训练) | `/workspace/r1lite_data/part1_r1_lite_compressed_wp_0001/waypoint_filtered_rlds/` |
| Waypoint indices JSON | `/workspace/r1lite_data/part1_r1_lite_compressed_wp_0001/waypoint_indices.json` |
| 原始 RLDS (AE 训练) | 训练时下载 (src_data_dir 见 waypoint_indices.json config) |

**R1 Lite 数据统计:**
- 23,039 个 episode, 5,108,454 原始步, 2,221,696 waypoint 步
- Waypoint 提取 key: `joint_position_arm_left + joint_position_arm_right + gripper_changes`
- 误差阈值: 0.001

### 模型权重

| 用途 | 路径 |
|------|------|
| Pi0.5 base (PyTorch) | `/workspace/models/pi05_base_pytorch/model.safetensors` |

**Pi0.5 base 权重 key 结构:**
- `action_in_proj.*` — action 投影层 (Linear 32→1024)
- `action_out_proj.*` — action 输出层 (Linear 1024→32)
- `time_mlp_in.*` / `time_mlp_out.*` — Pi0.5 时间步 MLP (Linear 1024→1024)
- `paligemma_with_expert.paligemma.*` — PaliGemma (Gemma 2B + SigLIP)
- `paligemma_with_expert.gemma_expert.*` — Action Expert (Gemma 300M)

---

## 四、Robot 配置 (`robot_config.py`)

### LIBERO 配置

```python
RobotConfig(
    robot_type = "libero",
    actual_action_dim = 7,   # EEF delta [3 pos, 3 rot, 1 gripper]
    actual_proprio_dim = 8,  # EEF state [3 pos, 3 rot, 2 gripper_qpos]
    action_dim_indices = [0,1,2,3,4,5,6],
    state_obs_keys = ["state"],
    camera_views = ["primary", "wrist"],
    camera_rlds_keys = {"primary": "image", "wrist": "wrist_image"},
    camera_model_keys = {"primary": "base_0_rgb", "wrist": "left_wrist_0_rgb"},
    normalize_gripper = normalize_gripper_libero,
    # Gripper dim (index 6) NOT normalized — stays in [0,1] after gripper transform
    action_norm_mask = [True]*6 + [False],
)
```

**LIBERO Gripper 处理:**
```
原始 RLDS action[-1]: [-1, +1]  (-1=open, +1=close)
  ↓ normalize_gripper_libero()
    clip(x, 0, 1) → [0, 1]
    1 - x          → [0, 1]  (0=close, 1=open)
  ↓ q99 normalize: mask=False → gripper 不归一化，保持 [0, 1]
  ↓ 推理时反归一化 → 同样 mask=False → 保持 [0,1]
  ↓ 转换到执行空间:
    x * 2.0 - 1.0    # [0,1] → [-1,+1]
    np.sign(x)       # binarize → {-1, +1}
    -x               # invert (LIBERO: -1=open, +1=close)
```

### R1 Lite 配置

```python
RobotConfig(
    robot_type = "galaxea_r1_lite",
    actual_action_dim = 14,  # 左臂(6+1 gripper) + 右臂(6+1 gripper)
    actual_proprio_dim = 14, # 同上
    action_dim_indices = [0..6, 7..13],  # 26d → 14d (去掉 torso/chassis)
    camera_views = ["head", "wrist_left", "wrist_right"],
    camera_rlds_keys = {...},
    camera_model_keys = {...},
    normalize_gripper = normalize_gripper_r1_lite,
    # Gripper dims 6 (left) 和 13 (right) NOT normalized
    action_norm_mask = [T,T,T,T,T,T,F, T,T,T,T,T,T,F],
)
```

---

## 五、归一化方案 (`normalize.py`)

### 统一使用 q99 归一化

```python
# 正向归一化 (训练时)
normalized = clip(2 * (x - q01) / (q99 - q01 + ε) - 1, -1, 1)
# 对于 q01 == q99 的维度 (常量维度): normalized = 0

# 反向归一化 (推理时)
unnormed = 0.5 * (x + 1) * (q99 - q01) + q01

# 对 mask=False 的维度 (如 gripper): 直接透传，不做变换
```

### 统计量 JSON 格式

```json
{
  "action": {"mean": [...], "std": [...], "min": [...], "max": [...], "q01": [...], "q99": [...]},
  "proprio": {"mean": [...], "std": [...], "min": [...], "max": [...], "q01": [...], "q99": [...]},
  "num_transitions": 66984,
  "num_trajectories": 454
}
```

---

## 六、Token 设计 (`tokenizer.py`)

### PaliGemma Vocab 末尾 token 分配

PaliGemma vocab size = 257,152。与 Pi0-FAST 一致，从末尾跳过 128 个保留 token (SKIP=128)：

```
Token ID 范围 (从高到低):
  257023 .. 256768  →  256 proprio bins (bin 0=257023, bin 255=256768)
  256767 .. 256734  →  34 duration tokens (d=0: 256767, d=33: 256734)
  256733            →  <wp> delimiter
  256732            →  <dur> delimiter
```

**共 292 个专用 token，不与 PaliGemma 原有 token 冲突。**

### WaypointTokenizer 编码格式

训练时完整 token 序列：
```
[BOS] "Task: {instruction}, State: {state_bins};\n" "Action: "
<wp> p₁..p₈ <dur> d₁
<wp> p₁..p₈ <dur> d₂
...
<wp> p₁..p₈ <dur> d_M
"|" [EOS]
```

- **Prefix (bidirectional, ar_mask=0)**: BOS + 文本 prompt + 状态
- **Postfix (causal, ar_mask=1)**: "Action: " + waypoint tokens + "|" + EOS
- **Loss mask**: 只在有效 waypoint 的 proprio tokens 和 duration tokens 上计算 CE loss
- `<wp>`, `<dur>`, "Action: ", "|" 等结构性 token → `loss_mask = False`

**LIBERO token 数量统计:**
- `tokens_per_waypoint = 1 + 8 + 1 + 1 = 11` (proprio_dim=8)
- M=7 waypoints → 最多 77 waypoint tokens
- prefix ≈ 40-60 tokens (instruction + state)
- 总序列 ≤ 256 tokens (在 max_token_len 内)

### ProprioTokenizer 量化误差

256 bins 均匀量化 [-1, 1]，bin 宽度 = 2/256 ≈ 0.0078。
最大量化误差 < 0.004 (bin 宽度的一半)。

---

## 七、Action Expert (`ae_model.py`)

### 架构变化 (相对于 Pi0.5 base)

| 组件 | Pi0.5 base | PI0WaypointAE |
|------|-----------|---------------|
| State conditioning | 1 个 state token (Linear 32→1024) | 2 个 proprio tokens: [start_wp, end_wp] (Linear 32→1024, shared) |
| Time conditioning | `time_emb` → AdaRMSNorm | `cat(time_emb, dur_emb)` → `time_mlp_in` → AdaRMSNorm |
| Duration | 无 | SinusoidalPosEmb(duration/33) |
| `time_mlp_in` 尺寸 | Linear(1024, 1024) | **Linear(2048, 1024)** (从头训练) |
| `time_mlp_out` 尺寸 | Linear(1024, 1024) | Linear(1024, 1024) (从 base 加载) |
| `action_in_proj` | Linear(32, 1024) | Linear(32, 1024) (从 base 加载) |

### 权重加载策略

```python
# 使用 strict=False 加载 Pi0.5 base 权重
# 可加载: paligemma_with_expert.*, action_in_proj.*, action_out_proj.*,
#         time_mlp_out.*, proprio_encoder (全新, 随机初始化)
# 不匹配 (全新随机初始化): time_mlp_in (shape 变为 2048→1024)
safetensors.load_model(model, weight_path, strict=False)
```

### Attention 结构

```
VLM_tokens   start_wp   end_wp   action₁..action₃₂
VLM_tokens     ✓
start_wp       ✓          ✓
end_wp         ✓          ✓        ✓
action(i)      ✓          ✓        ✓ (causal within actions)
```

### Loss 计算

```python
loss_per_element = MSE(u_t, v_t)          # [B, T, D]
loss_per_element *= (~action_pad_mask)     # mask padding steps
loss_per_element *= action_dim_mask        # mask padding dims
loss = sum(loss_per_element) / num_valid_elements
```

- `action_pad_mask[t] = True` → step t 是 zero-padding (duration < horizon_steps)
- `action_dim_mask[d] = True` → dim d 是真实维度 (非 model_dim 补零)

---

## 八、VLM (`vlm_model.py`)

### 架构

只使用 PaliGemma (Gemma 2B + SigLIP)，无 Action Expert。  
参考 Pi0-FAST 的 CE loss 架构。

### 训练 Loss

```python
# Prefix-LM attention:
#   图像 token + 语言 token: bidirectional (ar_mask=0)
#   waypoint tokens:         causal (ar_mask=1)

logits = paligemma_lm_head(transformer_output[:-1])  # predict next token
loss = CE(logits, targets[1:], mask=loss_mask[1:])
```

### 推理 (Constrained Decoding)

```python
# 强制在正确位置注入 <wp>/<dur> delimiter
# 模型只自由预测 proprio token 位置 (8 个 per WP) 和 duration token 位置 (1 个 per WP)
for step in range(max_steps):
    pos_in_wp = (wp_token_count) % tokens_per_waypoint
    if pos_in_wp == 0:
        force_token = wp_token_id       # 强制 <wp>
    elif pos_in_wp == proprio_dim + 1:
        force_token = dur_token_id      # 强制 <dur>
    else:
        force_token = argmax(logits)    # 模型自由预测
```

---

## 九、训练命令

### Action Expert

```bash
cd /workspace/openpi

# 单卡 (调试)
.venv/bin/python scripts/train_waypoint.py \
  --mode ae --config configs/waypoint_ae_libero.yaml

# 双卡 DDP
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_waypoint.py \
  --mode ae --config configs/waypoint_ae_libero.yaml

# 断点续训
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_waypoint.py \
  --mode ae --config configs/waypoint_ae_libero.yaml --resume
```

### VLM

```bash
cd /workspace/openpi

# 必须使用 venv 的 torchrun（系统 torchrun 使用错误的 Python 解释器）
# 必须设置 expandable_segments 避免 CUDA 内存碎片化导致 OOM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_waypoint.py \
  --mode vlm --config configs/waypoint_vlm_libero.yaml

# 断点续训
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_waypoint.py \
  --mode vlm --config configs/waypoint_vlm_libero.yaml --resume
```

### 归一化统计量生成

VLM 和 AE 都需要 q99 归一化统计量。如需从 waypoint-filtered RLDS 重新计算：

```bash
.venv/bin/python scripts/compute_wp_norm_stats.py \
  --rlds_dir /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0 \
  --robot_type libero \
  --output_dir /workspace/data/libero/libero_object_wp_001/norm_stats
```

### 评测

```bash
# 安装评测依赖（训练环境中不包含 LIBERO 仿真）
uv pip install --python .venv/bin/python \
    robosuite==1.4.1 transforms3d bddl easydict "gym==0.26.2"
uv pip install --python .venv/bin/python -e third_party/libero

# 修复 LIBERO torch.load 兼容性（PyTorch 2.6+ weights_only 默认值变更）
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    third_party/libero/libero/libero/benchmark/__init__.py

# 运行评测（单 GPU，需 ~20 GB 显存）
MUJOCO_GL=osmesa \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH \
PYTHONFAULTHANDLER=1 \
.venv/bin/python -u -m openpi.waypoint.eval_libero \
    --config configs/eval_waypoint_libero.yaml
```

评测配置（`configs/eval_waypoint_libero.yaml`）中需设置正确的 checkpoint 路径和 video 输出路径：
```yaml
vlm_checkpoint: /path/to/vlm/checkpoint_dir    # 须含 model.safetensors
ae_checkpoint: /path/to/ae/checkpoint_dir
video_out_path: data/libero/videos_wp           # 每个 episode 保存回放视频
```

---

## 十、配置文件关键参数

### `waypoint_ae_libero.yaml`

```yaml
robot_type: libero
original_rlds_dir: /workspace/data/libero/libero_object_no_noops
wp_indices_path: /workspace/data/libero/libero_object_wp_001/waypoint_indices.json
dataset_statistics_path: /workspace/data/libero_object_no_noops/1.0.0

model_action_dim: 32    # 模型统一维度 (LIBERO 7d 会 zero-pad 到 32d)
model_proprio_dim: 32   # 同上
horizon_steps: 32       # action chunk 长度
max_duration: 32        # 过滤 duration > 32 的 pair

paligemma_variant: gemma_2b
action_expert_variant: gemma_300m
precision: bfloat16

pretrained_weight_path: /workspace/models/pi05_base_pytorch

batch_size: 32
num_train_steps: 30000
peak_lr: 5.0e-5
norm_type: q99

# LoRA (false = 全量 finetune)
lora_enabled: false
```

### `waypoint_vlm_libero.yaml`

```yaml
wp_rlds_dir: /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0
dataset_statistics_path: /workspace/data/libero/libero_object_wp_001/norm_stats
# ↑ 注意: wp_rlds_dir 必须指向含 dataset_info.json 的 1.0.0 子目录

num_waypoints: 7
max_token_len: 256
stride: 4              # 每隔 4 步采样一次 VLM 训练样本
batch_size: 12         # per GPU; DDP 2 卡有效 batch=24
```

---

## 十一、Wandb 日志

每 `log_interval=50` 步记录：

| Metric | 含义 |
|--------|------|
| `train/loss` | 平均 loss (MSE for AE, CE for VLM) |
| `train/lr` | 当前 learning rate |
| `train/grad_norm` | 梯度 L2 范数 |
| `train/steps_per_sec` | 训练速度 |

Summary:
- `total_params`: 模型总参数量
- `trainable_params`: 可训练参数量 (LoRA 模式下为 LoRA 参数)
- `dataset_pairs`: AE 训练集 pair 总数

Project: `waypoint_vla`

---

## 十二、已知注意事项

### 权重加载
- `time_mlp_in` 从 `[1024, 1024]` (Pi0.5 base) 扩展为 `[2048, 1024]` (本项目)
- **不能用 `safetensors.load_model(strict=False)`** — safetensors 的 strict=False 只跳过 missing/unexpected keys，不跳过 shape mismatch，仍会报错
- 正确做法：手动循环 state_dict，遇到 shape 不匹配的 key 跳过并打印 warning（见 `train_waypoint.py`）

### 归一化
- LIBERO gripper (dim 6) 不参与 q99 归一化 (`action_norm_mask[-1] = False`)
- VLM 和 AE 使用**相同**的 q99 统计量，确保 VLM 输出的 proprio 可以直接作为 AE 的 end_proprio

### 数据格式兼容
- 原始 RLDS 的 stats JSON 格式是**扁平的** (`{"action": {...}, "proprio": {...}}`)
- 而非嵌套格式，`NormalizationHelper._find_stats` 已处理两种格式

### 图像张量格式
- 数据集 collator 输出图像为 `(B, C, H, W)` float32 [-1,1]
- `preprocessing_pytorch.py` 通过检查 `image.shape[1] == 3` 来识别 BCHW vs BHWC
- 必须在 collator 中做 `imgs.transpose(0, 3, 1, 2)` 把 `(B, H, W, C)` 转为 `(B, C, H, W)`

### DDP 后端
- 本环境使用 `backend = "gloo"` (不用 nccl)，与 openpi 已有的 `train_pytorch.py` 一致
- nccl 在该环境会导致 `CUDA illegal memory access`

### VLM-AE 接口
- VLM 输出 proprio 在 **q99 归一化空间** [-1, 1]
- AE 的 start/end_proprio 输入同样在 **q99 归一化空间**
- 两者使用相同的统计量 → 无需额外变换

### RLDS 数据路径
- TFDS `builder_from_directory` 需要指向**直接包含** `dataset_info.json` 的目录
- LIBERO: `/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0`（注意两层嵌套）

### VLM Gradient Checkpointing
- VLM 使用 HuggingFace 的 `PaliGemmaForConditionalGeneration`，**不是** openpi 自定义的 `PaliGemmaWithExpertModel`
- AE 的 `PaliGemmaWithExpertModel` 在 `gemma_pytorch.py` 中手动实现了逐层 `torch.utils.checkpoint.checkpoint`
- VLM 必须通过 HuggingFace API 激活 checkpointing：`self.paligemma.gradient_checkpointing_enable()`
  - 这会让每个 `GemmaDecoderLayer`（继承自 `GradientCheckpointingLayer`）在 `__call__` 中自动 checkpoint
  - 仅设置 `gradient_checkpointing = True` 属性**无效**——只会关闭 `use_cache`，不会减少激活内存
- 未正确激活时：batch_size=4 即接近 OOM（~94 GiB）；正确激活后：batch_size=16 单卡可用（~93 GiB）

### VLM Shuffle Buffer 启动策略
- VLM dataset 的 `__iter__` 必须使用 early-yield 策略（buffer 有 32 条即开始 yield），与 AE 一致
- 若等待 buffer 完全填满（如 5000 条），首个 batch 需 ~6 分钟（RLDS 遍历 5+ 轮）

### 依赖项
- openpi venv 默认不含 TensorFlow
- 需手动安装: `uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"`

### torchrun 路径
- 必须使用 `.venv/bin/torchrun`，系统 `torchrun`（`/venv/main/bin/torchrun`）会用错误的 Python 解释器，导致 `ModuleNotFoundError: No module named 'safetensors'`

### 评测 — VLM Checkpoint 格式兼容
- VLM 评测模型为 `PI0WaypointVLM`（仅 PaliGemma），key 前缀为 `paligemma.*`
- 但 `train_waypoint.py` 保存 VLM checkpoint 时可能使用了完整 AE 模型结构（key 前缀为 `paligemma_with_expert.paligemma.*`）
- `eval_libero.py` 的 `load_vlm()` 自动检测并 remap：`paligemma_with_expert.paligemma.X` → `paligemma.X`

### 评测 — AE bfloat16 dtype 对齐
- AE 以 bfloat16 加载（节省显存），但 attention mask 默认 float32
- `ae_model.py` 的 `sample_actions()` 中已添加自动 dtype 对齐：`prefix_att_4d = prefix_att_4d.to(model_dtype)`
- 不对齐会导致 SDPA 报 `RuntimeError: invalid dtype for bias`

### 评测 — 图像格式
- VLM 推理：图像以 BHWC 传入，`vlm_model.py` 内部 permute 为 BCHW
- AE 推理：图像**必须以 BCHW 传入**，经 `preprocessing_pytorch.py` 处理后保持 BCHW
- `eval_libero.py` 中 `predict_actions()` 对图像执行 `permute(2, 0, 1)` 转为 CHW

### 评测 — 视频录制
- 每个 episode 录制 agentview 相机的 256×256 图像（180° 旋转与训练一致）
- 使用 `imageio.mimwrite()` 保存为 MP4，10 FPS
- 文件名格式：`rollout_{task_name}_t{trial}_{success|failure}.mp4`

### 评测 — GPU 显存
- VLM (float32) ~11.7 GB + AE (bfloat16) ~7.5 GB ≈ **19.2 GB**
- 单张 RTX 4090 (24 GB) 可运行；两个模型同时在 GPU 上，无需 swap

---

## 十三、Batch Size 调优 (2x RTX PRO 6000 Blackwell 97.9GB)

每个 rank 独立运行 DataLoader，`batch_size` 是**单 GPU** 的 batch 大小。

### Action Expert

| batch_size (per GPU) | 实测 GPU 内存 | 有效总 batch | 备注 |
|----------------------|--------------|-------------|------|
| 32 | 59 GB / 50 GB | 64 | 训练正常 |
| 48 | ~73 GB / ~63 GB | 96 | **推荐** — 充分利用内存，有 ~25GB 余量 |
| 64 | ~88 GB / ~78 GB | 128 | 偏激进，约 10GB 余量 |

配合 LR 线性缩放规则: `peak_lr = 5e-5 × (new_batch / 64)`，即 batch=96 → `6e-5`

### VLM

VLM 全量 finetune PaliGemma 2B，内存占用显著高于 AE（因优化器状态和更长的序列）。
需要 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 避免 CUDA 内存碎片化。

| batch_size (per GPU) | 实测 GPU 内存 | 有效总 batch | 备注 |
|----------------------|--------------|-------------|------|
| 4 | ~48 GB | 8 | 单卡调试可用 |
| 12 | ~91-93 GB | 24 | **DDP 推荐** — 两卡均有 ~3-5 GB 余量 |
| 16 | ~93-94 GB | 32 | 单卡勉强可用，DDP OOM（DDP 额外开销） |
| 32 | OOM | — | 需 LoRA 或梯度累积 |
