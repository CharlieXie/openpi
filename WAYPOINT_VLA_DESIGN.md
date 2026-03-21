# Waypoint VLA — openpi 实现设计文档

> 最后更新: 2026-03-21 (rev 7 — Proprio 300-bin + Gripper 二值化 Token)
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
│   ├── joint_model.py    # PI0WaypointJoint (共享 backbone 联合 VLM+AE)  ← NEW
│   └── eval_libero.py    # 两段式 LIBERO 评测管线 (支持独立/联合两种模式)
├── scripts/
│   ├── train_waypoint.py       # 独立训练入口 (--mode ae|vlm)
│   └── train_waypoint_joint.py # 联合训练入口                         ← NEW
└── configs/
    ├── waypoint_ae_libero.yaml         # LIBERO AE 训练配置
    ├── waypoint_vlm_libero.yaml        # LIBERO VLM 训练配置
    ├── waypoint_ae_r1lite.yaml         # R1 Lite AE 训练配置
    ├── waypoint_joint_libero.yaml      # LIBERO 联合训练配置           ← NEW
    ├── eval_waypoint_libero.yaml       # LIBERO 评测配置 (独立 VLM+AE)
    └── eval_waypoint_joint_libero.yaml # LIBERO 评测配置 (联合模型)    ← NEW
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
    actual_action_dim = 7,        # EEF delta [3 pos, 3 rot, 1 gripper]
    actual_proprio_dim = 8,       # EEF 原始 state [3 pos, 3 rot, 2 gripper_qpos]
    continuous_proprio_dim = 6,   # VLM tokenizer 用的连续 proprio 维度 (dims 0-5)
    gripper_dim_index = 6,        # 原始 state 中用于二值化 gripper 的维度
    gripper_threshold = 0.02,     # grip_qpos_L > 0.02 → open
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

**LIBERO Action Gripper 处理 (AE 侧):**
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

**LIBERO State Gripper 处理 (VLM 侧, rev 7):**
```
原始 RLDS state[6]: grip_qpos_L ∈ [0.0007, 0.041]
  ↓ rc.binarize_gripper(state)
    state[6] > 0.02 → 1 (open), 否则 → 0 (close)
  ↓ 编码为 <grip_open> 或 <grip_close> token
  ↓ 解码时: grip_open → 1.0, grip_close → 0.0

State dims 0-5 (连续): 单独做 q99 归一化 → 300-bin 量化
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
  "action": {"mean": [7D], "std": [7D], "q01": [7D], "q99": [7D], ...},
  "proprio": {"mean": [6D], "std": [6D], "q01": [6D], "q99": [6D], ...},
  "num_transitions": 66984,
  "num_trajectories": 454
}
```

**rev 7 重要变更:**
- **action 统计**: `compute_wp_norm_stats.py` 先调用 `normalize_gripper()` 再计算统计量，
  使 gripper 维度的 q01/q99 反映训练值域 {0, 1} 而非原始 {-1, +1}
- **proprio 统计**: 只统计连续维度 (dims 0-5，共 6D)，排除 gripper_qpos (dims 6-7)。
  gripper 不参与 q99 归一化，改为二值化后编码为专用 token
- **旧统计文件不兼容**: proprio 维度从 8D → 6D，需要重新运行 `compute_wp_norm_stats.py`

---

## 六、Token 设计 (`tokenizer.py`)

### PaliGemma Vocab 末尾 token 分配

PaliGemma vocab size = 257,152。与 Pi0-FAST 一致，从末尾跳过 128 个保留 token (SKIP=128)：

```
Token ID 范围 (从高到低):
  257023 .. 256724  →  300 proprio bins (bin 0=257023, bin 299=256724)
  256723 .. 256690  →  34 duration tokens (d=0: 256723, d=33: 256690)
  256689            →  <wp> delimiter
  256688            →  <dur> delimiter
  256687            →  <grip_open> (gripper open token)
  256686            →  <grip_close> (gripper close token)
```

**共 338 个专用 token (300 + 34 + 2 + 2)，不与 PaliGemma 原有 token 冲突。**

### WaypointTokenizer 编码格式

训练时完整 token 序列：
```
[BOS] "Task: {instruction}, State: {state_bins}, Gripper: {open|closed};\n" "Action: "
<wp> p₁..p₆ G <dur> d₁
<wp> p₁..p₆ G <dur> d₂
...
<wp> p₁..p₆ G <dur> d_M
"|" [EOS]
```

其中 `G` ∈ {`<grip_open>`, `<grip_close>`}。

每个 waypoint 内部布局 (LIBERO, proprio_dim=6):
```
pos 0: <wp>          ← 强制注入，无 loss
pos 1: p₁            ← 300-bin logit mask，计算 loss
pos 2: p₂            ← 同上
pos 3: p₃
pos 4: p₄
pos 5: p₅
pos 6: p₆
pos 7: G             ← {grip_open, grip_close} logit mask，计算 loss
pos 8: <dur>         ← 强制注入，无 loss
pos 9: d             ← 34-bin logit mask，计算 loss
```

- **Prefix (bidirectional, ar_mask=0)**: BOS + 文本 prompt + 状态 + gripper 文本
- **Postfix (causal, ar_mask=1)**: "Action: " + waypoint tokens + "|" + EOS
- **Loss mask**: 在有效 waypoint 的 proprio tokens、gripper token 和 duration token 上计算 CE loss
- `<wp>`, `<dur>`, "Action: ", "|" 等结构性 token → `loss_mask = False`

**LIBERO token 数量统计 (rev 7):**
- `tokens_per_waypoint = 1 + 6 + 1 + 1 + 1 = 10` (proprio_dim=6 + gripper)
- M=7 waypoints → 最多 70 waypoint tokens (原 77)
- 每个 wp 最多 8 个 loss token (6 proprio + 1 gripper + 1 duration)
- prefix ≈ 40-60 tokens (instruction + state + gripper text)
- 总序列 ≤ 256 tokens (在 max_token_len 内)

### ProprioTokenizer 量化精度

300 bins 均匀量化 [-1, 1]，bin 宽度 = 2/300 ≈ 0.0067。
最大量化误差 < 0.0033 (bin 宽度的一半)。
相比旧版 256 bins (误差 < 0.004)，精度提升约 17%。

### Gripper Token 设计

State 的 gripper 维度 (grip_qpos) 本质上是二态信号 (夹爪开/关)，
用 256/300 bin 连续量化是浪费且增加分类难度。rev 7 改为：

- 专用 `<grip_open>` (ID 256687) 和 `<grip_close>` (ID 256686) 两个 token
- 训练/推理时为 2-class 分类 (大幅降低学习难度)
- 解码后映射为 1.0 (open) / 0.0 (close)，与 AE action gripper 约定一致
- VLM 和 AE 的 gripper 语义统一: 0=close, 1=open

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

# Masked logits: 只对 loss_mask=True 的位置计算 LM head + CE loss
# 有效 tokens 仅 ~63 (7 WP × 9)，全序列 ~768，节省 ~90% logits 内存
active_hidden = hidden[shift_loss_mask]        # [N_valid, 2048]
active_logits = lm_head(active_hidden.float()) # [N_valid, vocab_size]
loss = F.cross_entropy(active_logits, active_targets)
```

### 推理 (Constrained Decoding + Logit Masking)

```python
# 强制在正确位置注入 <wp>/<dur> delimiter
# 模型只自由预测 proprio token 位置 (8 个 per WP) 和 duration token 位置 (1 个 per WP)
#
# **关键**: 自由预测位置必须做 logit masking，将 argmax 限制在合法 token 子集内
# 否则 257K 词表中 pretrained 的语言 token 先验会压过 292 个 waypoint token
for step in range(max_steps):
    pos_in_wp = (wp_token_count) % tokens_per_waypoint
    if pos_in_wp == 0:
        force_token = wp_token_id       # 强制 <wp>
    elif pos_in_wp == proprio_dim + 1:
        force_token = dur_token_id      # 强制 <dur>
    elif 1 <= pos_in_wp <= proprio_dim:
        logits[outside_proprio_range] = -inf  # 只允许 256 个 proprio bin token
        force_token = argmax(logits)
    elif pos_in_wp == proprio_dim + 2:
        logits[outside_duration_range] = -inf # 只允许 34 个 duration token
        force_token = argmax(logits)
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

### 评测 — VLM Logit Masking (rev 5 修复)
- **问题**: VLM 的 `generate_waypoints()` 在自由预测位置（proprio / duration）对 257,152 个全词表做 argmax，pretrained 语言 token 先验远高于 292 个专用 waypoint token，导致 argmax 始终落在普通词表 token 上
- **表现**: 所有 proprio 解码为 0.9961（bin 255），duration 解码为 253416 等荒谬值
- **根因**: `ProprioTokenizer.decode()` 的 `np.clip(bin_indices, 0, 255)` 将任何普通词表 token 静默映射到 bin 255；`decode_duration()` 无范围检查
- **修复 (vlm_model.py)**: 在 `generate_waypoints()` 中，proprio 位置仅允许 token 256768–257023，duration 位置仅允许 token 256734–256767，其余设为 `-inf`
- **修复 (tokenizer.py)**: `decode()` 和 `decode_duration()` 添加越界 warning，不再静默吞掉错误
- **修复 (eval_libero.py)**: duration 超过 `horizon_steps` 时 clamp 并打 warning

### 训练 — DDP Forward 调用 (rev 5 修复)
- **问题**: `train_waypoint.py` 的 VLM/AE 训练循环使用 `raw_model(batch)`（绕过 DDP wrapper），可能导致 `find_unused_parameters=True` 时梯度同步异常
- **修复**: 改为 `model(batch)` / `model(observation=..., ...)` 直接调用 DDP 包裹的模型

### 评测 — 归一化统计量路径一致性
- 训练和评测**必须**使用完全相同的 `dataset_statistics.json`
- 不一致会导致 state prompt 离散化值有分布偏移，降低模型预测质量
- 评测配置中的 `dataset_statistics_path` 应与训练配置中的路径指向同一份统计量文件

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

VLM 全量 finetune PaliGemma 2B。v2 优化后内存占用大幅降低。
需要 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 避免 CUDA 内存碎片化。

**v2 优化 (三项合计节省 ~60 GB per GPU):**
1. **Masked logits**: 只对 loss_mask=True 的位置 (~63 tokens) 计算 LM head，不再对全序列 (~768) 计算 → 节省 ~40 GB
2. **bfloat16**: config `precision: bfloat16` 现在生效，模型参数+优化器状态减半 → 节省 ~20 GB
3. **跳过 dummy image**: LIBERO 只有 2 个 camera，不再处理全零的第 3 张图 → 节省 ~2 GB + 序列缩短 25%

| batch_size (per GPU) | 实测 GPU 内存 | 有效总 batch | 备注 |
|----------------------|--------------|-------------|------|
| 4 | ~48 GB | 8 | 单卡调试可用 (v1 float32) |
| 12 | ~91-93 GB | 24 | v1 DDP 极限 (float32 + 全序列 logits) |
| 14 | ~93 GB | 28 | v1 DDP 极限 |
| 32+ | 待测 | 64+ | **v2 优化后预期可行** — 需重新 profiling |

---

## 十四、Joint VLM + AE 联合训练 (rev 6)

### 动机

原始设计中 VLM 和 AE 是两个独立模型，各自含有一份 PaliGemma backbone (Gemma 2B + SigLIP)：
- 训练时：参数不共享，各自占用 ~11 GB × 2 = 22 GB backbone 权重
- 推理时：需分别加载两个 checkpoint，占用双倍显存
- 学习信号：VLM 的语言/视觉理解能力无法受益于 AE 的动作监督信号，反之亦然

联合模型将 VLM 和 AE 的 PaliGemma backbone 合并为单一实例，通过梯度策略控制两种 loss 的交互。

### 架构设计

```
PI0WaypointJoint (nn.Module)
├── paligemma_with_expert (PaliGemmaWithExpertModel)  ← 单一实例，共享权重
│   ├── paligemma (PaliGemmaForConditionalGeneration)
│   │   ├── vision_tower (SigLIP)           ← VLM + AE 共享
│   │   └── language_model (Gemma 2B)       ← VLM + AE 共享
│   └── gemma_expert (GemmaForCausalLM 300M) ← 仅 AE 使用
├── action_in_proj   (Linear 32→1024)       ← AE 专用
├── action_out_proj  (Linear 1024→32)       ← AE 专用
├── proprio_encoder  (Linear 32→1024)       ← AE 专用
├── time_mlp_in      (Linear 2048→1024)     ← AE 专用 (新尺寸)
└── time_mlp_out     (Linear 1024→1024)     ← AE 专用
```

关键区别于独立模型：
1. VLM 路径通过 `paligemma_with_expert.paligemma` 做 CE loss（与 `PI0WaypointVLM` 相同逻辑）
2. AE 路径通过完整 `paligemma_with_expert`（backbone + expert）做 MSE loss（与 `PI0WaypointAE` 相同逻辑）
3. backbone 参数只存一份，VLM CE loss 和 AE MSE loss 的梯度可同时流入 backbone

### 方法清单

| 方法 | 用途 | 来源 |
|------|------|------|
| `forward(mode, **kwargs)` | DDP 兼容的分发入口，`mode="vlm"` 或 `"ae"` | 新增 |
| `vlm_forward(batch)` | CE loss on waypoint tokens | 移植自 `vlm_model.py:106-201` |
| `ae_forward(obs, start_proprio, end_proprio, actions, duration, ...)` | MSE flow-matching loss，含 stop_gradient 支持 | 移植自 `ae_model.py:166-242` |
| `embed_prefix(images, img_masks, lang_tokens, lang_masks)` | 编码视觉+语言前缀 | 移植自 `ae_model.py:89-111` |
| `embed_suffix(start_proprio, end_proprio, noisy_actions, timestep, duration)` | 编码 AE 后缀（proprio + actions + time/dur） | 移植自 `ae_model.py:113-164` |
| `generate_waypoints(images, image_masks, prompt_tokens, prompt_mask, wp_tokenizer, ...)` | VLM constrained AR 推理 | 移植自 `vlm_model.py:203-383` |
| `sample_actions(observation, start_proprio, end_proprio, duration, ...)` | AE 迭代去噪推理 | 移植自 `ae_model.py:244-331` |
| `gradient_checkpointing_enable/disable()` | 同时控制 backbone + expert 的 checkpointing | 移植自 `ae_model.py:70-80` |
| `load_pretrained_weights(model, weight_path, device)` | 静态方法，shape 不匹配容错加载 | 移植自 `train_waypoint.py:198-211` |

### 梯度策略

通过构造函数参数 `gradient_strategy` 控制，支持三种模式：

#### `"none"` — 无隔离

```
VLM CE loss ──backprop──► backbone ◄──backprop── AE MSE loss
```

两种 loss 梯度自由流入 backbone 所有参数。最简单，但可能导致梯度冲突——CE 和 MSE 优化目标不同，可能让 backbone 在两个方向间震荡。

#### `"stop_gradient"` — Knowledge Insulation (Pi0.5 §5.2)

```
每一层 attention 中:

  Backbone:  Q_b, K_b, V_b  ──正常计算──► attn_output_b (不参与 MSE loss)
  Expert:    Q_a, sg(K_b), sg(V_b), K_a, V_a  ──► attn_output_a ──► MSE loss
                   ↑          ↑
              梯度阻断     梯度阻断
```

在 `ae_forward()` 中，通过 `knowledge_insulation=True` 传递给 `paligemma_with_expert.forward()`。
在 `gemma_pytorch.py` 的 `compute_layer_complete` 中，每层 attention 计算前对 backbone 的 K 和 V 施加 detach：

```python
if knowledge_insulation:
    key_states[0] = key_states[0].detach()    # sg(K_b)
    value_states[0] = value_states[0].detach()  # sg(V_b)
```

**对应论文公式 (5)**：`Q_a(X_a) @ sg(K_b(X_b))^T` — expert 的 query 可以 attend to backbone 的 key，但梯度不流回 backbone 的 `k_proj` 权重。

**对应论文公式 (6)**：`P_ab @ sg(V_b(X_b))` — expert 可以读取 backbone 的 value 表征，但梯度不流回 backbone 的 `v_proj` 权重。

**为什么全局 detach K_b/V_b 等价于论文的 selective sg**：attention mask 设计决定了 backbone 输出 `attn_output_b` 不参与 MSE loss 路径（prefix tokens 不出现在 action prediction 中）。因此 backbone attention 的 Q_b @ K_b^T 路径上的 detach 是无害的——该路径上原本就没有 MSE 梯度。

**效果**：backbone 只接收 VLM CE 梯度，AE 仅更新 expert + projection 层（proprio_encoder、time_mlp、action_in/out_proj）。与旧版 `prefix_embs.detach()` 相比，新方法彻底阻断了 MSE 梯度通过 backbone 层权重（q/k/v/o_proj、MLP）的路径。

#### `"freeze_backbone"` — 冻结 backbone

```python
# 构造函数中
for param in self.paligemma_with_expert.paligemma.parameters():
    param.requires_grad = False
```

backbone 完全冻结。VLM 和 AE 均无法更新 backbone。VLM CE loss 仍计算但不产生 backbone 梯度，AE 仅更新 expert + projection 层。

适合 backbone 已经在大规模预训练中收敛、只需微调 AE 组件的场景。

### 训练循环设计

```python
for global_step in range(num_steps):
    # 1. 更新 LR (cosine decay with warmup)
    lr = cosine_lr(global_step, warmup, peak_lr, decay_steps, end_lr)

    # 2. VLM forward + backward（不同步 DDP 梯度）
    with model.no_sync():              # DDP: 暂缓 allreduce
        vlm_loss = model(mode="vlm", batch=vlm_batch)
        vlm_loss.backward()            # 梯度累积到 .grad

    # 3. AE forward + backward（触发 DDP 梯度同步）
    ae_loss = model(mode="ae", observation=ae_obs, ...)
    (ae_loss_weight * ae_loss).backward()  # 触发 allreduce

    # 4. 优化器更新
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

**关键设计决策：**

1. **`model.no_sync()` + DDP**：VLM backward 不触发 allreduce，AE backward 触发 allreduce，同步 VLM+AE 两次 backward 累积的梯度。省去一次 allreduce 通信开销。

2. **`forward(mode, **kwargs)`** 分发器：DDP 要求所有 forward 调用经过 `nn.Module.__call__`（即 DDP wrapper），才能正确追踪参数使用和触发 gradient sync hooks。直接调用 `model.module.vlm_forward()` 会绕过 DDP 导致梯度不同步。

3. **两个独立 DataLoader**：VLM 和 AE 使用不同的数据集（`WaypointVLMDataset` vs `WaypointAEDataset`），各自有独立的 batch_size、shuffle_buffer 和 image augmentation 配置。

4. **显存峰值 ≈ max(VLM_peak, AE_peak)**：每次 backward 释放计算图后再做下一次 forward，不同时持有两个计算图的激活。

5. **`ae_loss_weight`**：AE loss 乘以权重系数后再 backward，控制 AE 梯度相对 VLM 梯度的比例。

### 配置文件 (`waypoint_joint_libero.yaml`)

```yaml
# --- 联合训练专用字段 ---
gradient_strategy: none    # "none" | "stop_gradient" | "freeze_backbone"
ae_loss_weight: 1.0        # AE MSE loss 乘以此系数

# 两套独立 batch size (per GPU)
vlm_batch_size: 64
ae_batch_size: 64

# 两套数据路径
# AE:
original_rlds_dir: /workspace/data/libero/libero_object_no_noops/...
wp_indices_path: /workspace/data/libero/libero_object_wp_001/waypoint_indices.json
# VLM:
wp_rlds_dir: /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0

# 两套 token len
ae_max_token_len: 64       # AE 语言 prompt 长度
vlm_max_token_len: 256     # VLM waypoint 序列总长度

# 两套 image augmentation
image_aug_cfg: { ... }     # VLM augmentation (HF-style)
ae_image_aug_cfg: { ... }  # AE augmentation (crop/rotation/color)

# 两套 shuffle buffer
vlm_shuffle_buffer_size: 5000
ae_shuffle_buffer_size: 1000
```

### 训练命令

```bash
cd /workspace/openpi

# 单卡
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train_waypoint_joint.py \
    --config configs/waypoint_joint_libero.yaml

# 双卡 DDP
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_waypoint_joint.py \
    --config configs/waypoint_joint_libero.yaml
```

### 权重加载

联合模型使用与 AE 相同的权重加载策略：
- 从 Pi0.5 base checkpoint (`model.safetensors`) 逐 key 加载
- `time_mlp_in` shape 不匹配 (`[1024,1024]` → `[2048,1024]`) 自动跳过
- 缺失的 key (如 `proprio_encoder`) 保持随机初始化

```python
PI0WaypointJoint.load_pretrained_weights(model, weight_path, device)
```

### Wandb 日志

联合训练额外记录：

| Metric | 含义 |
|--------|------|
| `train/vlm_loss` | VLM CE loss |
| `train/ae_loss` | AE MSE loss |
| `train/total_loss` | `vlm_loss + ae_loss_weight * ae_loss` |
| `train/lr` | 当前 learning rate |
| `train/grad_norm` | 梯度 L2 范数 (VLM+AE 梯度合计) |

Summary 额外字段：
- `gradient_strategy`: 使用的梯度策略
- `ae_loss_weight`: AE loss 权重系数

### 评测 — 联合模型

`eval_libero.py` 支持两种加载模式，通过配置文件自动检测：

**独立模式**（现有行为）：
```yaml
vlm_checkpoint: /path/to/vlm/checkpoint
ae_checkpoint: /path/to/ae/checkpoint
```

**联合模式**（新增）：
```yaml
joint_checkpoint: /path/to/joint/checkpoint
```

当配置中存在 `joint_checkpoint` 时，`evaluate()` 自动切换到联合模式：
```python
if "joint_checkpoint" in cfg:
    joint_model = load_joint(cfg, device)
    vlm = joint_model      # generate_waypoints() 在 joint model 上
    ae_model = joint_model  # sample_actions() 在 joint model 上
```

**接口兼容性**：`PI0WaypointJoint` 同时实现了 `generate_waypoints()` 和 `sample_actions()`，签名与独立模型完全相同，因此 `predict_waypoints()` 和 `predict_actions()` 无需修改。

**显存优势**：联合模型只加载一份 backbone，推理时节省 ~5 GB 显存（约 ~14 GB vs ~19 GB）。

### 验证清单

1. **冒烟测试**: 单 GPU 运行 1 步确认无报错
   ```bash
   python scripts/train_waypoint_joint.py --config configs/waypoint_joint_libero.yaml
   ```

2. **梯度策略验证**: 分别测试三种策略，检查 backbone 参数梯度
   - `none`: backbone 有来自 VLM CE + AE MSE 两个 loss 的梯度
   - `stop_gradient`: backbone 只有 VLM CE 梯度（AE MSE 梯度被 detach 阻断）
   - `freeze_backbone`: backbone 无梯度（`requires_grad=False`）

3. **Wandb 日志**: 确认 `vlm_loss` 和 `ae_loss` 分别可见且数值合理

4. **评测联合 checkpoint**: 使用 `eval_waypoint_joint_libero.yaml` 验证联合 checkpoint 可正常推理

### 未修改的文件

联合训练不修改以下现有文件，保持独立训练路径完整可用：

| 文件 | 原因 |
|------|------|
| `ae_model.py` | 独立 AE 模型保持不变 |
| `vlm_model.py` | 独立 VLM 模型保持不变 |
| `ae_dataset.py` | 数据集不变，联合训练使用两个独立 DataLoader |
| `vlm_dataset.py` | 同上 |
| `train_waypoint.py` | 不修改，联合训练脚本从中 import 工具函数 |
| `gemma_pytorch.py` | 不修改 attention 实现，梯度隔离在模型层面用 detach 实现 |

---

## 十五、Proprio 300-bin + Gripper 二值化 Token (rev 7)

### 动机

1. **Proprio 量化精度**: 256 bins → 300 bins，bin 宽度从 0.0078 缩小到 0.0067，精度提升 ~17%，代价极小（序列长度不变，计算量不变）
2. **Gripper 二值化**: State 的 grip_qpos (dims 6-7) 本质是二态信号（开/关），用 256-bin 连续量化 → 256 分类，浪费且增加学习难度。改为专用 `<grip_open>` / `<grip_close>` 2 分类
3. **VLM-AE 对齐**: 改造后 VLM 解码的 waypoint proprio 为 7D (6D continuous + 1D gripper binary)，正好匹配 AE 的 `action_dim=7`，消除了旧版 8D→7D 截断导致 grip_qpos_R 被丢弃的问题
4. **Norm stats 修正**: 统计脚本改为对 action 先做 `normalize_gripper` 再统计，proprio 只统计连续 6D

### 数据流总览

```
原始 state (8D): [eef_x, eef_y, eef_z, rot1, rot2, rot3, grip_qpos_L, grip_qpos_R]
  ├─→ continuous = state[0:6]  → q99 normalize → 300-bin quantize → 6 proprio tokens
  └─→ gripper = state[6] > 0.02 → binary → 1 gripper token {<grip_open> / <grip_close>}

VLM 解码 waypoint: 6D continuous (300-bin decode) + 1D gripper (1.0/0.0)
  = 7D proprio → pad_to_dim(32) → AE proprio_encoder(Linear(7, W))
```

### Token 布局变化

```
旧 (rev 6):  <wp> p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ <dur> d     (11 tokens/wp, p∈256bins)
新 (rev 7):  <wp> p₁ p₂ p₃ p₄ p₅ p₆ G <dur> d           (10 tokens/wp, p∈300bins)
```

- 7 waypoints: 77 → 70 tokens (节省 7 tokens)
- Constrained decoding: pos 7 (gripper) 使用 2-class logit mask

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `tokenizer.py` | `PROPRIO_N_BINS=300`; 新增 `_GRIP_OPEN_ID`, `_GRIP_CLOSE_ID`; `tokens_per_waypoint` 10; `encode_gripper/decode_gripper`; `tokenize()` 接受 `current_gripper`, `wp_grippers` 参数 |
| `robot_config.py` | 新增 `continuous_proprio_dim=6`, `gripper_dim_index=6`, `gripper_threshold=0.02`; 新增 `binarize_gripper()`, `split_proprio()` 方法 |
| `vlm_dataset.py` | 拆分 proprio 为 6D continuous + binary gripper; 只对 continuous dims 做 q99 归一化 |
| `ae_dataset.py` | 拆分 proprio 为 6D continuous + 1D binary → 拼成 7D 传给 AE; `proprio_dim_mask` 改用 `continuous_proprio_dim+1` |
| `vlm_model.py` | `generate_waypoints()` 更新 constrained decoding: 新增 gripper position logit mask (2 IDs) |
| `joint_model.py` | 同 `vlm_model.py` |
| `eval_libero.py` | `predict_waypoints()` 接受 `state_continuous_norm` + `gripper_binary`; `run_episode()` 拆分 proprio; `WaypointTokenizer` 改用 `continuous_proprio_dim` |
| `compute_wp_norm_stats.py` | action 先 `normalize_gripper()` 再统计; proprio 只统计 `state[:continuous_proprio_dim]` (6D) |
| `train_waypoint.py` | `WaypointTokenizer` 改用 `continuous_proprio_dim` + `use_gripper_token=True` |
| `train_waypoint_joint.py` | 同上 |

### 不兼容变更

- **checkpoint 不兼容**: token layout 变化 (ID 偏移、tpw 11→10)，需从头训练
- **统计文件不兼容**: proprio 从 8D → 6D，需重新运行 `compute_wp_norm_stats.py`
- **配置文件**: 无需修改 (proprio_dim 现在自动从 `rc.continuous_proprio_dim` 获取)

### 验证清单

1. 重新运行 `compute_wp_norm_stats.py` 生成新的 `dataset_statistics.json` (proprio 6D, action 7D with correct gripper stats)
2. 冒烟测试 joint training 1 步确认无报错
3. 确认 VLM loss 和 AE loss 正常下降
4. 推理验证: 确认 gripper token 预测正确 (全开/全关 pattern 符合任务逻辑)
