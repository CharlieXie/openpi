# Waypoint VLA — 联合训练设计文档

> 基于 Pi0.5 (PyTorch) 在 openpi 项目中实现两段式 Waypoint VLA 系统（联合训练）。

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
│   ├── joint_model.py    # PI0WaypointJoint (共享 backbone 联合 VLM+AE) — 主模型
│   ├── ae_dataset.py     # WaypointAEDataset (RLDS → AE training samples)
│   ├── vlm_dataset.py    # WaypointVLMDataset (RLDS → VLM token sequences)
│   ├── eval_libero.py    # 两段式 LIBERO 评测管线
│   └── eval_calvin.py    # 两段式 CALVIN 评测管线 (chain-task 5 subtasks × 1000 sequences)
├── scripts/
│   └── train_waypoint_joint.py # 联合训练入口
└── configs/
    ├── waypoint_joint_libero.yaml      # LIBERO 联合训练配置
    ├── eval_waypoint_joint_libero.yaml # LIBERO 评测配置
    ├── waypoint_joint_calvin.yaml      # CALVIN 联合训练配置
    └── eval_waypoint_joint_calvin.yaml # CALVIN 评测配置
```

---

## 三、数据资产路径

### LIBERO

| 用途 | 路径 |
|------|------|
| 原始 RLDS (AE 训练) | `/workspace/data/libero/libero_object_no_noops/` |
| Waypoint-filtered RLDS (VLM 训练) | `/workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/` |
| Waypoint indices JSON (AE 训练) | `/workspace/data/libero/libero_object_wp_001/waypoint_indices.json` |
| 归一化统计量 | `/workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json` |

**LIBERO 数据统计:**
- 原始: 454 个 episode, 66,984 步
- Waypoint 压缩后: 8,863 步 (~13.2% 保留率)
- Waypoint pair 数: ~7,000+ (duration ≤ 32 的有效 pair)

### LIBERO 多 Suite 联合 (spatial + object + goal + libero_10)

| 用途 | 路径 |
|------|------|
| 合并原始 RLDS (AE 训练) | `/workspace/data/modified_libero_rlds/libero_all_no_noops/1.0.0/` |
| Waypoint-filtered RLDS (VLM 训练) | `/workspace/data/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/` |
| Waypoint indices JSON (AE 训练) | `/workspace/data/libero_object_wp_001/waypoint_indices.json` |
| 归一化统计量 | `/workspace/data/modified_libero_rlds/dataset_statistics.json` |

| Suite | Episodes | AE pairs | VLM windows |
|-------|----------|----------|-------------|
| spatial | 432 | 7,977 | 7,977 |
| object | 454 | 8,409 | 8,409 |
| goal | 428 | 7,187 | 7,188 |
| libero_10 | 379 | 12,486 | 12,487 |
| **Total** | **1,693** | **36,059** | **36,061** |

### CALVIN (ABC_D)

| 用途 | 路径 |
|------|------|
| 原始 RLDS (AE 训练) | `/workspace/data/calvin_abc_rlds/1.0.0` |
| Waypoint-filtered RLDS (VLM 训练) | `/workspace/calvin_abc_wp_001/calvin_abc_wp/1.0.0` |
| Waypoint indices JSON (AE 训练) | `/workspace/calvin_abc_wp_001/waypoint_indices.json` |
| 归一化统计量 | `/workspace/calvin_abc_wp_001/norm_stats` |

**CALVIN 数据说明:**
- Waypoint 使用 `awe/example/rlds_wp_extract_calvin_parallel.py` 提取
- RLDS 中包含 `waypoint_duration` 和 `is_waypoint_end` 字段
- 观测: `observation/rgb_static` (200x200), `observation/rgb_gripper` (84x84), `observation/state` (15D)
- 动作: `action` (7D: delta_pos(3) + delta_euler(3) + gripper(1))

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

**LIBERO State Gripper 处理 (VLM 侧):**
```
原始 RLDS state[6]: grip_qpos_L ∈ [0.0007, 0.041]
  ↓ rc.binarize_gripper(state)
    state[6] > 0.02 → 1 (open), 否则 → 0 (close)
  ↓ 编码为 <grip_open> 或 <grip_close> token
  ↓ 解码时: grip_open → 1.0, grip_close → 0.0

State dims 0-5 (连续): 单独做 q99 归一化 → 300-bin 量化
```

### CALVIN 配置

```python
RobotConfig(
    robot_type = "calvin",
    actual_action_dim = 7,        # delta [3 pos, 3 euler, 1 gripper]
    actual_proprio_dim = 15,      # robot_obs 15D
    continuous_proprio_dim = 6,   # VLM tokenizer 用的连续 proprio 维度 (dims 0-5: TCP + euler)
    gripper_dim_index = 6,        # grip_width (meters, 0~0.08)
    gripper_threshold = 0.04,     # grip_width > 0.04m → open
    action_dim_indices = [0,1,2,3,4,5,6],
    state_obs_keys = ["state"],
    camera_views = ["primary", "wrist"],
    camera_rlds_keys = {"primary": "rgb_static", "wrist": "rgb_gripper"},
    camera_model_keys = {"primary": "base_0_rgb", "wrist": "left_wrist_0_rgb"},
    normalize_gripper = normalize_gripper_calvin,
    action_norm_mask = [True]*6 + [False],
)
```

**CALVIN vs LIBERO Gripper 语义差异:**

| 阶段 | LIBERO | CALVIN |
|------|--------|--------|
| 原始 action[-1] 语义 | -1=open, +1=close | -1=close, +1=open |
| `normalize_gripper()` | `clip(0,1)` 然后 `1-x` (反转) | `clip(0,1)` (无反转) |
| 训练后 model output | [0,1], 0=close, 1=open | [0,1], 0=close, 1=open |
| Eval 后处理 | `x*2-1 → sign → negate` | `x*2-1 → sign` (无 negate) |
| Env 期望 | -1=open, +1=close | +1=open, -1=close |

**CALVIN State (robot_obs) 15D 结构:**

| Dim | 描述 |
|-----|------|
| 0-2 | TCP position (x, y, z) |
| 3-5 | TCP orientation (euler x, y, z) |
| 6 | Gripper width (meters, 0~0.08) |
| 7-13 | 7 joint angles (radians) |
| 14 | Gripper action (-1/+1, +1=open) |

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

**重要:**
- **action 统计**: `compute_wp_norm_stats.py` 先调用 `normalize_gripper()` 再计算统计量，
  使 gripper 维度的 q01/q99 反映训练值域 {0, 1} 而非原始 {-1, +1}
- **proprio 统计**: 只统计连续维度 (dims 0-5，共 6D)，排除 gripper_qpos
- gripper 不参与 q99 归一化，改为二值化后编码为专用 token

---

## 六、Token 设计 (`tokenizer.py`)

### PaliGemma Vocab 末尾 token 分配

PaliGemma vocab size = 257,152。从末尾跳过 128 个保留 token (SKIP=128)：

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

**Token 数量统计:**
- `tokens_per_waypoint = 1 + 6 + 1 + 1 + 1 = 10` (proprio_dim=6 + gripper)
- M=7 waypoints → 最多 70 waypoint tokens
- 每个 wp 最多 8 个 loss token (6 proprio + 1 gripper + 1 duration)
- prefix ≈ 40-60 tokens (instruction + state + gripper text)
- 总序列 ≤ 256 tokens (在 max_token_len 内)

### ProprioTokenizer 量化精度

300 bins 均匀量化 [-1, 1]，bin 宽度 = 2/300 ≈ 0.0067。最大量化误差 < 0.0033。

### Gripper Token 设计

State 的 gripper 维度本质上是二态信号 (夹爪开/关)，专用 `<grip_open>` / `<grip_close>` 两个 token 做 2-class 分类，解码后映射为 1.0 (open) / 0.0 (close)。

---

## 七、联合模型架构 (`joint_model.py`)

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

关键设计：
1. VLM 路径通过 `paligemma_with_expert.paligemma` 做 CE loss
2. AE 路径通过完整 `paligemma_with_expert`（backbone + expert）做 MSE loss
3. backbone 参数只存一份，VLM CE loss 和 AE MSE loss 的梯度可同时流入 backbone

### AE 部分 — 架构变化 (相对于 Pi0.5 base)

| 组件 | Pi0.5 base | PI0WaypointJoint (AE) |
|------|-----------|---------------|
| State conditioning | 1 个 state token (Linear 32→1024) | 2 个 proprio tokens: [start_wp, end_wp] (Linear 32→1024, shared) |
| Time conditioning | `time_emb` → AdaRMSNorm | `cat(time_emb, dur_emb)` → `time_mlp_in` → AdaRMSNorm |
| Duration | 无 | SinusoidalPosEmb(duration/33) |
| `time_mlp_in` 尺寸 | Linear(1024, 1024) | **Linear(2048, 1024)** (从头训练) |

### AE Attention 结构

```
VLM_tokens   start_wp   end_wp   action₁..action₃₂
VLM_tokens     ✓
start_wp       ✓          ✓
end_wp         ✓          ✓        ✓
action(i)      ✓          ✓        ✓ (causal within actions)
```

### AE Loss 计算

```python
loss_per_element = MSE(u_t, v_t)          # [B, T, D]
loss_per_element *= (~action_pad_mask)     # mask padding steps
loss_per_element *= action_dim_mask        # mask padding dims
loss = sum(loss_per_element) / num_valid_elements
```

### VLM 部分 — 推理 (Constrained Decoding + Logit Masking)

```python
for step in range(max_steps):
    pos_in_wp = (wp_token_count) % tokens_per_waypoint
    if pos_in_wp == 0:
        force_token = wp_token_id       # 强制 <wp>
    elif pos_in_wp == proprio_dim + 1:
        force_token = dur_token_id      # 强制 <dur>
    elif 1 <= pos_in_wp <= proprio_dim:
        logits[outside_proprio_range] = -inf  # 只允许 300 个 proprio bin token
        force_token = argmax(logits)
    elif pos_in_wp == proprio_dim + 2:
        logits[outside_duration_range] = -inf # 只允许 34 个 duration token
        force_token = argmax(logits)
```

### 方法清单

| 方法 | 用途 |
|------|------|
| `forward(mode, **kwargs)` | DDP 兼容的分发入口，`mode="vlm"` 或 `"ae"` |
| `vlm_forward(batch)` | CE loss on waypoint tokens |
| `ae_forward(obs, start_proprio, end_proprio, actions, duration, ...)` | MSE flow-matching loss，含梯度策略支持 |
| `embed_prefix(images, img_masks, lang_tokens, lang_masks)` | 编码视觉+语言前缀 |
| `embed_suffix(start_proprio, end_proprio, noisy_actions, timestep, duration)` | 编码 AE 后缀 |
| `generate_waypoints(...)` | VLM constrained AR 推理 |
| `sample_actions(...)` | AE 迭代去噪推理 |
| `gradient_checkpointing_enable/disable()` | 同时控制 backbone + expert 的 checkpointing |
| `load_pretrained_weights(model, weight_path, device)` | 静态方法，shape 不匹配容错加载 |

---

## 八、梯度策略

共享 backbone 时，VLM (CE loss) 和 AE (MSE loss) 的梯度可能冲突。通过 `gradient_strategy` 控制，支持四种模式：

### `"none"` — 无隔离

两种 loss 梯度自由流入 backbone 所有参数。最简单，但可能导致梯度冲突。

### `"stop_gradient"` — Knowledge Insulation (Pi0.5 §5.2)

在 `gemma_pytorch.py` 的 `compute_layer_complete` 中，每层 attention 计算前对 backbone 的 K 和 V 施加 detach：

```python
if knowledge_insulation:
    key_states[0] = key_states[0].detach()    # sg(K_b)
    value_states[0] = value_states[0].detach()  # sg(V_b)
```

backbone 只接收 VLM CE 梯度，AE 仅更新 expert + projection 层。

### `"scale_gradient"` — 软 Knowledge Insulation（推荐）

```python
if isinstance(knowledge_insulation, float) and 0 < knowledge_insulation < 1:
    scale = knowledge_insulation  # e.g., 0.3
    key_states[0] = key_states[0] * scale + key_states[0].detach() * (1 - scale)
    value_states[0] = value_states[0] * scale + value_states[0].detach() * (1 - scale)
```

前向传播值不变，但反向传播时 AE MSE 梯度通过 K_b/V_b 传入 backbone 的幅度缩放为原来的 `scale` 倍。backbone 收到 `∇CE + scale * ∇MSE_via_KV`。推荐 `gradient_scale: 0.1~0.3`。

### `"freeze_backbone"` — 冻结 backbone

backbone 完全冻结，VLM 和 AE 均无法更新 backbone。适合只需微调 AE 组件的场景。

---

## 九、训练循环设计

```python
for global_step in range(num_steps):
    # 1. 更新 LR (cosine decay with warmup)
    lr = cosine_lr(global_step, warmup, peak_lr, decay_steps, end_lr)

    # 2. VLM forward + backward（不同步 DDP 梯度）
    with model.no_sync():
        vlm_loss = model(mode="vlm", batch=vlm_batch)
        vlm_loss.backward()

    # 3. AE forward + backward（触发 DDP 梯度同步）
    ae_loss = model(mode="ae", observation=ae_obs, ...)
    (ae_loss_weight * ae_loss).backward()

    # 4. 优化器更新
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

**关键设计决策：**

1. **`model.no_sync()` + DDP**：VLM backward 不触发 allreduce，AE backward 触发 allreduce，同步累积的梯度。省去一次 allreduce 通信开销。
2. **`forward(mode, **kwargs)`** 分发器：DDP 要求所有 forward 调用经过 `nn.Module.__call__`，直接调用 `model.module.vlm_forward()` 会绕过 DDP 导致梯度不同步。
3. **两个独立 DataLoader**：VLM 和 AE 使用不同的数据集，各自有独立的 batch_size、shuffle_buffer 和 image augmentation 配置。
4. **显存峰值 ≈ max(VLM_peak, AE_peak)**：每次 backward 释放计算图后再做下一次 forward，不同时持有两个计算图的激活。
5. **`ae_loss_weight`**：AE loss 乘以权重系数后再 backward，控制 AE 梯度相对 VLM 梯度的比例。

### Episode Shuffle (多 Suite 联合训练)

采用两层 shuffle 策略解决多 suite 数据按顺序排列的问题：

**第 1 层：Episode 级 shuffle (TF dataset 层)** — 在 TFDS `as_dataset()` 之后，对 episode 调用 `dataset.shuffle(buffer_size=N)`，打乱 episode 读取顺序。

**第 2 层：Sample 级 shuffle (Python reservoir buffer)** — 现有的 `__iter__()` shuffle buffer，负责 episode 内部和相邻 episode 间的样本混合。

**AE 的 `ep_idx` 映射**：使用 `tf.data.Dataset.enumerate()` 在 shuffle 之前绑定原始索引，确保 shuffle 后仍能正确查找 `waypoint_indices.json`：

```python
dataset = dataset.enumerate()        # (orig_idx, episode) pairs
dataset = dataset.shard(W, rank)     # DDP 分片
dataset = dataset.shuffle(buf)       # episode 级 shuffle
```

---

## 十、训练与评测命令

### 联合训练

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

# 断点续训: 在配置文件中设置 resume: true，或添加 --resume
```

### 归一化统计量生成

```bash
.venv/bin/python scripts/compute_wp_norm_stats.py \
  --rlds_dir /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0 \
  --robot_type libero \
  --output_dir /workspace/data/libero/libero_object_wp_001/norm_stats
```

### LIBERO 评测

```bash
# 安装评测依赖
uv pip install --python .venv/bin/python \
    robosuite==1.4.1 transforms3d bddl easydict "gym==0.26.2"
uv pip install --python .venv/bin/python -e third_party/libero

# 运行评测（单 GPU，联合模型 ~14 GB 显存）
MUJOCO_GL=osmesa \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH \
.venv/bin/python -u -m openpi.waypoint.eval_libero \
    --config configs/eval_waypoint_joint_libero.yaml
```

评测配置中需设置联合 checkpoint 路径：
```yaml
joint_checkpoint: /path/to/joint/checkpoint_dir    # 须含 model.safetensors
video_out_path: data/libero/videos_wp
```

### CALVIN 训练与评测

```bash
# 1. 生成归一化统计量
.venv/bin/python scripts/compute_wp_norm_stats.py \
  --rlds_dir /workspace/calvin_abc_wp_001/calvin_abc_wp/1.0.0 \
  --robot_type calvin \
  --output_dir /workspace/calvin_abc_wp_001/norm_stats

# 2. 联合训练
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_waypoint_joint.py \
    --config configs/waypoint_joint_calvin.yaml

# 3. 评测
CALVIN_ROOT=/path/to/calvin \
PYTHONPATH=$PWD:$PYTHONPATH \
.venv/bin/python -u -m openpi.waypoint.eval_calvin \
    --config configs/eval_waypoint_joint_calvin.yaml
```

---

## 十一、配置文件关键参数

### `waypoint_joint_libero.yaml`

```yaml
robot_type: libero

# --- Data paths ---
original_rlds_dir: /workspace/data/object/libero_object_no_noops/libero_object_no_noops/1.0.0
wp_indices_path: /workspace/data/object/libero_object_wp_001/waypoint_indices.json
wp_rlds_dir: /workspace/data/object/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0
dataset_statistics_path: /workspace/openpi/data

# --- Model architecture ---
paligemma_variant: gemma_2b
action_expert_variant: gemma_300m
precision: bfloat16

# Model dimensions (AE)
model_action_dim: 32
model_proprio_dim: 32
horizon_steps: 32
max_duration: 32
ae_max_token_len: 64

# Waypoint config (VLM)
num_waypoints: 7
vlm_max_token_len: 256
stride: 1

# Pretrained weights
pretrained_weight_path: /workspace/models/pi05_base_pytorch

# --- Joint training ---
gradient_strategy: scale_gradient
gradient_scale: 0.3
ae_loss_weight: 1.0

# Batch sizes (per GPU)
vlm_batch_size: 192
ae_batch_size: 192

# Training hyperparameters
num_train_steps: 1200
warmup_steps: 40
peak_lr: 1.0e-4
end_lr: 1.0e-7
norm_type: q99

# Image augmentation — VLM
image_aug: true
image_aug_cfg:
  random_resized_crop_scale: [0.9, 1.0]
  brightness: 0.2
  contrast: [0.8, 1.2]
  saturation: [0.8, 1.2]
  hue: 0.05

# Image augmentation — AE
ae_image_aug_cfg:
  crop_scale: 0.95
  rotation_deg: 5.0
  brightness_lo: 0.7
  brightness_hi: 1.3
  contrast_lo: 0.6
  contrast_hi: 1.4
  saturation_lo: 0.5
  saturation_hi: 1.5

# Shuffle buffers
vlm_shuffle_buffer_size: 5000
ae_shuffle_buffer_size: 1000

# Logging and checkpointing
exp_name: waypoint_joint_libero_sg_03
checkpoint_dir: checkpoints/{exp_name}
save_interval: 100
log_interval: 3
wandb_enabled: true
wandb_project: waypoint_e2e

# LoRA
lora_enabled: false
lora_rank: 16
lora_alpha: 16.0
train_vision_encoder: true

# Resume
resume: false
```

### 多 Suite 联合训练额外参数

```yaml
episode_shuffle_buffer_size: 500
vlm_shuffle_buffer_size: 10000
ae_shuffle_buffer_size: 5000
num_train_steps: 5000
warmup_steps: 100
```

---

## 十二、Wandb 日志

Project: `waypoint_e2e`，每 `log_interval` 步记录：

| Metric | 含义 |
|--------|------|
| `train/vlm_loss` | VLM CE loss |
| `train/ae_loss` | AE MSE loss |
| `train/total_loss` | `vlm_loss + ae_loss_weight * ae_loss` |
| `train/lr` | 当前 learning rate |
| `train/grad_norm` | 梯度 L2 范数 |
| `train/steps_per_sec` | 训练速度 |

Summary: `total_params`, `trainable_params`, `gradient_strategy`, `ae_loss_weight`

---

## 十三、Proprio 300-bin + Gripper 二值化 Token

### 数据流总览

```
原始 state (8D): [eef_x, eef_y, eef_z, rot1, rot2, rot3, grip_qpos_L, grip_qpos_R]
  ├─→ continuous = state[0:6]  → q99 normalize → 300-bin quantize → 6 proprio tokens
  └─→ gripper = state[6] > 0.02 → binary → 1 gripper token {<grip_open> / <grip_close>}

VLM 解码 waypoint: 6D continuous (300-bin decode) + 1D gripper (1.0/0.0)
  = 7D proprio → pad_to_dim(32) → AE proprio_encoder(Linear(7, W))
```

---

## 十四、已知注意事项

### 权重加载
- `time_mlp_in` 从 `[1024, 1024]` 扩展为 `[2048, 1024]`，不能用 `strict=False`，需手动循环 state_dict 跳过 shape 不匹配的 key

### 归一化
- Gripper (dim 6) 不参与 q99 归一化 (`action_norm_mask[-1] = False`)
- VLM 和 AE 使用**相同**的 q99 统计量

### 图像张量格式
- 数据集 collator 输出图像为 `(B, C, H, W)` float32 [-1,1]
- 必须在 collator 中做 `imgs.transpose(0, 3, 1, 2)` 把 `(B, H, W, C)` 转为 `(B, C, H, W)`

### DDP 后端
- 使用 `backend = "gloo"` (不用 nccl)，nccl 在该环境会导致 `CUDA illegal memory access`

### VLM-AE 接口
- VLM 输出 proprio 在 **q99 归一化空间** [-1, 1]
- AE 的 start/end_proprio 输入同样在 **q99 归一化空间**
- 两者使用相同的统计量 → 无需额外变换

### VLM Gradient Checkpointing
- 必须通过 HuggingFace API 激活：`self.paligemma.gradient_checkpointing_enable()`
- 仅设置 `gradient_checkpointing = True` 属性**无效**

### VLM Shuffle Buffer 启动策略
- VLM dataset 的 `__iter__` 必须使用 early-yield 策略（buffer 有 32 条即开始 yield）
- 若等待 buffer 完全填满，首个 batch 需 ~6 分钟

### 依赖项
- 需手动安装 TF: `uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"`

### torchrun 路径
- 必须使用 `.venv/bin/torchrun`

### VLM Logit Masking
- 自由预测位置必须做 logit masking，将 argmax 限制在合法 token 子集内
- proprio 位置仅允许 token 256724–257023，duration 位置仅允许 token 256690–256723，其余设为 `-inf`

### 归一化统计量路径一致性
- 训练和评测**必须**使用完全相同的 `dataset_statistics.json`

### 视频录制
- 每个 episode 录制 agentview 相机的 256×256 图像
- 使用 `imageio.mimwrite()` 保存为 MP4，10 FPS
- 文件名格式：`rollout_{task_name}_t{trial}_{success|failure}.mp4`

### RLDS 数据路径
- TFDS `builder_from_directory` 需要指向**直接包含** `dataset_info.json` 的目录

---

## 十五、Batch Size 调优 (2x RTX PRO 6000 Blackwell 97.9GB)

每个 rank 独立运行 DataLoader，`batch_size` 是**单 GPU** 的 batch 大小。

### 内存优化手段
1. **Masked logits**: 只对 loss_mask=True 的位置 (~63 tokens) 计算 LM head → 节省 ~40 GB
2. **bfloat16**: 模型参数+优化器状态减半 → 节省 ~20 GB
3. **跳过 dummy image**: LIBERO 只有 2 个 camera，不再处理全零的第 3 张图 → 节省 ~2 GB + 序列缩短 25%

---

## 十六、CALVIN 评测管线 (`eval_calvin.py`)

```
for seq_i in 1..1000:
    initial_state, [subtask_1, ..., subtask_5] = get_sequences()
    env.reset(robot_obs, scene_obs)
    for subtask in subtasks:
        success = run_calvin_subtask(env, vlm, ae, subtask, ep_len=360)
        if not success: break
    results.append(num_consecutive_successes)

report: avg_seq_len, chain_sr[1/5 .. 5/5]
```

**`run_calvin_subtask()` 关键逻辑:**
1. 从 CALVIN env 获取 `obs["robot_obs"]` (15D) 和 `obs["rgb_obs"]` (static + gripper)
2. `rc.split_proprio()` 拆分 15D → 6D continuous + binary gripper
3. VLM 预测 waypoints → AE 填充动作 → 执行
4. Gripper post-processing: `x*2-1 → sign` (**不 negate**，因 CALVIN env 期望 +1=open)
5. 每步通过 `task_oracle.get_task_info_for_set()` 检查任务完成

**Center crop 支持:**
- `get_calvin_images()` 接受 `center_crop_scale` 参数
- 默认配置 `center_crop_scale: 0.9`，与训练时的 `RandomResizedCrop(scale=[0.9, 1.0])` 匹配
