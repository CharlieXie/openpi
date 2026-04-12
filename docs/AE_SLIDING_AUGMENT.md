# AE Sliding Window Data Augmentation — 背景与实现

## 一、问题背景

### Waypoint Joint Model 的两段式架构

Waypoint Joint Model 基于 Pi0.5，采用 VLM + Action Expert (AE) 的两段式架构：

1. **VLM** 自回归预测稀疏 waypoint 轨迹：每个 waypoint 包含 `(proprio, duration)`
2. **AE** 在相邻 waypoint 之间填充密集动作：输入 `(images, instruction, start_proprio, end_proprio, duration)` → 输出 action chunk

相比原始 Pi0.5，waypoint joint model 的 AE 多了 `start_proprio`、`end_proprio`、`duration` 三个条件信号。

### 数据量差异 — 核心问题

原始 Pi0.5 和 waypoint AE 在训练数据量上有显著差距：

| Pipeline | 数据提取方式 | LIBERO-10 训练样本数 |
|----------|-------------|---------------------|
| **Pi0.5 原始** | stride=1 滑动窗口，每个 timestep 产生 1 条样本 | ~98,000 |
| **Waypoint AE（增强前）** | 每个 waypoint pair 恰好产生 1 条样本 | ~12,486 |

**差距约 8 倍。**

原因：waypoint 提取（err_threshold=0.006）将 101,469 步压缩到 16,138 个 waypoint，相邻 waypoint 形成 ~12,486 个 valid pair（duration ≤ 32）。每个 pair 只产生一条训练样本。

### 数据来源

- 原始 RLDS 数据：`/workspace/data/libero_10_no_noops/1.0.0`（379 个 episode，101,469 步）
- Waypoint 索引：`/workspace/data/libero_10_wp_0006/waypoint_indices.json`（err_threshold=0.006）
- Waypoint-filtered RLDS（VLM 用）：`/workspace/data/libero_10_wp_0006/waypoint_filtered_rlds__libero/1.0.0`

`waypoint_indices.json` 结构示例：
```json
{
  "config": {
    "total_episodes": 379,
    "total_src_steps": 101469,
    "total_wp_steps": 16138,
    "err_threshold": 0.006
  },
  "episodes": [
    {
      "src_ep_idx": 0,
      "original_steps": 214,
      "waypoint_indices": [0, 14, 21, 28, 37, 40, ..., 207, 213]
    }
  ]
}
```

---

## 二、增强方案：滑动窗口

### 核心思想

对于一个 waypoint pair `(w_start=0, w_end=10, duration=10)`，原始只产生 1 条数据：

```
原始: actions[0:10], start_proprio=state[0], end_proprio=state[10], duration=10
```

增强后，保持 duration 不变，在轨迹上滑动窗口，产生 D 条数据：

```
shift=0: actions[0:10],  start=state[0],  end=state[10],  duration=10  (原始)
shift=1: actions[1:11],  start=state[1],  end=state[11],  duration=10
shift=2: actions[2:12],  start=state[2],  end=state[12],  duration=10
...
shift=9: actions[9:19],  start=state[9],  end=state[19],  duration=10
```

### 为什么语义上合理

每条增强数据中：
- `start_state` 和 `end_state` 是轨迹上真实的机器人状态
- `actions[new_start:new_end]` 是真实执行过的动作序列
- `duration` 精确等于动作数量
- 图像来自 `new_start` 时刻的真实观测

AE 的任务是学习 `(images, start_state, end_state, duration) → actions`。增强数据中这个映射关系完全成立——它描述了一段真实发生的机器人轨迹。

### 数据量效果

增强后每个 pair 产生 D 条样本，总量 = 所有 valid pair 的 duration 之和 ≈ 90,000-100,000。

| 模式 | 训练样本数 | 相对 Pi0.5 |
|------|-----------|-----------|
| 增强前 | ~12,486 | ~1/8 |
| **增强后** | **~90,000-100,000** | **~1/1** |

---

## 三、实现细节

### 改动文件

1. `src/openpi/waypoint/ae_dataset.py` — 核心逻辑
2. `scripts/train_waypoint_joint.py` — 参数透传
3. `configs/waypoint_joint_libero.yaml` — 配置开关

### 3.1 `ae_dataset.py` 的改动

#### `__init__` 新增参数

```python
ae_sliding_augment: bool = False  # 默认关闭，向后兼容
```

#### `__init__` 中存储 episode 长度并计算准确的总样本数

从 `waypoint_indices.json` 读取每个 episode 的 `original_steps`，用于在 init 阶段精确计算增强后的数据总量：

```python
self.episode_lengths: dict[int, int] = {}
# 对每个 valid pair:
total_augmented += min(dur, max(0, ep_len - w_start - dur))
```

公式说明：shift `s` 需满足 `new_end = w_start + s + dur < ep_len`，故 `s < ep_len - w_start - dur`；同时 `s < dur`。

#### `_process_episode` 的改动

**预计算所有 proprio**（与已有的 `all_actions` 预计算并列）：

```python
all_proprios_7d = np.empty((num_steps, rc.continuous_proprio_dim + 1), dtype=np.float32)
for t in range(num_steps):
    obs_dict = {k: steps[t]["observation"][k] for k in steps[t]["observation"]}
    raw = extract_proprio_from_obs(obs_dict, rc.state_obs_keys)
    cont, grip = rc.split_proprio(raw)
    cont_norm = self.norm_helper.normalize_proprio(cont)
    all_proprios_7d[t, :rc.continuous_proprio_dim] = cont_norm
    all_proprios_7d[t, rc.continuous_proprio_dim] = float(grip)
```

**在 pair 循环中加入 shift 内层循环**：

```python
for w_start, w_end, duration in wp_pairs:
    shifts = range(duration) if self.ae_sliding_augment else range(1)
    for shift in shifts:
        new_start = w_start + shift
        new_end = new_start + duration
        if new_end >= num_steps:
            break  # 必须是 break，不是 continue

        # 图像: steps[new_start]（不是 w_start）
        # proprio: all_proprios_7d[new_start] 和 all_proprios_7d[new_end]
        # actions: all_actions[new_start:new_end]
        # duration: 不变
        yield sample
```

关键点：
- **`break` 而非 `continue`**：shift 递增，一旦越界后续 shift 必然也越界
- **图像取自 `steps[new_start]`**：图像必须对应滑动后的起始时刻
- **`shift=0` 路径**输出与原代码完全一致，确保向后兼容
- **不预缓存图像**：内存代价太高（~82MB/episode），且图像增强是随机的

### 3.2 `train_waypoint_joint.py` 的改动

在构造 `WaypointAEDataset` 时透传参数：

```python
ae_sliding_augment=cfg.get("ae_sliding_augment", False),
```

### 3.3 `waypoint_joint_libero.yaml` 的改动

```yaml
ae_sliding_augment: true
ae_shuffle_buffer_size: 5000  # 从 1000 增大，因为每个 episode 产生更多样本
```

---

## 四、不受影响的组件

| 组件 | 原因 |
|------|------|
| `WaypointAECollator` | 样本 dict 格式完全不变 |
| `WaypointVLMDataset` / `WaypointVLMCollator` | VLM 数据与 AE 增强无关 |
| `joint_model.py` (PI0WaypointJoint) | AE forward 接收的张量格式不变 |
| `normalize.py` | 归一化逻辑不变 |
| `robot_config.py` | 机器人配置不变 |
| `_raw_sample_iter` / `__iter__` | 只是 `_process_episode` 的外层迭代器和 shuffle buffer |

---

## 五、边界条件处理

| 场景 | 处理方式 |
|------|----------|
| shift 后 `new_end` 超出 episode 末尾 | `if new_end >= num_steps: break` |
| episode 最后一个 pair 紧贴末尾 | 公式 `min(dur, max(0, ep_len-w_start-dur))` 自然处理 |
| `duration=1` 的极短 pair | `range(1)` 只有 shift=0，与不增强一致 |
| `ae_sliding_augment=False` | `range(1)` 保证只有 shift=0，输出与原代码一致 |

---

## 六、注意事项

1. **Shuffle buffer 大小**：增强后每个 episode 产生 ~211 个样本（vs 原来 ~33 个）。`ae_shuffle_buffer_size` 应从 1000 增大到 5000+，确保 buffer 中混合足够多 episode 的数据。

2. **训练-推理分布偏移**：推理时 AE 的 start/end proprio 来自 VLM 预测的 waypoint 位置。增强数据中的 start/end 可能不在"语义 waypoint"位置上。但这反而有助于 AE 对微小偏移的鲁棒性。

3. **跨 waypoint pair 边界**：shift 较大时，action 窗口会跨越当前 pair 的终点进入后续 pair 的领地。这不影响正确性——action 仍是真实轨迹，(start, end, duration) → actions 映射仍然成立。

4. **安装问题**：修改 `ae_dataset.py` 后，如果包是通过 `pip install -e .` 安装的，需要确认 Python 导入的是源文件而非缓存的 `.pyc`。如果出现 `unexpected keyword argument 'ae_sliding_augment'` 错误，说明运行时加载的不是修改后的文件，需检查 `PYTHONPATH` 或重新安装包。
