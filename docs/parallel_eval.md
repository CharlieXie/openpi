# 并行评估设计文档

## 运行方式

```bash
# 基本运行（num_parallel 默认等于 num_tasks，全任务并行）
cd /workspace/exp/openpi
python -m openpi.waypoint.eval_libero --config configs/eval_waypoint_libero.yaml

# 控制并发 env 数量（显式指定，适合内存受限场景）
python -m openpi.waypoint.eval_libero --config configs/eval_waypoint_libero.yaml
# 在 YAML 中设置: num_parallel: 4
```

YAML 所有可配项：

```yaml
# Evaluation
num_trials_per_task: 50    # 每个 task 跑多少 trial
num_steps_wait: 10         # 每个 trial 开头的 warm-up dummy action 步数
num_parallel: 10           # 同时运行的 task 环境数（默认 = num_tasks）
```

---

## 背景与问题

原始串行评估的执行顺序为：

```
for task:
  for trial:
    while not done:
      VLM()          # 全精度推理，batch=1
      for wp in waypoints:
        AE()         # 全精度推理，batch=1
        execute actions
```

Action Expert (AE) 是推理瓶颈：每次 replan 最多调用 7 次（waypoint 数），每次调用包含 10 步 flow-matching 去噪，batch size 始终为 1，GPU 利用率极低。

**直接拼 batch（单 episode 内多 waypoint pair）行不通**，原因在于不同 waypoint pair 的 duration 不同，实际执行步数也不同，且每个 AE 调用需要最新的 obs 图像，难以提前预计算并行。

---

## 解计思路

核心思路的转变：**不把一个 episode 内的多个 waypoint pair 拼 batch，而是把多个 episode 各自当前的推理请求跨 episode 拼成一个 batch**。

每次 AE 调用的输入 shape 是固定的：

| 字段 | Shape |
|------|-------|
| 图像 | `(B, 3, 224, 224)` |
| 语言 token | `(B, 64)` |
| start\_proprio | `(B, 32)` |
| end\_proprio | `(B, 32)` |
| duration | `(B,)` |
| 输出 actions | `(B, 32, 32)` |

B 个 episode 同时处于"需要 AE 推理"状态时，只需在 `dim=0` 拼接，一次 forward 解决所有请求。

---

## 架构设计

### 状态机

每个 episode 对应一个 `_Slot` 对象，维护以下 5 个状态：

```
                   ┌────────────────────────────────────┐
                   │                                    │
              ┌────▼────┐    warm-up done    ┌──────────┴──────────┐
   trial  ───►│  WAIT   ├───────────────────►│     NEED_VLM        │
   start       └─────────┘                   └──────────┬──────────┘
                                                        │ waypoints ready
                                                        ▼
                                             ┌──────────────────────┐
                    ┌────────────────────────┤      NEED_AE         │
                    │  next valid wp         └──────────┬───────────┘
                    │                                   │ actions ready
                    │                                   ▼
                    │                        ┌──────────────────────┐
                    │   actions exhausted    │      EXECUTING       │
                    └────────────────────────┤  (step 1 action/tick)│
                                             └──────────┬───────────┘
                                                        │ done / timeout
                                                        ▼
                                             ┌──────────────────────┐
                                             │        DONE          │
                                             └──────────────────────┘
```

**关键设计**：`EXECUTING` 状态每个 tick 只执行 **1 步 env.step()**，这样不同 episode 可以在同一个主循环 tick 中共存。

### 主循环（每个 tick 的执行顺序）

```python
while slots:
    # 1. WAIT   → 各自 step dummy action，warm-up 结束后转 NEED_VLM
    # 2. EXECUTING → 各自 step 一个 action，action 耗尽后转 NEED_AE 或 NEED_VLM
    # 3. NEED_VLM  → 各自串行跑 VLM（autoregressive，需顺序推理）
    # 4. NEED_AE   → 收集所有 NEED_AE 的 slot，一次 batched AE forward
    # 5. DONE      → 记录结果，开启下一个 trial，或替换为新 task
```

### 动态 Batching（`_batched_predict_actions`）

```python
def _batched_predict_actions(ae_model, requests, device, pg_tok):
    # requests: list of {images, instruction, start_wp, end_wp, duration}
    N = len(requests)
    # 拼接所有 N 个 episode 的输入
    img_tensors = {key: torch.stack([...]) for key in IMAGE_KEYS}   # (N, 3, 224, 224)
    prompt_tokens = torch.tensor([...])                              # (N, 64)
    start_t = torch.stack([...])                                     # (N, 32)
    end_t   = torch.stack([...])                                     # (N, 32)
    dur_t   = torch.tensor([...])                                    # (N,)
    # 一次 AE forward（内部包含 10 步去噪循环）
    actions = ae_model.sample_actions(obs_batch, start_t, end_t, dur_t)
    # actions: (N, 32, 32)  → 拆回各 episode
    return [actions[i].cpu().numpy() for i in range(N)]
```

### KV Cache 正确性

AE 的 `sample_actions` 内部分两阶段：

1. **Prefix pass**（`use_cache=True`）：将 image + language token 经 PaliGemma 计算出 `past_kv`，写入 `DynamicCache`。
2. **10 步去噪循环**（`use_cache=False`）：每步只读 concat prefix KV，**不修改** cache（代码中 `else: key_states = torch.cat([past_kv[l][0], suffix_key], dim=2)`）。

因此 batch 内多个 episode 的 KV 完全通过 **batch 维度** 隔离：

```
past_kv[layer][0]  →  (N, num_kv_heads, prefix_len, head_dim)
                        [0] = episode 0 的 prefix K
                        [1] = episode 1 的 prefix K
                        ...
```

**不存在跨 episode 的 KV 污染问题**，batch 化对推理结果无任何影响。

---

## 与串行版本的等价性

| 项目 | 串行版 | 并行版 |
|------|--------|--------|
| VLM 推理 | batch=1，顺序 | batch=1，顺序（不变） |
| AE 推理 | batch=1 | batch=N（跨 episode） |
| AE 每次调用是否用最新图像 | 是 | 是（fresh\_images 在 NEED\_AE 时抓取） |
| AE prefix KV cache 复用 | 无 | 无（每次调用独立算 prefix） |
| Action noise 分布 | randn(1, H, D) | randn(N, H, D) 拆分 |
| 确定性 | 视 seed 而定 | 与串行不完全一致（noise 顺序不同） |
| 成功率期望 | — | **与串行完全等价** |

> **精度保证**：每个 AE 调用使用最新图像（`NEED_AE` 时 `get_libero_images(s.env, s.obs)`），不共享前缀 KV cache，因此与方案二不同，**没有任何精度损失**。

---

## 性能分析

设 `N` = num\_parallel，AE 单次推理时间 `T_ae`，env step 时间 `T_env`，VLM 推理时间 `T_vlm`：

- 原始串行：`N × (T_vlm + k × T_ae + steps × T_env)`（k = waypoints 数）
- 并行后：`T_vlm_serial + T_ae_batched/N + steps × T_env_parallel`

AE 是主要瓶颈时，理论加速比接近 `N`（受限于 GPU 显存和 batch 饱和点）。

实际瓶颈转移：当 N 较大时，VLM 串行推理和 env.step() 的 CPU 开销会成为新的瓶颈，可进一步通过：
- VLM 也做跨 episode batch（需对齐 prompt 长度，用 left-padding）
- env.step() 多进程异步（SubprocVecEnv）

来进一步优化，但这超出了当前方案的范围。

---

## 文件变更

| 文件 | 变更内容 |
|------|----------|
| `src/openpi/waypoint/eval_libero.py` | 重写 `evaluate()`；新增 `_EpState`、`_Slot`、`_batched_predict_actions`、`_record_replay_frame` |
| `configs/eval_waypoint_libero.yaml` | 新增 `num_parallel` 注释说明 |
