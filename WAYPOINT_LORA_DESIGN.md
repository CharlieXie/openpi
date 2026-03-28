# Waypoint Joint Model — LoRA 设计文档

> PEFT 风格的 LoRA 实现，用于 PI0WaypointJoint 联合模型的参数高效微调。

---

## 一、设计目标

1. **零拷贝注入** — 包装原始 `nn.Linear` 为 `LoRALinear`，不 `.clone()` 权重，不浪费内存
2. **默认全覆盖 + 白名单排除** — 所有 `nn.Linear` 默认用 LoRA，通过 `modules_to_not_lora` 排除不适合的层
3. **LoRA-only 保存/加载** — 只保存可训练参数（LoRA A/B + 白名单全量训练层），大幅减小 checkpoint 体积
4. **DDP 兼容** — 冻结的 base 权重不参与梯度同步
5. **与 gradient_strategy 兼容** — LoRA 模式下 scale_gradient / stop_gradient 正常工作

---

## 二、与旧实现的对比

| 维度 | 旧实现 (`LoRALinear` 替换模式) | 新实现 (PEFT 包装模式) |
|------|------|------|
| 权重处理 | `.clone()` → 内存翻倍 | `self.base_layer = linear` → 零拷贝 |
| 覆盖方式 | 白名单 target_modules | 全覆盖 + `modules_to_not_lora` 排除 |
| LoRA 参数 | `nn.Parameter` 裸张量 | `nn.Linear` 子模块（与 PEFT 一致） |
| 保存 | 全量 state_dict | LoRA-only（可训练参数） |
| merge | 不可逆，丢弃 LoRA 参数 | merge 后替换回 base_layer |
| 视觉编码器 | `train_vision_encoder: bool` | `vision_encoder_mode: freeze/lora/full` |

---

## 三、模型完整 Linear 层清单

### PI0WaypointJoint (顶层)

| 层 | 尺寸 | LoRA? | 理由 |
|----|------|-------|------|
| `action_in_proj` | 32→1024 | **否** — 全量训练 | 输入维度 32，rank=16 无压缩空间 |
| `action_out_proj` | 1024→32 | **否** — 全量训练 | 输出维度 32，同上 |
| `proprio_encoder` | 32→1024 | **否** — 全量训练 | 同 action_in_proj |
| `time_mlp_in` | 2048→1024 | **否** — 全量训练 | 联合模型新增层，必须从头训练 |
| `time_mlp_out` | 1024→1024 | **否** — 全量训练 | 配套 time_mlp_in |

### PaliGemma (VLM backbone)

| 层 | 尺寸 | LoRA? | 理由 |
|----|------|-------|------|
| `multi_modal_projector.linear` | 1152→2048 | **否** — 全量训练 | 单层，参数量小 |
| `embed_tokens` | Emb(257152, 2048) | **否** — 全量训练 | 含新增 waypoint token |
| `lm_head` | 2048→257152 | **否** — 全量训练 | 与 embed_tokens tied |

### SigLIP Vision Tower (27 layers)

| 层 | 尺寸 | LoRA? | 说明 |
|----|------|-------|------|
| `q_proj`, `k_proj`, `v_proj` | 1152→1152 | 可配置 | `vision_encoder_mode` 控制 |
| `out_proj` | 1152→1152 | 可配置 | 同上 |
| `fc1` | 1152→4304 | 可配置 | 同上 |
| `fc2` | 4304→1152 | 可配置 | 同上 |

### Gemma 2B Language Model (18 layers) — LoRA

| 层 | 尺寸 | LoRA |
|----|------|------|
| `q_proj` | 2048→2048 | **是** |
| `k_proj` | 2048→256 | **是** |
| `v_proj` | 2048→256 | **是** |
| `o_proj` | 2048→2048 | **是** |
| `gate_proj` | 2048→16384 | **是** |
| `up_proj` | 2048→16384 | **是** |
| `down_proj` | 16384→2048 | **是** |

### Gemma 300M Action Expert (18 layers) — LoRA

| 层 | 尺寸 | LoRA |
|----|------|------|
| `q_proj` | 1024→1024 | **是** |
| `k_proj` | 1024→1024 | **是** |
| `v_proj` | 1024→1024 | **是** |
| `o_proj` | 1024→1024 | **是** |
| `gate_proj` | 1024→4096 | **是** |
| `up_proj` | 1024→4096 | **是** |
| `down_proj` | 4096→1024 | **是** |

### embed_tokens / lm_head — 重要说明

| 层 | 类型 | 尺寸 | LoRA? | 理由 |
|----|------|------|-------|------|
| `embed_tokens` | `nn.Embedding` | 257152×2048 (527M) | **不能** | 不是 `nn.Linear`，当前 LoRA 不支持 Embedding |
| `paligemma.lm_head` | `nn.Linear` | 2048→257152 | **不能** | weight 与 embed_tokens **tied**（同一 tensor） |
| `gemma_expert.lm_head` | `nn.Linear` | 1024→257152 (263M) | **否** — 冻结 | embed_tokens=None, 未使用 |

**weight tying 关键分析：**
- `lm_head.weight IS embed_tokens.weight` — HuggingFace `post_init()` / `tie_weights()` 使它们是同一个 tensor
- Joint model **从不调用** `lm_head.forward()`，而是直接用 `embed_tokens.weight` 做 `F.linear()`
- 如果 LoRA wrap `lm_head`：(a) LoRA forward 永远不被调用 → 死参数 (b) state_dict key 变化 → 打断 weight tying
- PEFT 也显式排除 `get_output_embeddings()` (见 `tuners_utils.py`)

**在全量训练（无 LoRA）中的占比：**
- `embed_tokens` (=lm_head, tied): 527M = **全部 3.5B 参数的 15%**
- `expert.lm_head` (独立, 未使用): 263M = 7.5%
- 两者合计: 790M = **22.5% 的参数用于从不使用的 lm_head 或不可 LoRA 的 Embedding**

---

## 四、白名单机制

LoRA 使用两个列表控制每个层的行为：

```
所有 nn.Linear
  ├── 在 lora_modules_to_skip 中？ → 不加 LoRA
  │     ├── 在 lora_trainable_modules 中？ → 全量训练 (requires_grad=True)
  │     └── 不在？ → 冻结 (requires_grad=False)
  └── 不在？ → 加 LoRA (base 权重冻结, lora_A/B 可训练)
```

### `lora_modules_to_skip`（不加 LoRA 的层）

包含模块路径子串。任何 `nn.Linear` 的 `named_modules()` 路径匹配其中一项就跳过 LoRA。

| 默认值 | 理由 |
|--------|------|
| `action_in_proj` | 输入维度 32，rank=16 无压缩空间 |
| `action_out_proj` | 输出维度 32，同上 |
| `proprio_encoder` | 同 action_in_proj |
| `time_mlp_in` | 联合模型新增层 (2048→1024)，必须全量训练 |
| `time_mlp_out` | 配套 time_mlp_in |
| `multi_modal_projector` | 视觉-语言桥接，单层参数量小 |
| `embed_tokens` | `nn.Embedding`，不是 `nn.Linear`，不能 LoRA |
| `lm_head` | 与 embed_tokens weight tied，不能 LoRA |

### `lora_trainable_modules`（跳过的层中哪些保持可训练）

| 默认值 | 理由 |
|--------|------|
| `action_in_proj` | AE 专用，必须训练 |
| `action_out_proj` | AE 专用，必须训练 |
| `proprio_encoder` | AE 专用，必须训练 |
| `time_mlp_in` | 时间条件 MLP，必须训练 |
| `time_mlp_out` | 时间条件 MLP，必须训练 |
| `multi_modal_projector` | 视觉-语言桥接，需适配 |
| `embed_tokens` | 包含 waypoint token (300 proprio + 34 duration + 4 special)，预测 waypoint 需要训练 |
| `lm_head` | 与 embed_tokens tied（同一 tensor），需一起训练 |

不在此列表的跳过层（如 `gemma_expert.lm_head`）会被冻结。

### 实际效果示例

| 层 | 在 skip? | 在 trainable? | 结果 |
|---|---|---|---|
| `language_model.layers.0.self_attn.q_proj` | 否 | — | **LoRA** |
| `action_in_proj` | 是 | 是 | **全量训练** |
| `embed_tokens` | 是 | 是 | **全量训练** |
| `gemma_expert.lm_head` | 是 | 否 | **冻结** |

---

## 五、参数量估计 (rank=16)

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Gemma 2B backbone LoRA | 19.6M | 18层 × 7 linear, rank=16 |
| Gemma 300M expert LoRA | 6.8M | 18层 × 7 linear, rank=16 |
| AE heads (全量训练) | 3.2M | action/proprio/time projections |
| multi_modal_projector (全量训练) | 2.4M | 视觉-语言桥接 |
| embed_tokens (全量训练) | 527M | waypoint token 预测必须训练 |
| **总可训练** | **~559M** | **~16% of model** |
| 总冻结 | ~2,941M | backbone + expert base weights + SigLIP |

> embed_tokens 占可训练参数的 94%。若需更激进的参数高效，可在 config 中
> 从 `lora_trainable_modules` 移除 `embed_tokens` 和 `lm_head`，
> 可训练参数降至 32M (0.91%)。

---

## 六、配置示例

```yaml
# configs/waypoint_joint_libero.yaml

lora_enabled: true
lora_rank: 16
lora_alpha: 16.0
lora_dropout: 0.0
lora_use_rslora: false
lora_init: true                 # true/"kaiming" (default) or "gaussian"
lora_apply_to: all              # "all" | "backbone_only" | "expert_only"
vision_encoder_mode: freeze     # "freeze" | "lora" | "full"

# 白名单配置 (可选，一般不需要修改)
# lora_modules_to_skip:         # 不加 LoRA 的层
#   - action_in_proj
#   - action_out_proj
#   - proprio_encoder
#   - time_mlp_in
#   - time_mlp_out
#   - multi_modal_projector
#   - embed_tokens
#   - lm_head
# lora_trainable_modules:       # 跳过的层中哪些保持可训练
#   - action_in_proj
#   - action_out_proj
#   - proprio_encoder
#   - time_mlp_in
#   - time_mlp_out
#   - multi_modal_projector
#   - embed_tokens
#   - lm_head
```

---

## 七、核心实现 (`lora_pytorch.py`)

### LoRALinear 包装层

```python
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        self.base_layer = base_layer          # 零拷贝引用
        self.lora_A = nn.Linear(in, rank, bias=False)
        self.lora_B = nn.Linear(rank, out, bias=False)
        # B = zeros (always), A = kaiming (default) or gaussian
        # → 初始 LoRA 输出为零

    def forward(self, x):
        return base_layer(x) + lora_B(lora_A(dropout(x))) * scaling
```

### 注入流程

```
apply_lora_to_model(model, config)
  ├─ 遍历所有 named_modules
  ├─ isinstance(module, nn.Linear)?
  ├─ _should_skip_lora(name, config)?  → 跳过白名单/视觉/apply_to 过滤
  ├─ setattr(parent, attr, LoRALinear(module, lora_config))
  └─ _freeze_for_lora(model, config)
       ├─ lora_A/B → trainable
       ├─ trainable_non_lora_modules → trainable
       ├─ vision_tower → depends on vision_encoder_mode
       └─ everything else → frozen
```

### 保存/加载/合并

训练时只保存 LoRA 权重（~100MB），不保存完整模型（~7GB），**不额外占用 GPU 内存**。

```
训练输出:
  checkpoints/exp/500/
  ├── lora.safetensors     # ~100MB, LoRA + 非 LoRA 可训练参数
  └── metadata.pt          # step 信息
```

```python
# 保存 (只保存可训练参数, 训练时自动调用)
save_lora_checkpoint(model, "checkpoints/500/lora.safetensors")

# 加载 (恢复训练)
load_lora_checkpoint(model, "checkpoints/500/lora.safetensors")

# 推理：用独立脚本合并 (不需要 GPU)
# python scripts/merge_lora.py \
#     --base /path/to/base/model.safetensors \
#     --lora checkpoints/500/lora.safetensors \
#     --config configs/waypoint_joint_libero.yaml \
#     --output checkpoints/500/model_merged.safetensors
```

**eval 自动检测**: eval 脚本优先加载 `model_merged.safetensors`，不存在则回退到 `model.safetensors`。

---

## 八、与 gradient_strategy 的交互

LoRA 模式下，`gradient_strategy` 仍然正常工作：

| gradient_strategy | backbone LoRA 收到的梯度来源 |
|---|---|
| `none` | VLM CE + AE MSE (通过 K/V) |
| `scale_gradient` (推荐) | VLM CE + `scale * AE_MSE` (通过 K/V) |
| `stop_gradient` | 仅 VLM CE |
| `freeze_backbone` | 无（backbone 完全冻结，LoRA 也不训练） |

注意：`freeze_backbone` + LoRA 会导致 backbone LoRA 参数也被冻结（因为 `joint_model.py` 中 `freeze_backbone` 会冻结所有 paligemma 参数）。如果想冻结 backbone base 权重但训练 backbone LoRA，应使用 `stop_gradient` 而非 `freeze_backbone`。

---

## 九、文件变更

| 文件 | 变更 |
|------|------|
| `src/openpi/models_pytorch/lora_pytorch.py` | 重写: PEFT 风格 LoRA |
| `scripts/train_waypoint_joint.py` | 更新 LoRA 初始化 + LoRA-only 保存 |
| `scripts/merge_lora.py` | 新增: 独立 LoRA merge 脚本 |
| `src/openpi/waypoint/eval_libero.py` | 更新: 自动检测 model_merged.safetensors |
| `configs/waypoint_joint_libero.yaml` | 更新: 白名单配置项 |
| `configs/waypoint_joint_calvin.yaml` | 更新: 白名单配置项 |
