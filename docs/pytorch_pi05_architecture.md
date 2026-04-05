# PyTorch Pi0.5 训练与推理流程详解

本文档详细记录了 openpi 项目中 PyTorch Pi0.5 模型的完整训练和推理流程，
包括模型架构、数据流、注意力机制、Mask 设计和 RoPE 位置编码。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [模型构建](#2-模型构建)
3. [训练流程](#3-训练流程)
4. [推理流程](#4-推理流程)
5. [LoRA 微调](#5-lora-微调)
6. [函数调用关系总览](#6-函数调用关系总览)

---

## 1. 整体架构概览

Pi0.5 是一个基于 **Flow Matching** 的机器人动作生成模型。核心思想是将
VLM (Vision-Language Model, PaliGemma) 与 Action Expert (Gemma 300M) 组合，
通过"去噪"过程从高斯噪声中生成机器人动作序列。

Pi0.5 相比 Pi0 有两个关键区别（定义在 `src/openpi/models/pi0_config.py`）：

> - state 输入从 suffix 的连续向量变为 prefix 的离散语言 token
> - action expert 使用 adaRMSNorm 注入 flow matching 的 timestep

整体模型架构：

```
┌─────────────────────────────────────────────────────┐
│                    PI0Pytorch                        │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  PaliGemmaWithExpertModel                    │    │
│  │  ┌───────────────┐  ┌─────────────────────┐ │    │
│  │  │ PaliGemma VLM  │  │  Gemma Action Expert│ │    │
│  │  │ (gemma_2b)     │  │  (gemma_300m)       │ │    │
│  │  │ + SigLIP ViT   │  │  + adaRMSNorm       │ │    │
│  │  └───────────────┘  └─────────────────────┘ │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  action_in_proj (Linear 32 → 1024)                  │
│  action_out_proj (Linear 1024 → 32)                 │
│  time_mlp_in / time_mlp_out  (Pi0.5专用)            │
└─────────────────────────────────────────────────────┘
```

---

## 2. 模型构建

### 2.1 配置定义 (`Pi0Config`)

文件：`src/openpi/models/pi0_config.py`

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None    # Pi0.5=200, Pi0=48
    pi05: bool = False
    discrete_state_input: bool = None  # Pi0.5=True
```

两个 Gemma 模型的参数对比（定义在 `src/openpi/models/gemma.py` 的 `get_config`）：

| 参数 | VLM (gemma_2b) | Action Expert (gemma_300m) |
|------|---------------|---------------------------|
| **width** (hidden_size) | 2048 | 1024 |
| **depth** (层数) | 18 | 18 |
| **num_heads** | 8 | 8 |
| **num_kv_heads** | 1 | 1 |
| **head_dim** | 256 | 256 |

关键点：两个模型的 depth、num_heads、num_kv_heads、head_dim 完全相同，
只有 width 不同。这使得逐层联合注意力成为可能——Q/K/V 投影后的维度只取决于
`num_heads × head_dim` 和 `num_kv_heads × head_dim`，与 width 无关。

### 2.2 模型初始化 (`PI0Pytorch.__init__`)

文件：`src/openpi/models_pytorch/pi0_pytorch.py`

```python
class PI0Pytorch(nn.Module):
    def __init__(self, config):
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config, action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
        )
        self.action_in_proj = nn.Linear(32, action_expert_config.width)   # 32 → 1024
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)  # 1024 → 32

        if self.pi05:
            self.time_mlp_in = nn.Linear(1024, 1024)
            self.time_mlp_out = nn.Linear(1024, 1024)
        else:
            self.state_proj = nn.Linear(32, 1024)
            self.action_time_mlp_in = nn.Linear(2048, 1024)
            self.action_time_mlp_out = nn.Linear(1024, 1024)
```

Pi0.5 vs Pi0 初始化差异：

- `use_adarms=[False, True]`：VLM 不用 adaRMS，Action Expert 用 adaRMS
- Pi0.5 用 `time_mlp_in/out` 将 timestep 编码为 adaRMS 的 conditioning signal
- Pi0 用 `state_proj` + `action_time_mlp_in/out` 将 state 和 timestep 拼接后嵌入

### 2.3 双模型架构 (`PaliGemmaWithExpertModel`)

文件：`src/openpi/models_pytorch/gemma_pytorch.py`

```python
class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, vlm_config, action_expert_config, use_adarms, precision):
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None  # Expert不需要自己的embedding层
```

包含两个子模型：
- **PaliGemma** (2B)：处理图像 + 语言 (prefix)
- **Gemma Expert** (300M)：处理动作 (suffix)，启用 adaRMSNorm

---

## 3. 训练流程

### 3.1 训练入口

文件：`scripts/train_pytorch.py`

```bash
# 单卡
python scripts/train_pytorch.py pi05_libero --exp_name my_run
# 多卡DDP
torchrun --nproc_per_node=2 scripts/train_pytorch.py pi05_libero --exp_name my_run
```

入口函数：

```python
def main():
    init_logging()
    config = _config.cli()    # 从命令行解析 TrainConfig
    train_loop(config)
```

### 3.2 训练配置

训练配置注册在 `src/openpi/training/config.py`，例如 `pi05_libero`：

```python
TrainConfig(
    name="pi05_libero",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(repo_id="physical-intelligence/libero", ...),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
)
```

### 3.3 数据流 (Data Pipeline)

```
LeRobot Dataset
    → TransformedDataset (应用 transforms)
        → RepackTransform      # 重组字典结构
        → DeltaActions          # 绝对→相对动作
        → PadStatesAndActions   # 填充到32维
        → Normalize             # z-score归一化
        → TokenizePrompt        # 语言 prompt → token
    → TorchDataLoader (collate & batch)
    → DataLoaderImpl.__iter__()
        → Observation.from_dict(batch), batch["actions"]
```

`Observation` 包含（定义在 `src/openpi/models/model.py`）：

```python
class Observation:
    images: dict[str, Float[ArrayT, "*b h w c"]]       # 3个视角图像
    image_masks: dict[str, Bool[ArrayT, "*b"]]          # 图像mask
    state: Float[ArrayT, "*b s"]                        # 机器人状态 (32维)
    tokenized_prompt: Int[ArrayT, "*b l"] | None        # tokenized语言指令
    tokenized_prompt_mask: Bool[ArrayT, "*b l"] | None  # prompt mask
```

### 3.4 训练循环 (`train_loop`)

关键步骤：

```python
# 1. 设置DDP
use_ddp, local_rank, device = setup_ddp()

# 2. 构建数据加载器
loader, data_config = build_datasets(config)

# 3. 构建模型
model = PI0Pytorch(model_cfg).to(device)
model.gradient_checkpointing_enable()

# 4. 可选：加载预训练权重
safetensors.torch.load_model(model, model_path)

# 5. 可选：LoRA微调
lora_utils.apply_lora_to_pi0_pytorch(model, config.lora_config)

# 6. DDP封装
model = DistributedDataParallel(model, ...)

# 7. 训练循环
while global_step < config.num_train_steps:
    for observation, actions in loader:
        losses = model(observation, actions)
        loss = losses.mean()
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=...)
        optim.step()
        optim.zero_grad(set_to_none=True)
```

### 3.5 训练前向传播 (`PI0Pytorch.forward`)

文件：`src/openpi/models_pytorch/pi0_pytorch.py`

```python
def forward(self, observation, actions, noise=None, time=None) -> Tensor:
    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

    noise = self.sample_noise(actions.shape, actions.device)         # [B, 50, 32]
    time = self.sample_time(actions.shape[0], actions.device)        # [B]

    time_expanded = time[:, None, None]                              # [B, 1, 1]
    x_t = time_expanded * noise + (1 - time_expanded) * actions      # 线性插值
    u_t = noise - actions                                             # 目标向量场

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

    # ... mask构建、联合前向 ...

    v_t = self.action_out_proj(suffix_out)
    return F.mse_loss(u_t, v_t, reduction="none")
```

Flow Matching 公式：
- 采样时间 `t ~ Beta(1.5, 1.0)`，缩放到 `[0.001, 1.0]`
- 噪声数据：`x_t = t * noise + (1-t) * actions`
- 目标向量场：`u_t = noise - actions`
- 模型预测向量场：`v_t`
- 损失：`MSE(u_t, v_t)`

---

## 4. 推理流程

### 4.1 创建推理 Policy

文件：`src/openpi/policies/policy_config.py`

```python
def create_trained_policy(train_config, checkpoint_dir, ...):
    is_pytorch = os.path.exists(os.path.join(checkpoint_dir, "model.safetensors"))
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
    return Policy(model, transforms=[...], output_transforms=[...], is_pytorch=is_pytorch)
```

### 4.2 推理入口 (`Policy.infer`)

文件：`src/openpi/policies/policy.py`

```python
def infer(self, obs, *, noise=None):
    inputs = self._input_transform(obs)        # 数据预处理 transforms
    inputs = torch.from_numpy(...).to(device)[None, ...]  # 转tensor加batch维
    observation = Observation.from_dict(inputs)
    actions = self._sample_actions(device, observation, **sample_kwargs)
    outputs = self._output_transform(outputs)  # 反归一化 etc.
    return outputs
```

### 4.3 动作采样 (`PI0Pytorch.sample_actions`)

Euler 去噪循环：

```python
@torch.no_grad()
def sample_actions(self, device, observation, noise=None, num_steps=10):
    # Step 1: 编码prefix (图像+语言)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)

    # Step 2: 预计算 prefix 的 KV cache（只需一次）
    _, past_key_values = self.paligemma_with_expert.forward(
        inputs_embeds=[prefix_embs, None], use_cache=True)

    # Step 3: Euler去噪循环 (从 t=1.0 到 t≈0)
    dt = -1.0 / num_steps
    x_t = noise              # 从纯噪声开始
    time = 1.0
    while time >= -dt / 2:
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
        x_t = x_t + dt * v_t  # Euler step
        time += dt
    return x_t
```

推理优化：prefix 的 KV cache 只需计算一次，之后 10 个去噪步骤只需要处理 suffix。

### 4.4 去噪步骤 (`denoise_step`)

```python
def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

    # 构建suffix对prefix的注意力mask
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    # 位置ID从prefix末尾继续
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

    # 只运行Expert部分 (带KV cache)
    outputs_embeds, _ = self.paligemma_with_expert.forward(
        inputs_embeds=[None, suffix_embs], past_key_values=past_key_values,
        adarms_cond=[None, adarms_cond])

    suffix_out = outputs_embeds[1][:, -self.config.action_horizon:]
    return self.action_out_proj(suffix_out)
```

---

## 5. LoRA 微调

文件：`src/openpi/models_pytorch/lora_pytorch.py`

在加载预训练权重后注入 LoRA 适配器：

```python
def apply_lora_to_pi0_pytorch(model, lora_config):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            lora_linear = apply_lora_to_linear(module, lora_cfg)
            setattr(parent, attr_name, lora_linear)
    frozen_count, trainable_count = freeze_for_lora_training(model, lora_config)
```

LoRA 可配置只应用在 paligemma、expert 或两者的 attention/FFN 层上，
同时保持 `action_in_proj`、`action_out_proj`、`time_mlp` 等非 LoRA 层可训练。

---

## 6. 函数调用关系总览

### 训练调用链

```
main() → train_loop(config)
  → build_datasets(config)
    → create_data_loader(config, framework="pytorch")
      → create_torch_dataset() → transform_dataset() → TorchDataLoader → DataLoaderImpl
  → PI0Pytorch(model_cfg)
    → PaliGemmaWithExpertModel(vlm_config, expert_config, use_adarms=[False,True])
  → model.forward(observation, actions)
    → _preprocess_observation() → preprocess_observation_pytorch()
    → sample_noise(), sample_time()
    → embed_prefix(images, img_masks, lang_tokens, lang_masks)
      → paligemma_with_expert.embed_image() (SigLIP)
      → paligemma_with_expert.embed_language_tokens() (* sqrt(dim))
    → embed_suffix(state, x_t, time)
      → create_sinusoidal_pos_embedding() → time_mlp → adarms_cond
      → action_in_proj(noisy_actions)
    → make_att_2d_masks()
    → paligemma_with_expert.forward([prefix_embs, suffix_embs], adarms_cond=[None, time_cond])
      → compute_layer_complete() × 18层
        → input_layernorm(hidden_states, cond=adarms_cond) → GemmaRMSNorm
        → q_proj, k_proj, v_proj → cat → RoPE → eager_attention_forward → split → o_proj
        → _gated_residual() → post_attention_layernorm → MLP → _gated_residual()
    → action_out_proj(suffix_out)
    → MSE(u_t, v_t)
```

### 推理调用链

```
create_trained_policy(config, checkpoint_dir)
  → Pi0Config.load_pytorch() → PI0Pytorch + safetensors.load_model
  → Policy(model, transforms=[...], output_transforms=[...], is_pytorch=True)

Policy.infer(obs)
  → input_transform(obs)
  → Observation.from_dict(inputs)
  → sample_actions(device, observation, num_steps=10)
    → _preprocess_observation(train=False)
    → embed_prefix() → [img_emb, lang_emb]
    → paligemma_with_expert.forward([prefix_embs, None], use_cache=True) → KV cache
    → loop 10次:
        denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
          → embed_suffix(state, x_t, time) → suffix_embs, adarms_cond
          → paligemma_with_expert.forward([None, suffix_embs], past_key_values, adarms_cond)
          → action_out_proj → v_t
        x_t = x_t + dt * v_t  (Euler step)
  → output_transform(outputs) → Unnormalize → AbsoluteActions
```
