# Pi0.5 联合注意力机制与 Mask 深度解析

本文档详细解释 Pi0.5 中双模型联合注意力的实现细节，包括 Q/K/V 维度变换、
Mask 设计原理、RoPE 位置编码，以及 `embed_prefix` 和 `embed_suffix` 的
完整数据流。

---

## 目录

1. [embed_prefix 详解](#1-embed_prefix-详解)
2. [embed_suffix 详解](#2-embed_suffix-详解)
3. [训练 Forward 中的 Mask 构建](#3-训练-forward-中的-mask-构建)
4. [Q/K/V 的 Concat 维度详解](#4-qkv-的-concat-维度详解)
5. [联合注意力计算详解](#5-联合注意力计算详解)
6. [三种 Mask 的原理](#6-三种-mask-的原理)
7. [position_ids 与 RoPE 详解](#7-position_ids-与-rope-详解)
8. [att_2d_masks_4d 的宏观与微观作用](#8-att_2d_masks_4d-的宏观与微观作用)

---

## 1. embed_prefix 详解

文件：`src/openpi/models_pytorch/pi0_pytorch.py`，`embed_prefix` 方法

### 输入维度

输入来自 `_preprocess_observation` 的拆包：

| 变量 | 来源 | 维度 | 说明 |
|------|------|------|------|
| `images` | `list(observation.images.values())` | 长度为 3 的 list，每个 `[B, 3, 224, 224]` | 3 个摄像头图像 (base, left_wrist, right_wrist) |
| `img_masks` | `list(observation.image_masks.values())` | 长度为 3 的 list，每个 `[B]` bool | 该摄像头是否存在/有效 |
| `lang_tokens` | `observation.tokenized_prompt` | `[B, 200]` int32 | tokenized 后的语言指令 |
| `lang_masks` | `observation.tokenized_prompt_mask` | `[B, 200]` bool | 哪些位置是实际 token，哪些是 padding |

### 步骤 1：图像嵌入（循环 3 次）

```python
for img, img_mask in zip(images, img_masks):
    img_emb = self.paligemma_with_expert.embed_image(img)
    # 调用链: embed_image → paligemma.model.get_image_features(img)
    # SigLIP: 224×224, patch_size=14 → 16×16 = 256 patches
    # 经 multi-modal projector → [B, 256, 2048]

    embs.append(img_emb)                                             # [B, 256, 2048]
    pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))  # [B, 256] bool
    att_masks += [0] * num_img_embs                                  # 追加 256 个 0
```

每张图像处理后：

| 变量 | 维度 | 说明 |
|------|------|------|
| `img_emb` | `[B, 256, 2048]` | 256 个视觉 token，维度=VLM width |
| `num_img_embs` | `256` | |
| `img_mask[:, None].expand(B, 256)` | `[B, 256]` bool | 摄像头有效→全True；无效→全False |
| `att_masks += [0]*256` | | |

循环 3 次后累积：embs = 3 × `[B, 256, 2048]`，att_masks = `[0] × 768`

### 步骤 2：语言嵌入

```python
lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
# 调用链: embed_language_tokens → paligemma.language_model.embed_tokens(tokens)
# Embedding: vocab_size=257152, hidden_size=2048

lang_emb = lang_emb * math.sqrt(2048)    # 标准缩放: [B, 200, 2048]

embs.append(lang_emb)                     # [B, 200, 2048]
pad_masks.append(lang_masks)              # [B, 200] bool
att_masks += [0] * 200                    # 追加 200 个 0
```

### 步骤 3：拼接输出

```python
embs = torch.cat(embs, dim=1)             # [B, 968, 2048]
pad_masks = torch.cat(pad_masks, dim=1)   # [B, 968] bool
att_masks = torch.tensor(att_masks)        # [968] → expand → [B, 968]
```

最终输出：

| 返回值 | 维度 | 内容 |
|--------|------|------|
| `prefix_embs` | `[B, 968, 2048]` | `[img0(256) | img1(256) | img2(256) | lang(200)]` |
| `prefix_pad_masks` | `[B, 968]` bool | 每个位置是否有效 |
| `prefix_att_masks` | `[B, 968]` | **全 0**——所有 prefix token 完全双向互看 |

---

## 2. embed_suffix 详解

文件：`src/openpi/models_pytorch/pi0_pytorch.py`，`embed_suffix` 方法

### 输入维度

| 变量 | 维度 | 说明 |
|------|------|------|
| `state` | `[B, 32]` float32 | 机器人当前关节状态 |
| `noisy_actions` (x_t) | `[B, 50, 32]` float32 | flow matching 插值后的噪声动作 |
| `timestep` (time) | `[B]` float32 | 采样的时间步 |

### Pi0.5 分支 (`self.pi05 = True`)

`if not self.pi05` 被跳过，**state 不进入 suffix**。

**时间步嵌入：**

```python
time_emb = create_sinusoidal_pos_embedding(timestep, 1024, ...)
# timestep: [B] → sin/cos编码 → [B, 1024]
```

**动作嵌入：**

```python
action_emb = self.action_in_proj(noisy_actions)
# Linear(32 → 1024): [B, 50, 32] → [B, 50, 1024]
```

**Pi0.5 的 time MLP：**

```python
time_emb = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
# [B, 1024] → Linear(1024,1024) → SiLU → Linear(1024,1024) → SiLU → [B, 1024]

action_time_emb = action_emb     # [B, 50, 1024]  — 只有 action 嵌入
adarms_cond = time_emb           # [B, 1024]       — timestep 通过 adaRMS 注入
```

**Mask 与拼接：**

```python
embs.append(action_time_emb)                                        # [B, 50, 1024]
pad_masks.append(torch.ones(B, 50, bool))                           # [B, 50] 全True
att_masks += [1] + ([0] * (self.config.action_horizon - 1))         # [1, 0, 0, ..., 0]
```

**最终输出：**

| 返回值 | 维度 | 内容 |
|--------|------|------|
| `suffix_embs` | `[B, 50, 1024]` | 50 个动作 token |
| `suffix_pad_masks` | `[B, 50]` bool | 全 True |
| `suffix_att_masks` | `[B, 50]` | `[1, 0, 0, ..., 0]` |
| `adarms_cond` | `[B, 1024]` | timestep 编码 |

### Pi0 分支对比 (`self.pi05 = False`)

Pi0 多一个 state token 和不同的 timestep 注入方式：

```python
# state 嵌入为连续向量
state_emb = self.state_proj(state)     # Linear(32→1024): [B,32]→[B,1024]
embs.append(state_emb[:, None, :])     # [B, 1, 1024]
att_masks += [1]                       # state 单独一个 mask=1

# timestep 与 action 拼接
time_emb = time_emb[:, None, :].expand_as(action_emb)     # [B,50,1024]
action_time_emb = torch.cat([action_emb, time_emb], dim=2) # [B,50,2048]
action_time_emb = mlp(action_time_emb)                      # [B,50,1024]
adarms_cond = None
```

| | Pi0 | Pi0.5 |
|-|-----|-------|
| suffix 内容 | `[state(1) | actions(50)]` | `[actions(50)]` |
| suffix_embs | `[B, 51, 1024]` | `[B, 50, 1024]` |
| suffix_att_masks | `[1, 1, 0, ..., 0]` (长51) | `[1, 0, ..., 0]` (长50) |
| timestep 注入 | 与 action 拼接后过 MLP | 通过 adaRMS conditioning |
| adarms_cond | `None` | `[B, 1024]` |

---

## 3. 训练 Forward 中的 Mask 构建

文件：`src/openpi/models_pytorch/pi0_pytorch.py`，`forward` 方法

以 Pi0.5 为例，总序列长度 `L = 968 + 50 = 1018`。

```python
pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
# [B, 968] cat [B, 50] → [B, 1018] bool

att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
# [B, 968] cat [B, 50] → [B, 1018]
# 值: [0,0,...,0, 1,0,...,0]  (968个0 + 50个)

att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
# [B, 1018, 1018] bool

position_ids = torch.cumsum(pad_masks, dim=1) - 1
# [B, 1018] int

att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
# [B, 1, 1018, 1018] float (0.0 或 -2.38e38)
```

### pad_masks — 标记哪些位置是真实 token

具体例子（3 个摄像头有效，prompt 15 个 token）：

```
位置:    0─────────767  768────782  783──967  968──1017
内容:    img0..img2      lang实际     lang_pad   actions
pad:     T T T ... T     T T ... T   F F ... F  T T ... T
```

作用：在后续的 2D mask 中，padding 位置不会参与任何注意力计算。

### att_masks — 定义注意力组的边界

Pi0.5 的值：

```
位置:    0──────────────967   968   969──1017
内容:    所有prefix token      act₀   act₁..act₄₉
值:      0  0  0  ...  0      1      0  0  ...  0
```

`1` 的含义：在这个位置切一刀，之前的 token 不能看到这个位置及之后。

### 传给 `paligemma_with_expert.forward` 的参数

| 参数 | 维度 | 说明 |
|------|------|------|
| `attention_mask` | `[B, 1, 1018, 1018]` float | 统一的 2D 注意力 mask |
| `position_ids` | `[B, 1018]` int | 统一的位置 ID |
| `inputs_embeds` | `[[B,968,2048], [B,50,1024]]` | 两个不同维度的 embedding |
| `adarms_cond` | `[None, [B,1024]]` | VLM 无 conditioning，Expert 接收 timestep |

---

## 4. Q/K/V 的 Concat 维度详解

文件：`src/openpi/models_pytorch/gemma_pytorch.py`，`compute_layer_complete` 函数

### Concat 前

循环 `i=0`（VLM），`hidden_states: [B, P, 2048]`：

```
q_proj: Linear(2048 → 8×256=2048)
  [B, P, 2048] → view [B, P, 8, 256] → transpose → [B, 8, P, 256]

k_proj: Linear(2048 → 1×256=256)
  [B, P, 2048] → view [B, P, 1, 256] → transpose → [B, 1, P, 256]

v_proj: Linear(2048 → 1×256=256)
  [B, P, 2048] → view [B, P, 1, 256] → transpose → [B, 1, P, 256]
```

循环 `i=1`（Expert），`hidden_states: [B, S, 1024]`：

```
q_proj: Linear(1024 → 8×256=2048)
  [B, S, 1024] → view [B, S, 8, 256] → transpose → [B, 8, S, 256]

k_proj: Linear(1024 → 1×256=256)
  [B, S, 1024] → view [B, S, 1, 256] → transpose → [B, 1, S, 256]

v_proj: Linear(1024 → 1×256=256)
  [B, S, 1024] → view [B, S, 1, 256] → transpose → [B, 1, S, 256]
```

核心观察：尽管两个模型的 width 不同（2048 vs 1024），Q/K/V 投影后
head 维度完全一致（`[B, 8, *, 256]` 和 `[B, 1, *, 256]`），只有序列长度不同。

### Concat 操作（沿 dim=2 序列维度）

```python
query_states = torch.cat(query_states, dim=2)
# cat([B,8,P,256], [B,8,S,256]) → [B, 8, P+S, 256]

key_states = torch.cat(key_states, dim=2)
# cat([B,1,P,256], [B,1,S,256]) → [B, 1, P+S, 256]

value_states = torch.cat(value_states, dim=2)
# cat([B,1,P,256], [B,1,S,256]) → [B, 1, P+S, 256]
```

拼接后，位置 `0..P-1` 对应 VLM 的 prefix token，
位置 `P..P+S-1` 对应 Expert 的 suffix token。

---

## 5. 联合注意力计算详解

文件：`src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`

### GQA 展开

因为 `num_kv_heads=1, num_heads=8`，K/V 需要 repeat 到 8 个 head：

```python
key_states = repeat_kv(key, num_key_value_groups=8)
# [B, 1, P+S, 256] → [B, 8, P+S, 256]
value_states = repeat_kv(value, 8)
# [B, 1, P+S, 256] → [B, 8, P+S, 256]
```

### 注意力分数

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
# [B, 8, P+S, 256] @ [B, 8, 256, P+S] = [B, 8, P+S, P+S]
```

这是一个 (P+S) × (P+S) 的全序列注意力矩阵。

### Mask 应用

```python
attn_weights = attn_weights + causal_mask
# causal_mask: [B, 1, P+S, P+S]  (0.0 允许, -2.38e38 屏蔽)
# 在 head 维度广播
```

### 输出拆分与 o_proj

```python
att_output = att_output.reshape(batch_size, -1, 8 * 256)  # [B, P+S, 2048]

# 拆回两段
VLM部分:   att_output[:, 0:P, :]    → VLM o_proj Linear(2048→2048)  → [B, P, 2048]
Expert部分: att_output[:, P:P+S, :] → Expert o_proj Linear(2048→1024) → [B, S, 1024]
```

各自回到自己的 width 空间后独立做残差连接和 MLP。

### 完整数据流图（每一层）

```
VLM hidden:  [B, P, 2048]              Expert hidden: [B, S, 1024]
       │                                       │
   LayerNorm (standard)                    LayerNorm (adaRMS, cond=time_emb)
       │                                       │
   Q₀ [B,8,P,256]                         Q₁ [B,8,S,256]
   K₀ [B,1,P,256]                         K₁ [B,1,S,256]
   V₀ [B,1,P,256]                         V₁ [B,1,S,256]
       │                                       │
       └──────────── cat(dim=2) ───────────────┘
                         │
              Q: [B, 8, P+S, 256]
              K: [B, 1, P+S, 256] → repeat_kv → [B, 8, P+S, 256]
              V: [B, 1, P+S, 256] → repeat_kv → [B, 8, P+S, 256]
                         │
                    RoPE 旋转
                         │
              ┌─── Attention(Q, K, V, mask) ───┐
              │    [B, 8, P+S, P+S] weights    │
              │    (mask 阻止 prefix→suffix)     │
              │    → [B, P+S, 2048] output      │
              └────────────────────────────────┘
                         │
              ┌──── split(dim=1) ─────────┐
              │                            │
     [B, P, 2048]                  [B, S, 2048]
         │                              │
   VLM o_proj                    Expert o_proj
   Linear(2048→2048)             Linear(2048→1024)
         │                              │
     [B, P, 2048]                  [B, S, 1024]
         │                              │
   + residual                    + gated_residual (gate from adaRMS)
   + MLP (16384)                 + MLP (4096)
   + residual                    + gated_residual
         │                              │
   → Next Layer                  → Next Layer
```

---

## 6. 三种 Mask 的原理

### 6.1 pad_masks — 为什么 AND 能隔离 padding？

```python
pad_2d_masks[b, i, j] = pad_masks[b, i] AND pad_masks[b, j]
```

#### 背景

Attention 矩阵 `A[i, j]` 表示 token i（query）对 token j（key/value）的注意力权重：

- **第 i 行**：token i 从其他所有 token 收集多少信息
- **第 j 列**：token j 被多少 token "看到"

#### AND 的效果

假设 token 2 是 padding（`pad_masks[b, 2] = False`）：

**方向 1：padding 不能"看"别人（第 i=2 行全灭）**

```
pad_2d_masks[b, 2, j] = False AND (任何值) = False  对所有 j
```

整个第 2 行全 False → token 2 的 attention 权重全部被屏蔽。

**方向 2：别人不能"看" padding（第 j=2 列全灭）**

```
pad_2d_masks[b, i, 2] = (任何值) AND False = False  对所有 i
```

整个第 2 列全 False → 没有任何 token 能注意到 token 2。

```
             j=0(有效)  j=1(有效)  j=2(pad)  j=3(有效)
       ┌──────────────────────────────────────────────┐
 i=0   │    T          T          F          T        │
 i=1   │    T          T          F          T        │
 i=2   │    F          F          F          F        │  ← 全行F
(pad)  │                                              │
 i=3   │    T          T          F          T        │
       └──────────────────────────────────────────────┘
                                  ↑ 全列F
```

AND 的布尔逻辑天然满足这两个需求：一个 False 就能毒化整行或整列。
如果只用 `pad_masks[b, j]`（只屏蔽列），padding token 作为 query
还会从有效 token 收集信息并产生输出，污染后续计算。

### 6.2 att_masks — 为什么用 cumsum？

#### 为什么不能直接用 True/False？

如果只用 True/False 标记每个位置，只能表达"这个位置能不能参与注意力"
——这是一维的，无法区分"谁看谁"的方向性。

核心需求：**suffix 能看 prefix，但 prefix 不能看 suffix**。
一维的 True/False 无法编码这种非对称性。

#### cumsum 把边界标记转化为组号

`att_masks` 中的 `1` 不是"屏蔽这个位置"，而是"在这里切一刀，开始新组"。

```
att_masks: [0, 0, ..., 0,  1, 0, 0, ..., 0]
                            ↑ 切一刀
cumsum:    [0, 0, ..., 0,  1, 1, 1, ..., 1]
            组0(prefix)     组1(suffix)
```

规则 `cumsum[j] <= cumsum[i]` 表达：**i 只能注意到组号 ≤ 自己的 token**。

- 组 0 的 token：只能看组 0（`0≤0`），不能看组 1（`1>0`）
- 组 1 的 token：能看组 0（`0≤1`）和组 1（`1≤1`）→ 看所有

更复杂的例子：

```
全部 att_masks = [1,1,1,1,1,1]
cumsum = [1,2,3,4,5,6]
→ 每个位置只能看 ≤ 自己组号的位置
→ 纯因果（自回归）注意力

prefix-LM: att_masks = [0,0,0,1,1,1]
cumsum = [0,0,0,1,2,3]
→ 前3个双向互看，后3个因果注意力
```

#### Pi0.5 的 2D Mask 展开

```
                       Key (j)
                    prefix(cs=0)    suffix(cs=1)
              j:  0  1  ...  967   968  969 ... 1017
         ┌──────────────────────────────────────────┐
 prefix  │  0≤0=T  0≤0=T  ...  T │ 1≤0=F ... F    │  Prefix只能看Prefix
 (cs=0)  │   ✅     ✅     ...  ✅ │  ❌   ...  ❌   │
         ├────────────────────────┼─────────────────┤
 suffix  │  0≤1=T  0≤1=T  ...  T │ 1≤1=T ... T    │  Suffix能看所有
 (cs=1)  │   ✅     ✅     ...  ✅ │  ✅   ...  ✅   │
         └────────────────────────┴─────────────────┘
```

Pi0 的 2D Mask（3 个组）：

```
                  prefix(cs=0)   state(cs=1)  actions(cs=2)
       ┌──────────────────────────────────────────────────┐
prefix │   ✅ 互看            │    ❌        │    ❌        │
(cs=0) │                      │              │             │
       ├──────────────────────┼──────────────┼─────────────┤
state  │   ✅ 看prefix        │    ✅        │    ❌        │
(cs=1) │                      │              │             │
       ├──────────────────────┼──────────────┼─────────────┤
actions│   ✅ 看所有           │    ✅        │    ✅ 互看   │
(cs=2) │                      │              │             │
       └──────────────────────┴──────────────┴─────────────┘
```

### 6.3 三层 Mask 的关系总结

```
① att_masks (1D, [B, 1018])
   → 定义"注意力组边界"
   → [0,...,0, 1, 0,...,0]
          cumsum ↓
② att_2d_masks (2D, [B, 1018, 1018])
   → cumsum[j] ≤ cumsum[i] 决定 i 能否看到 j
          AND ↓
③ pad_masks (1D, [B, 1018])
   → 外积 → pad_2d_masks: padding位置被彻底隔离
          ↓
最终: att_2d_masks = (注意力组约束) & (padding约束)
      → 4D化 + 值转换: True→0.0, False→-2.38e38
      → 送入 Transformer 的 attention 计算
```

| Mask | 维度 | 什么决定它 | 功能 |
|------|------|-----------|------|
| `pad_masks` | `[B, 1018]` bool | 图像是否存在、语言是否是实际 token | 隔离 padding |
| `att_masks` | `[B, 1018]` | 手动设定的组边界 | 定义注意力方向性 |
| `att_2d_masks` | `[B, 1018, 1018]` bool | `cumsum` + `pad_masks` 外积 | token-to-token 可见性矩阵 |
| `att_2d_masks_4d` | `[B, 1, 1018, 1018]` float | `att_2d_masks` 转换 | 直接加到 attention score |

---

## 7. position_ids 与 RoPE 详解

### 7.1 为什么需要位置编码

Transformer 的 attention 本身不区分顺序。如果把序列 `[A, B, C]` 打乱成
`[C, A, B]`，不加位置编码的 attention 计算结果完全不变——因为 `Q·K^T`
只看向量内容，不看位置。必须在向量里"注入"位置信息。

### 7.2 RoPE 的核心思想：用旋转编码相对距离

RoPE (Rotary Position Embedding) 的规则：

> 位置为 p 的 token，其 Q 和 K 向量被旋转角度 `p × θ`（θ 是预设频率）。

对于 head_dim=256，有 128 个频率对：

```
inv_freq = [θ₀, θ₁, θ₂, ..., θ₁₂₇]

其中 θ_k = 1 / (10000^(2k/256))

θ₀ = 1.0          (高频，对近距离敏感)
θ₁ = 0.83...
...
θ₁₂₇ ≈ 0.00001   (低频，对远距离敏感)
```

当 token A 在位置 3、token B 在位置 7 时：
- A 的 Q 被旋转 `3θ`
- B 的 K 被旋转 `7θ`

计算 attention score `Q_A · K_B` 时，数学上等价于
**未旋转的 Q·K，然后旋转 `(3-7)θ = -4θ`**。

**关键性质：Q·K 的结果只取决于位置差 `i - j`（相对距离），不取决于绝对位置。**

### 7.3 position_ids 的作用

position_ids 给每个 token 一个**位置编号**。RoPE 拿到编号后计算旋转角度。

代码中 RoPE 如何使用 position_ids：

```python
# GemmaRotaryEmbedding.forward
inv_freq_expanded = self.inv_freq[None, :, None]    # [1, D/2, 1]  频率基底
position_ids_expanded = position_ids[:, None, :]     # [B, 1, L]    位置编号

freqs = inv_freq_expanded @ position_ids_expanded    # [B, D/2, L]  矩阵乘法
# 含义: freqs[b, k, p] = position_ids[b, p] × θ_k   即 "位置p在频率k上的旋转角度"

freqs = freqs.transpose(1, 2)                        # [B, L, D/2]
emb = torch.cat((freqs, freqs), dim=-1)              # [B, L, D]
cos = emb.cos()                                      # [B, L, D]
sin = emb.sin()                                      # [B, L, D]
```

核心：**position_ids 中的数字直接乘以频率，得到旋转角度。**

然后旋转 Q 和 K：

```python
q_embed = (q * cos) + (rotate_half(q) * sin)   # 旋转Q
k_embed = (k * cos) + (rotate_half(k) * sin)   # 旋转K
```

### 7.4 为什么 position_ids 用 `cumsum(pad_masks) - 1`

#### 错误做法：所有位置顺序编号

```
位置:       img(768)    lang(3个有效)  padding(197个)  action(50个)
pad_mask:   T ... T     T  T  T       F  F  ... F     T  ...  T

position_ids = [0, 1, ..., 767, 768, 769, 770, 771, ..., 967, 968, ..., 1017]
```

- `lang₂`（最后一个有效语言 token）position_id = 770
- `act₀`（第一个动作 token）position_id = 968
- RoPE 认为相对距离 = 968 - 770 = **198**

但中间 197 个 padding 是不存在的！语义上 `act₀` 应紧跟 `lang₂`。

#### 正确做法：`cumsum(pad_masks) - 1`

```
位置:       img(768)    lang(3个有效)  padding(197个)       action(50个)
pad_mask:   T ... T     T  T  T       F   F   ...  F       T   ...  T
cumsum:     1 ... 768  769 770 771   771 771  ... 771     772  ...  821
cumsum-1:   0 ... 767  768 769 770   770 770  ... 770     771  ...  820
```

- `lang₂` position_id = 770
- `act₀` position_id = 771
- RoPE 认为相对距离 = 771 - 770 = **1** ← 正确，紧邻！

padding 位置的 position_id 停滞不变。它们在 `pad_2d_masks` 中已被完全屏蔽，
所以 RoPE 给它们编的号不会参与任何实际的 attention 计算。

### 7.5 联合注意力中的 RoPE 调用

```python
# gemma_pytorch.py, compute_layer_complete 函数
dummy_tensor = torch.zeros(B, P+S, 256, ...)   # 只用于获取device/dtype
cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

此时：
- `query_states: [B, 8, 1018, 256]` — VLM(968) + Expert(50) 拼接后
- `position_ids: [B, 1018]` — VLM 和 Expert 共享同一套位置 ID

```
position_ids = [0, 1, ..., 767,  768, 769, 770,  770, ..., 770,  771, ..., 820]
                ↑ img tokens ↑   ↑ lang有效 ↑    ↑ lang padding ↑  ↑ action ↑
                (VLM的Q/K)                                          (Expert的Q/K)
```

**两个模型通过共享 position_ids + RoPE，在同一个位置坐标系中运作。**
Expert 的动作 token 能正确感知与图像/语言 token 之间的相对距离。

### 7.6 推理时的 position_ids

推理时用了 KV cache，`denoise_step` 只给 suffix 的 position_ids：

```python
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]  # prefix有效token总数
position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
# 从 prefix 末尾偏移量开始递增: [771, 772, ..., 820]
```

保证 suffix token 的位置编号与训练时一致。

### 7.7 完整因果链总结

```
pad_masks: [T, T, ..., F, F, ..., T, T]
    │
    │ cumsum - 1
    ▼
position_ids: [0, 1, ..., 770, 770, ..., 771, 772]
    │                    ↑ padding不递增
    │
    │ rotary_emb: freqs = position_ids × inv_freq
    ▼
旋转角度: 每个位置p、每个频率k → angle = p × θ_k
    │
    │ cos(angle), sin(angle)
    ▼
cos, sin: [B, 1018, 256]
    │
    │ apply_rotary_pos_emb: Q_rotated = Q*cos + rotate_half(Q)*sin
    ▼
旋转后的 Q, K
    │
    │ Q_i · K_j → 只取决于 position_ids[i] - position_ids[j] (相对距离)
    ▼
attention score 自然编码了相对位置信息
```

一句话总结：`position_ids` 中的每个整数是 RoPE 的输入，决定该位置的旋转角度；
两个位置的 attention score 取决于它们 position_id 之差（相对距离）。
`cumsum(pad_masks) - 1` 确保 padding 不占用位置编号，
有效 token 之间的相对距离不被 padding 撑开。

---

## 8. att_2d_masks_4d 的宏观与微观作用

### 宏观：整个模型的"信息流图"

`att_2d_masks_4d` 是一张精心设计的**通信规则表**，从根本上决定了 Pi0.5 模型的
架构特性——哪些模块之间有信息交换，哪些被隔离。

```
              ┌───────────────────┐
              │     VLM (Prefix)   │
              │  图像 ↔ 语言       │  ← 双向全连接，建立多模态理解
              │  (自己的小世界)     │
              └───────┬───────────┘
                      │
                      │ 信息单向流动 ↓ (suffix→prefix ✅, prefix→suffix ❌)
                      │
              ┌───────▼───────────┐
              │  Expert (Suffix)   │
              │  动作 ↔ 图像/语言  │  ← 动作token读取VLM的全部理解
              │  动作 ↔ 动作       │  ← 动作token之间双向协调
              └───────────────────┘
```

设计意图：
- VLM 是"感知模块"——处理视觉和语言时不受动作噪声干扰
- Expert 是"决策模块"——读取 VLM 的全部理解来生成动作

完整注意力表：

| 关系 | 能否 Attend | 说明 |
|------|:-----------:|------|
| 图像 ↔ 图像 | ✅ | 3 个视角 768 token 完全双向 |
| 图像 ↔ 语言 | ✅ | 图文互看，理解任务指令 |
| 图像/语言 → 动作 | ❌ | VLM 完全看不到动作 token |
| 动作 → 图像/语言 | ✅ | 动作生成依赖视觉和语言 |
| 动作 ↔ 动作 | ✅ | 50 个动作 token 双向协调 |

### 微观：在 Attention 计算中的精确作用

`att_2d_masks_4d` 的值：`0.0`（允许）和 `-2.3819763e38`（屏蔽）。

在 `eager_attention_forward` 中作为**加法偏置**：

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
# [B, 8, P+S, P+S] 原始注意力分数

attn_weights = attn_weights + causal_mask
# causal_mask = att_2d_masks_4d: [B, 1, P+S, P+S]

attn_weights = softmax(attn_weights, dim=-1)
# 被屏蔽位置: e^(-2.38e38) ≈ 0 → softmax后权重趋近于0
```

为什么不用乘法 0/1：因为 softmax 是对整行归一化的。
用加法+极大负数的方式，softmax 自然把被屏蔽位置的权重压到 0，且梯度也正确为 0。

### 维度流转

```
att_2d_masks: [B, 1018, 1018] bool
     │
     │ [:, None, :, :] 插入 head 维度
     ▼
att_2d_masks_4d: [B, 1, 1018, 1018] float
     │                 ↑ 广播到 8 个 head
     ▼
attn_weights: [B, 8, 1018, 1018]  原始分数
     + causal_mask: [B, 1, 1018, 1018]  (自动广播)
     = masked_weights: [B, 8, 1018, 1018]
     │
     │ softmax(dim=-1)
     ▼
attn_probs: [B, 8, 1018, 1018]  注意力概率（被屏蔽位置≈0）
     │
     │ @ value_states
     ▼
attn_output: [B, 8, 1018, 256]
```

Head 维度是 1（所有 head 共享 mask）：因为"prefix 不能看 suffix"这个约束
对所有 head 都成立。不同 head 学到不同的注意力模式，但必须遵守同样的可见性约束。

---

## 附录：adaRMSNorm 机制

文件：`src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`

Pi0.5 的核心创新。在 Expert 每层的 LayerNorm 中：

```python
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, cond_dim=None):
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3)  # 产出 scale, shift, gate

    def forward(self, x, cond=None):
        normed_inputs = self._norm(x)   # 标准RMS归一化

        if cond is None:
            return normed_inputs * (1 + self.weight), None

        # adaptive RMSNorm
        modulation = self.dense(cond)                    # [B, dim*3]
        scale, shift, gate = torch.chunk(modulation, 3)  # 各 [B, dim]
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs, gate
```

time_emb 通过 `dense` 层生成三元组：
- **scale** 和 **shift**：条件化归一化 `output = normalized * (1 + scale) + shift`
- **gate**：残差连接的调制 `residual + output * gate`（通过 `_gated_residual` 函数）
