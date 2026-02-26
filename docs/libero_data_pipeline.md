# LIBERO 数据集字段转换记录

记录从原始 LIBERO HDF5 到最终 Waypoint RLDS 的完整字段映射关系，共分四个阶段。

---

## 阶段概览

```
原始 LIBERO HDF5 (128x128, 含 no-ops)
        │
        ▼  regenerate_libero_dataset.py  (OpenVLA)
        │  - 在仿真环境中重放演示，升分辨率至 256x256
        │  - 过滤 no-op 动作，仅保留成功轨迹
        │
重生成 HDF5 (*_no_noops, 256x256)
        │
        ▼  rlds_dataset_builder/LIBERO_Object_dataset_builder.py  (moojink/OpenVLA)
        │  - HDF5 → RLDS TFRecord
        │  - 图像做 [::-1,::-1] 旋转 180°（修正环境渲染上下颠倒问题）
        │  - state = ee_states(6D) + gripper_states(2D) = 8D
        │
OpenVLA RLDS (256x256, 标准 RLDS 字段)
        │
        ▼  rlds_wp_extract.py  (AWE waypoint 扩展)
        │  - 图像 resize 到 224x224
        │  - 添加 waypoint_duration, is_waypoint_end, original_step_index
        │
Waypoint RLDS (224x224, 含 waypoint 标注)
  ← 对应 libero_openvla_rlds_features.json
```

---

## 阶段 1 → 阶段 2：原始 HDF5 → 重生成 HDF5

`regenerate_libero_dataset.py` 只读取原始 HDF5 中的 `actions` 和 `states[0]`，
其余所有观测都是通过**在仿真环境中重放**重新采集的，原始 HDF5 的图像数据完全未被使用。

| 原始 HDF5 字段 | 用途 | 重生成 HDF5 字段 | 来源 |
|---|---|---|---|
| `data/demo_{i}/actions` (T,7) | 重放动作（去 no-op） | `data/demo_{i}/actions` (T,7) | 原始动作，过滤 no-op 后保留 |
| `data/demo_{i}/states` (T,...) | 仅取 `states[0]` 初始化环境 | `data/demo_{i}/states` (T,...) | 首步复制自原始，后续从 `env.sim.get_state()` |
| _(环境重放)_ | `obs["agentview_image"]` | `obs/agentview_rgb` (T,256,256,3) | 环境渲染，256x256 |
| _(环境重放)_ | `obs["robot0_eye_in_hand_image"]` | `obs/eye_in_hand_rgb` (T,256,256,3) | 环境渲染，256x256 |
| _(环境重放)_ | `obs["robot0_eef_pos"]` (3) + `quat2axisangle(obs["robot0_eef_quat"])` (3) | `obs/ee_states` (T,6) | pos(3) + axisangle(3) 拼接 |
| _(同上拆分)_ | `ee_states[:, :3]` | `obs/ee_pos` (T,3) | 仅位置 |
| _(同上拆分)_ | `ee_states[:, 3:]` | `obs/ee_ori` (T,3) | 仅朝向 |
| _(环境重放)_ | `obs["robot0_gripper_qpos"]` | `obs/gripper_states` (T,2) | 夹爪关节位置 |
| _(环境重放)_ | `obs["robot0_joint_pos"]` | `obs/joint_states` (T,7) | 7 关节角度 |
| _(环境重放)_ | `gripper_qpos + eef_pos + eef_quat` 拼接 | `robot_states` (T,7) | 复合机器人状态 |
| _(计算)_ | 最后一步为 1 | `rewards` (T,) | — |
| _(计算)_ | 最后一步为 1 | `dones` (T,) | — |

**no-op 过滤规则**（`is_noop()` 函数）：满足以下两个条件则视为 no-op：
1. 动作前 6 维（EEF）的 L2 范数 < `1e-4`
2. 夹爪动作与上一步相同（防止误删只动夹爪的动作）

---

## 阶段 2 → 阶段 3：重生成 HDF5 → OpenVLA RLDS

`LIBERO_Object_dataset_builder.py` 中 `_parse_example` 的字段映射：

| 重生成 HDF5 字段 | RLDS 字段 | 变换说明 |
|---|---|---|
| `obs/agentview_rgb` (T,256,256,3) | `observation/image` (256,256,3) | `[::-1,::-1]` 旋转 180°（修正环境渲染上下颠倒） |
| `obs/eye_in_hand_rgb` (T,256,256,3) | `observation/wrist_image` (256,256,3) | `[::-1,::-1]` 旋转 180° |
| `obs/ee_states` (T,6) + `obs/gripper_states` (T,2) | `observation/state` (8,) | `np.concatenate` 拼接成 8D |
| `obs/joint_states` (T,7) | `observation/joint_state` (7,) | 直接复制 |
| `actions` (T,7) | `action` (7,) | 直接复制 |
| _(文件名解析)_ | `language_instruction` (Text) | 从 HDF5 文件名中提取任务描述字符串 |
| _(计算)_ | `reward` (float32) | 最后一步 1.0，其余 0.0 |
| _(常量)_ | `discount` (float32) | 恒为 1.0 |
| _(计算)_ | `is_first` (bool) | `i == 0` |
| _(计算)_ | `is_last` (bool) | `i == T-1` |
| _(计算)_ | `is_terminal` (bool) | `i == T-1` |
| _(文件路径)_ | `episode_metadata/file_path` (Text) | HDF5 文件完整路径 |

**以下 HDF5 字段未被使用（丢弃）：**
- `obs/ee_pos`、`obs/ee_ori`（已包含在 ee_states 中）
- `robot_states`（仿真复合状态）
- `states`（仿真内部状态，不用于训练）
- `rewards`、`dones`（在 RLDS 中重新计算）

---

## 阶段 3 → 阶段 4：OpenVLA RLDS → Waypoint RLDS

`rlds_wp_extract.py`（AWE waypoint 提取）在 OpenVLA RLDS 基础上的变更：

| OpenVLA RLDS 字段 | Waypoint RLDS 字段 | 变化 |
|---|---|---|
| `observation/image` (256,256,3) | `observation/image` **(224,224,3)** | resize |
| `observation/wrist_image` (256,256,3) | `observation/wrist_image` **(224,224,3)** | resize |
| `observation/state` (8,) | `observation/state` (8,) | 不变 |
| `observation/joint_state` (7,) | `observation/joint_state` (7,) | 不变 |
| `action` (7,) | `action` (7,) | 不变 |
| `language_instruction` | `language_instruction` | 不变 |
| `reward` / `discount` / `is_first` / `is_last` / `is_terminal` | 同名字段 | 不变 |
| _(不存在)_ | **`waypoint_duration`** (int32) | 新增：到达该 waypoint 的持续步数 |
| _(不存在)_ | **`is_waypoint_end`** (bool) | 新增：是否为 waypoint 终点 |
| _(不存在)_ | **`original_step_index`** (int32) | 新增：在原始完整轨迹中的步索引 |

---

## observation.state (8D) 的拼接来源

```
环境观测 (eval/replay)         重生成 HDF5              RLDS
───────────────────          ───────────          ──────────────────
robot0_eef_pos (3)       ─┐
                          ├→ ee_states (6)  ─┐
quat2axisangle(           │                  ├─→ observation/state (8,)
  robot0_eef_quat) (3)   ─┘                  │
                                             │
robot0_gripper_qpos (2)  ─→ gripper_states(2)┘
```

---

## action (7D) 各维含义

| 索引 | 维度 | 含义 | 备注 |
|---|---|---|---|
| `[0:3]` | 3 | EEF 位置增量 (Δx, Δy, Δz) | Delta 控制 |
| `[3:6]` | 3 | EEF 旋转增量 (Δaxis-angle) | Delta 控制 |
| `[6]` | 1 | 夹爪指令 | 原始：-1=开, +1=关；训练时归一化为 0=关, 1=开 |

夹爪归一化（`normalize_gripper_libero`）：原始值 `[-1,1]` → clip 到 `[0,1]` → 取反 `(1-x)`。
归一化时夹爪维度 (dim 6) **不参与** Q99 percentile 归一化，`action_norm_mask = [True]*6 + [False]`。

---

## 相关文件索引

| 文件 | 作用 |
|---|---|
| `projects/openvla/experiments/robot/libero/regenerate_libero_dataset.py` | 阶段 1→2：重放生成新 HDF5 |
| `projects/rlds_dataset_builder/LIBERO_Object/LIBERO_Object_dataset_builder.py` | 阶段 2→3：HDF5 转 RLDS |
| `projects/awe/example/rlds_wp_extract.py` | 阶段 3→4：添加 waypoint 标注 |
| `projects/libero_openvla_rlds_features.json` | 阶段 4 最终 RLDS feature schema |
| `projects/openpi/src/openpi/waypoint/robot_config.py` | 推理时的归一化配置 |
| `projects/openvla/experiments/robot/libero/libero_utils.py` | 推理时图像预处理（含 180° 旋转） |
