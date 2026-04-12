# LIBERO Eval 指南

> 目标读者：AI Agent。假设基础环境（apt 包、uv sync、TF、NCCL、transformers patch）已就绪。
> 项目根目录：`/workspace/ae_agu`

---

## 1. 安装评测依赖（首次需要，已装可跳过）

```bash
cd /workspace/ae_agu

# 1.1 robosuite 及相关包
uv pip install --python .venv/bin/python \
    robosuite==1.4.1 transforms3d bddl easydict "gym==0.26.2"

# 1.2 LIBERO（editable install）
uv pip install --python .venv/bin/python -e third_party/libero

# 1.3 修复 PyTorch 2.6+ weights_only 问题
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    third_party/libero/libero/libero/benchmark/__init__.py

# 1.4 验证
echo "N" | PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH \
    .venv/bin/python -c "
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()['libero_object']()
print(f'LIBERO Object: {bm.n_tasks} tasks')
"
# 期望输出: LIBERO Object: 10 tasks
```

---

## 2. 修改评测配置

配置文件：`configs/eval_waypoint_joint_libero.yaml`

**必须修改的字段：**

| 字段 | 说明 | 示例 |
|------|------|------|
| `joint_checkpoint` | checkpoint 目录，内含 `model_merged.safetensors` 或 `model.safetensors` | `/workspace/ae_agu/checkpoints/xxx/5000` |
| `video_out_path` | 视频输出目录（每次 eval 用不同路径避免覆盖） | `<checkpoint_dir>/eval_libero_videos` |

**可选修改的字段：**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `dataset_statistics_path` | `/workspace/data/dataset_statistics.json` | 归一化统计文件（action=7 维, proprio=6 维） |
| `num_trials_per_task` | `3` | 每个 task 跑几个 trial，10 tasks × N trials |
| `eval_seed` | `7` | 固定随机种子，注释掉则不固定 |

> LoRA 训练的模型必须先 merge 后才能 eval。eval 会从 `joint_checkpoint` 目录下优先加载 `model_merged.safetensors`，不存在则加载 `model.safetensors`。

---

## 3. 启动前核查

```bash
cd /workspace/ae_agu

# 检查模型文件
ls <joint_checkpoint_dir>/model_merged.safetensors 2>/dev/null \
    || ls <joint_checkpoint_dir>/model.safetensors && echo "model OK"

# 检查统计文件
.venv/bin/python -c "
import json, sys; d = json.load(open('/workspace/data/dataset_statistics.json'))
print('action:', len(d['action']['q99']), 'proprio:', len(d['proprio']['q99']))
"
# 期望: action: 7  proprio: 6

# 检查 OpenGL
.venv/bin/python -c "from OpenGL.GL import glGetError; print('OpenGL OK')"
```

---

## 4. 启动评测

**关键要点：**
- 用 `CUDA_VISIBLE_DEVICES` 指定 GPU（选显存空闲最多的一张，eval 约需 ~8-10GB）
- 用 `export` 设置环境变量（不能用 inline 方式，否则管道后的进程拿不到）
- 用 `echo "N" |` 管道自动回答 LIBERO 的 dataset path 交互提示
- LIBERO 需要通过 `PYTHONPATH` 导入（pip editable install 不够，还需要 PYTHONPATH）

```bash
cd /workspace/ae_agu
mkdir -p logs .torch_cache

# 选 GPU：查看哪张卡空闲显存最多
nvidia-smi --query-gpu=index,memory.free --format=csv

# 启动（以 GPU 1 为例，LOG_NAME 自定义）
tmux kill-session -t eval 2>/dev/null; sleep 1
tmux new-session -d -s eval \
    'cd /workspace/ae_agu && \
     export CUDA_VISIBLE_DEVICES=1 && \
     export MUJOCO_GL=osmesa && \
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
     export TORCHINDUCTOR_CACHE_DIR=/workspace/ae_agu/.torch_cache && \
     export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH && \
     export PYTHONFAULTHANDLER=1 && \
     echo "N" | .venv/bin/python -u -m openpi.waypoint.eval_libero \
         --config configs/eval_waypoint_joint_libero.yaml \
         2>&1 | tee logs/<LOG_NAME>.log'
```

> `torch_compile: true` 首次编译 CUDA kernel 需 5-10 分钟，缓存在 `.torch_cache/`，后续复用。

---

## 5. 监控与结果

```bash
# 查看日志
tail -f logs/<LOG_NAME>.log

# 或 attach 到 tmux
tmux attach -t eval   # Ctrl+B, D 退出不杀进程
```

**正常启动标志**（模型加载后 ~30s）：
```
INFO:__main__:Task 0/10: pick_up_the_alphabet_soup...
INFO:__main__:  [replan 1] VLM: 7 waypoints, 7 valid, durations=[...], vlm_time=...ms
```

**评测完成标志**：
```
Overall success rate: XX.XX% (N/M)
```

**耗时参考**：10 tasks × 3 trials ≈ 30 分钟，10 tasks × 50 trials ≈ 8-10 小时。

---

## 6. 常见问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'robosuite'` | 未安装评测依赖 | 执行步骤 1.1 |
| `ModuleNotFoundError: No module named 'libero'` | PYTHONPATH 未设置 | 启动命令中必须 `export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH` |
| 卡在 `Do you want to specify a custom path for the dataset folder?` | LIBERO 首次运行的交互提示 | 启动命令用 `echo "N" \|` 管道输入 |
| `libosmesa6-dev` 相关错误 | 缺少 MuJoCo 离屏渲染依赖 | `sudo apt-get install -y libosmesa6-dev` |
