# 训练中按需保存 Checkpoint（Force Save）

在 `train_waypoint_joint.py` 中实现了两种机制，允许在 **不中断训练** 的情况下，
在任意 step 强制保存完整 checkpoint（model + optimizer + metadata）。

---

## 背景

默认情况下，checkpoint 仅按 `save_interval`（如每 200 步）保存。
当需要在非整数倍 step 保存时（例如发现 loss 异常、想提前评估、或担心机器故障），
可以使用以下两种方式触发即时保存。

---

## 方式一：触发文件（SAVE_NOW）

### 原理

训练循环在每步 `optimizer.step()` 完成后，检查 checkpoint 目录下是否存在
名为 `SAVE_NOW` 的文件。若存在，立即调用 `save_checkpoint(force=True)`
保存当前 step，然后删除该文件。

### 代码位置

`train_waypoint_joint.py` 第 433–437 行：

```python
trigger_file = save_dir / "SAVE_NOW"
if is_main and trigger_file.exists():
    save_checkpoint(model, optimizer, step_for_save, save_dir,
                    is_main, save_interval, force=True)
    trigger_file.unlink(missing_ok=True)
```

### 使用方法

**立即保存（下一个 step 结束时生效）：**

```bash
touch checkpoints/<exp_name>/SAVE_NOW
```

例如：

```bash
touch checkpoints/waypoint_joint_calvin_sg_0.2_wp001/SAVE_NOW
```

文件创建后，最多等待一个 step 的时间（约 30 秒），checkpoint 就会保存。
保存完成后 `SAVE_NOW` 文件自动删除。

**在指定 step N 保存：**

需要在 step N-1 的 tqdm 输出出现后、step N 结束前创建文件。
可使用如下监控脚本自动完成：

```bash
# 示例：在 step 2485 保存
SAVE_DIR="checkpoints/waypoint_joint_calvin_sg_0.2_wp001"
TARGET_STEP=2484  # 监控到此 step 出现后创建文件，下一步(2485)保存

nohup bash -c '
while true; do
  STEP=$(tmux capture-pane -t joint_train -p 2>/dev/null \
         | grep -oP "step=\K[0-9]+" | tail -1)
  if [ -n "$STEP" ] && [ "$STEP" -ge '"$TARGET_STEP"' ] 2>/dev/null; then
    touch '"$SAVE_DIR"'/SAVE_NOW
    echo "[$(date)] step=$STEP >= '"$TARGET_STEP"' — trigger created"
    break
  fi
  sleep 3
done
' > /tmp/trigger_watch.log 2>&1 &
echo "Watcher PID: $!  (log: /tmp/trigger_watch.log)"
```

---

## 方式二：SIGUSR1 信号

### 原理

训练启动时，rank 0 进程注册 `SIGUSR1` 信号处理器。
收到信号后，处理器在当前 Python bytecode 间隙执行：

1. `torch.cuda.synchronize()` —— 确保 GPU 操作完成
2. `save_checkpoint(force=True)` —— 保存 model + optimizer + metadata

信号处理器通过闭包中的 `_save_state` 字典获取当前 step、model、optimizer 等引用。
训练循环在每步结束时更新 `_save_state["step"]`，保证保存的 step 编号正确。

### 代码位置

`train_waypoint_joint.py` 第 230–255 行（信号处理器注册），第 430 行（step 同步）。

### 使用方法

**1. 找到 rank 0 进程 PID：**

训练启动时日志会打印：

```
SIGUSR1 handler registered — `kill -SIGUSR1 <PID>` to force-save
```

也可以手动查找：

```bash
# 方法 a：从日志获取
grep "SIGUSR1 handler" logs/waypoint_joint_calvin.log

# 方法 b：从进程列表获取（取第一个 worker，即 rank 0）
pgrep -f "train_waypoint_joint" | head -1
```

**2. 发送信号：**

```bash
kill -SIGUSR1 <PID>
```

例如：

```bash
kill -SIGUSR1 $(pgrep -f "train_waypoint_joint" | head -1)
```

信号会在当前正在执行的 Python 指令完成后立即处理，保存耗时约 10–40 秒
（取决于模型大小和磁盘速度），期间训练暂停，保存完成后自动恢复。

---

## 保存内容

两种方式保存的文件与常规 checkpoint 完全一致：

```
checkpoints/<exp_name>/<step>/
├── model.safetensors   # 模型权重（safetensors 格式）
├── optimizer.pt        # 完整 optimizer 状态（含 Adam 一阶/二阶动量）
└── metadata.pt         # {"global_step": <step>, "timestamp": <unix_time>}
```

使用 `--resume` 重启时，`load_latest_checkpoint` 会自动找到编号最大的
checkpoint 目录恢复，包括手动触发保存的 checkpoint。

---

## 对训练的影响

- **不修改任何训练状态**：信号处理器和触发文件机制都只读取 model/optimizer
  状态并写入磁盘，不修改参数、梯度、学习率或训练步数。
- **常规 save_interval 保存不受影响**：`force=True` 走独立分支，
  正常的 `step % save_interval == 0` 逻辑不变。
- **短暂暂停**：保存期间 rank 0 主线程暂停（10–40 秒），rank 1 在 NCCL
  同步点等待。NCCL 默认超时 30 分钟，远大于保存耗时，不会触发超时。
- **可重复使用**：信号处理器注册后持续生效，可多次发送 SIGUSR1；
  触发文件消费后删除，下次需要时重新 `touch` 即可。

---

## 注意事项

1. **仅在 rank 0 生效**：信号应发送给 rank 0 进程，触发文件由 rank 0 检查。
   DDP 下 rank 1 的模型参数与 rank 0 同步，无需额外保存。

2. **保存时机**：
   - 触发文件在 `optimizer.step()` 之后检查，保存的是当前 step 更新后的模型。
   - SIGUSR1 可能在 step 的任意位置被处理。若在 `optimizer.step()` 之前，
     保存的是上一步的模型状态；若在之后，保存的是当前步的。
     `_save_state["step"]` 始终反映最近一次 `optimizer.step()` 后的 step 编号。

3. **磁盘空间**：每个 checkpoint 约 20 GB（7 GB model + 13 GB optimizer），
   频繁触发保存前请确认磁盘空间充足。

4. **ptrace 受限环境**：本方案不依赖 gdb/ptrace 注入，在 Docker 容器
   （seccomp 限制 ptrace）中可正常使用。
