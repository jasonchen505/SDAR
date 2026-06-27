# SDAR 项目复现计划 (8x RTX 3090)

> 基于论文配置适配 3090 资源的完整复现方案

---

## 一、资源评估与约束分析

### 1.1 硬件对比

| 项目 | 论文配置 | 实际资源 | 差距 |
|------|---------|---------|------|
| GPU | 8x H800 (80GB HBM) | 8x RTX 3090 (24GB GDDR6X) | 显存 3.3x |
| 互联 | NVLink | PCIe 4.0 | 带宽 ~6x |
| FP16 算力 | ~990 TFLOPS | ~356 TFLOPS | ~2.8x |
| BF16 支持 | 原生 | 支持 | - |

### 1.2 显存需求估算

**Qwen2.5-3B-Instruct (FP16/BF16)**：
```
模型参数：3B × 2 bytes = 6 GB
优化器状态（Adam）：3B × 8 bytes = 24 GB（需要 offload）
梯度：3B × 2 bytes = 6 GB
激活值：~2-4 GB（gradient checkpointing 后）
vLLM KV Cache：~4-8 GB
总计：~42-48 GB（单卡放不下）
```

**Qwen2.5-7B-Instruct (FP16/BF16)**：
```
模型参数：7B × 2 bytes = 14 GB
优化器状态：7B × 8 bytes = 56 GB（必须 offload）
梯度：7B × 2 bytes = 14 GB
激活值：~4-6 GB
vLLM KV Cache：~6-10 GB
总计：~94-100 GB（必须多卡 + offload）
```

### 1.3 关键限制

```python
# 1. 显存限制
#    - 3090 单卡 24GB，需要 FSDP 分片 + offload
#    - 7B 模型必须启用 optimizer offload

# 2. 通信限制
#    - PCIe 带宽 ~32 GB/s，NVLink ~900 GB/s
#    - FSDP 通信开销更大，需要减少通信频率

# 3. 计算限制
#    - 3090 算力较弱，训练时间更长
#    - 需要减小 batch size，增加梯度累积
```

---

## 二、分阶段复现计划

### Phase 0: 环境搭建 (Day 1)

**目标**：完成基础环境配置

```bash
# 1. 创建 conda 环境
conda create -n sdar python=3.12 -y
conda activate sdar

# 2. 安装 PyTorch (CUDA 11.8)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 vLLM (3090 兼容版本)
pip install vllm==0.6.6.post1  # 3090 兼容版本

# 4. 安装 flash-attn
pip install flash-attn==2.6.3 --no-build-isolation

# 5. 安装项目依赖
cd /home/chenyizhou/SDAR
pip install -e .

# 6. 安装 ALFWorld
pip install alfworld
alfworld-download -f

# 7. 验证安装
python -c "import torch; print(torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
python -c "import alfworld; print('ALFWorld OK')"
```

**验证点**：
- [ ] `torch.cuda.is_available()` 返回 True
- [ ] 8 张 GPU 都能被识别
- [ ] vLLM 能正常初始化

### Phase 1: 数据准备 (Day 1)

**目标**：准备训练数据

```bash
# 1. ALFWorld 数据
export ALFWORLD_DATA=$HOME/data/alfworld
alfworld-download -f

# 2. 准备训练/验证数据
cd /home/chenyizhou/SDAR
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size 128

# 3. 验证数据
ls -la $HOME/data/verl-agent/text/
# 应该看到 train.parquet 和 test.parquet
```

**验证点**：
- [ ] `train.parquet` 存在且非空
- [ ] `test.parquet` 存在且非空
- [ ] 数据格式正确（包含 prompt 字段）

### Phase 2: 单卡功能验证 (Day 2)

**目标**：验证代码能在单卡上运行

```bash
# 最小配置测试（仅用 1 张 GPU）
cd /home/chenyizhou/SDAR

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=0.01 \
    +algorithm.sdar.gate_beta=5.0 \
    +algorithm.sdar.skills_dir=skills/alfworld \
    +algorithm.sdar.skill_all=false \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=5 \
    env.rollout.n=2 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='sdar_test' \
    trainer.experiment_name='single_gpu_test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    trainer.val_before_train=True
```

**验证点**：
- [ ] 训练能正常启动
- [ ] 没有 OOM
- [ ] Loss 能正常下降
- [ ] 验证能正常运行

### Phase 3: 3B 模型多卡训练 (Day 3-4)

**目标**：在 8x 3090 上训练 Qwen2.5-3B

**配置文件**：`run_alfworld_3b_3090.sh`

```bash
#!/bin/bash
set -x

ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# SDAR 超参数
sdar_coef=0.01
gate_beta=5.0
skill_all=false

# 训练配置（适配 3090）
train_data_size=16
val_data_size=128
group_size=4  # 从 8 降到 4，减少显存

experiment_name="sdar_3b_3090_coef${sdar_coef}_beta${gate_beta}"
export ALFWORLD_DATA=$HOME/data/alfworld

export WANDB_API_KEY=your_key_here

# 准备数据
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=1536 \
    data.max_response_length=384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=$sdar_coef \
    +algorithm.sdar.gate_beta=$gate_beta \
    +algorithm.sdar.skills_dir=skills/alfworld \
    +algorithm.sdar.skill_all=$skill_all \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld_3090' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
```

**关键调整**：

| 参数 | 原始值 | 3090 适配值 | 原因 |
|------|--------|------------|------|
| `group_size` | 8 | 4 | 减少显存占用 |
| `max_prompt_length` | 2048 | 1536 | 减少序列长度 |
| `max_response_length` | 512 | 384 | 减少序列长度 |
| `ppo_micro_batch_size_per_gpu` | 32 | 8 | 减少单卡 batch |
| `param_offload` | False | True | 卸载参数到 CPU |
| `optimizer_offload` | False | True | 卸载优化器到 CPU |
| `gpu_memory_utilization` | 0.6 | 0.5 | 减少 vLLM 显存 |
| `enforce_eager` | False | True | 减少显存碎片 |

**验证点**：
- [ ] 8 卡训练能正常启动
- [ ] 显存占用 < 24GB/卡
- [ ] 训练速度可接受（~10-15 min/epoch）
- [ ] Loss 正常下降
- [ ] 验证指标正常

### Phase 4: 完整训练与验证 (Day 5-7)

**目标**：完成 150 epochs 训练，复现论文结果

```bash
# 启动完整训练
bash examples/sdar_trainer/run_alfworld_3b_3090.sh

# 监控训练
# 终端 1：GPU 使用率
watch -n 1 nvidia-smi

# 终端 2：WandB 日志
# 查看 dashboard

# 终端 3：训练日志
tail -f outputs/sdar_3b_3090/training.log
```

**监控指标**：

```python
# 关键指标（通过 WandB 监控）
metrics_to_watch = {
    # 性能指标
    "episode/success_rate": "目标 > 80%",
    "critic/score/mean": "应该逐渐上升",
    
    # 训练稳定性
    "sdar/teacher_student_gap_mean": "应该在 -0.5 ~ 0",
    "sdar/gate_active_ratio": "应该逐渐上升",
    "sdar/loss": "应该逐渐下降",
    
    # 资源使用
    "perf/throughput": "tokens/sec/GPU",
    "timing_s/step": "每步时间",
}
```

**预期结果**：

| 指标 | 预期值 | 容差 |
|------|--------|------|
| ALFWorld 成功率 | ~80% | ±5% |
| 训练时间 | ~24-36 小时 | - |
| 显存占用 | ~20-22 GB/卡 | - |

### Phase 5: 消融实验 (Day 8-10)

**目标**：验证 SDAR 各组件的有效性

**实验 1：门控策略对比**

```bash
# Gap gating (默认)
bash run_alfworld_3b_3090.sh --gate_type gap --gate_beta 5.0

# Entropy gating
bash run_alfworld_3b_3090.sh --gate_type entropy --gate_beta 5.0

# 无门控 (naive GRPO+OPSD)
bash run_alfworld_3b_3090.sh --gate_beta 0.0
```

**实验 2：β 参数消融**

```bash
for beta in 0 1 5 10; do
    bash run_alfworld_3b_3090.sh --gate_beta $beta
done
```

**实验 3：λ 参数消融**

```bash
for coef in 0.001 0.01 0.1; do
    bash run_alfworld_3b_3090.sh --sdar_coef $coef
done
```

### Phase 6: 7B 模型尝试 (Day 11-14)

**目标**：尝试在 3090 上训练 7B 模型

**配置文件**：`run_alfworld_7b_3090.sh`

```bash
#!/bin/bash
set -x

# 7B 模型在 3090 上的极限配置
# 必须启用所有 offload，使用最小 batch size

python3 -m verl.trainer.main_sdar \
    # ... 其他参数 ...
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    env.rollout.n=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    # ...
```

**风险**：
- 可能 OOM，需要进一步降低配置
- 训练速度会很慢（可能需要 3-5 天）
- 如果失败，可以只复现 3B 结果

---

## 三、关键问题与解决方案

### 3.1 OOM 问题

**症状**：`CUDA out of memory`

**排查步骤**：

```bash
# 1. 检查当前显存使用
nvidia-smi

# 2. 逐步降低配置
# 优先级：
# 1) 减小 batch size
# 2) 减小序列长度
# 3) 启用 offload
# 4) 减小 group size
# 5) 降低 gpu_memory_utilization
```

**解决方案**：

```python
# 方案 1：减小 batch size
ppo_micro_batch_size_per_gpu=4  # 从 8 降到 4

# 方案 2：减小序列长度
max_prompt_length=1024  # 从 1536 降到 1024
max_response_length=256  # 从 384 降到 256

# 方案 3：启用 offload
param_offload=True
optimizer_offload=True

# 方案 4：减小 group size
env.rollout.n=2  # 从 4 降到 2

# 方案 5：降低 vLLM 显存
gpu_memory_utilization=0.4  # 从 0.5 降到 0.4
```

### 3.2 训练速度慢

**症状**：每步训练时间过长

**优化方案**：

```python
# 1. 使用 torch compile（如果支持）
use_torch_compile=True

# 2. 启用 chunked prefill
enable_chunked_prefill=True

# 3. 调整 vLLM 参数
max_num_batched_tokens=4096  # 减少以降低显存
max_num_seqs=512  # 减少并发序列

# 4. 使用更高效的 rollout
# 如果 vLLM 太慢，尝试 sglang
rollout.name=sglang
```

### 3.3 通信瓶颈

**症状**：多卡训练比单卡慢

**排查**：

```bash
# 检查 GPU 间通信带宽
nvidia-smi topo -m

# 检查 PCIe 带宽
lspci -vvv | grep -i pci
```

**优化**：

```python
# 1. 减少通信频率
# 增大 micro_batch_size，减少梯度同步次数

# 2. 使用 gradient accumulation
# 在代码中实现梯度累积

# 3. 调整 FSDP 策略
# 使用更大的分片大小
fsdp_size=8  # 所有 GPU 一组
```

### 3.4 vLLM 兼容性

**症状**：vLLM 初始化失败或推理错误

**排查**：

```python
# 1. 检查 vLLM 版本
import vllm
print(vllm.__version__)

# 2. 检查 CUDA 版本兼容性
nvcc --version

# 3. 测试 vLLM 基本功能
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
```

**解决方案**：

```bash
# 如果 vLLM 有问题，降级到兼容版本
pip install vllm==0.6.6.post1

# 或者使用 HF rollout（更慢但更稳定）
actor_rollout_ref.rollout.name=hf
```

---

## 四、学习计划

### Week 1: 基础复现

| Day | 任务 | 产出 |
|-----|------|------|
| 1 | 环境搭建 + 数据准备 | 可运行的环境 |
| 2 | 单卡功能验证 | 最小配置能跑通 |
| 3-4 | 3B 模型多卡训练 | 训练正常进行 |
| 5-7 | 完整训练 | 3B 模型结果 |

### Week 2: 深入理解

| Day | 任务 | 产出 |
|-----|------|------|
| 8-10 | 消融实验 | 各组件有效性验证 |
| 11-14 | 7B 模型尝试 | 7B 结果（或失败分析） |

### Week 3: 代码理解

| Day | 任务 | 产出 |
|-----|------|------|
| 15-17 | 阅读核心代码 | 理解实现细节 |
| 18-21 | 修改和扩展 | 尝试改进 |

---

## 五、验证清单

### 5.1 环境验证

```bash
# 运行环境检查脚本
python -c "
import torch
import vllm
import alfworld
import verl

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'vLLM: {vllm.__version__}')
print('All checks passed!')
"
```

### 5.2 训练验证

```python
# 检查训练是否正常
# 1. Loss 应该下降
# 2. 成功率应该上升
# 3. 显存应该稳定

# 关键检查点
assert loss < initial_loss * 0.5, "Loss 没有下降"
assert success_rate > 0.5, "成功率太低"
assert gpu_memory < 24, "显存超限"
```

### 5.3 结果验证

```python
# 与论文结果对比
paper_results = {
    "ALFWorld-3B": 84.4,
    "ALFWorld-7B": 86.8,
    "SearchQA-3B": 58.5,
    "WebShop-Acc-3B": 77.4,
}

# 容差范围
tolerance = 5.0  # ±5%

for task, expected in paper_results.items():
    actual = get_result(task)
    diff = abs(actual - expected)
    status = "PASS" if diff <= tolerance else "WARN"
    print(f"[{status}] {task}: {actual:.1f}% (expected {expected:.1f}%, diff {diff:.1f}%)")
```

---

## 六、预期时间表

```
Week 1:
├── Day 1: 环境搭建 ✓
├── Day 2: 功能验证 ✓
├── Day 3-4: 3B 训练启动 ✓
└── Day 5-7: 3B 训练完成 ✓

Week 2:
├── Day 8-10: 消融实验
├── Day 11-12: 7B 训练尝试
└── Day 13-14: 结果整理

Week 3:
├── Day 15-17: 代码深入学习
└── Day 18-21: 改进尝试
```

**总预计时间**：2-3 周完成基本复现

---

## 七、备选方案

### 如果 3B 训练太慢

```python
# 方案 1：减少训练步数
trainer.total_epochs=50  # 从 150 降到 50

# 方案 2：使用更小的模型
model.path=Qwen/Qwen2.5-1.5B-Instruct

# 方案 3：减小验证频率
trainer.test_freq=20  # 从 5 改为 20
```

### 如果 7B 完全无法训练

```python
# 方案 1：使用 LoRA
model.lora_rank=16
model.lora_alpha=32

# 方案 2：使用 QLoRA（4-bit 量化）
# 需要额外配置 bitsandbytes

# 方案 3：只复现 3B 结果
# 论文中 3B 结果已经很有说服力
```

### 如果需要加速

```bash
# 方案 1：使用 DeepSpeed ZeRO
# 需要修改代码支持

# 方案 2：使用 gradient accumulation
# 在代码中添加梯度累积逻辑

# 方案 3：混合精度训练
# 确保使用 bf16 而不是 fp32
```

---

*计划创建时间：2026-06-27*
*硬件环境：8x RTX 3090 (24GB)*
*预计完成时间：2-3 周*
