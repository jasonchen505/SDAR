# SDAR 项目增量学习记录

> 基于前两轮分析，在实际复现过程中对比学习到的新知识点

---

## 一、前两轮知识回顾

### 1.1 第一轮：项目理解 (INTERVIEW_PREPARATION.md)

**已掌握**：
- SDAR 核心算法原理（token-level 门控）
- 项目代码结构（agent_system, verl, skills）
- 关键公式和超参数
- 面试考察点

### 1.2 第二轮：深度准备 (INTERVIEW_DEEP_DIVE.md)

**已掌握**：
- 五类面试问题应对策略
- 工程落地细节（分布式训练、内存优化）
- 问题定位方法论
- 业务场景分析

---

## 二、复现过程增量学习

### 2.1 环境搭建增量知识

#### 2.1.1 CUDA 版本兼容性问题

**新发现**：
```bash
# 论文使用 CUDA 12.x，但 3090 可能需要 CUDA 11.8
# 需要检查 PyTorch 和 vLLM 的 CUDA 兼容性

# 问题：vLLM 0.11.0 可能不支持 CUDA 11.8
# 解决方案：降级到 vLLM 0.6.x

# 增量知识点：
# 1. vLLM 版本与 CUDA 版本强绑定
# 2. 3090 (Ampere) 和 H800 (Hopper) 架构差异影响
# 3. flash-attn 版本也需要匹配
```

**代码验证**：
```python
# 检查 CUDA 兼容性
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU architecture: {torch.cuda.get_device_capability()}")
# 3090 应该返回 (8, 6)，H800 返回 (9, 0)
```

#### 2.1.2 显存管理细节

**新发现**：
```python
# 原始配置假设 80GB 显存，3090 只有 24GB
# 需要理解 FSDP 的显存分布

# 增量知识点：
# 1. FSDP 分片策略：参数、梯度、优化器状态分别分片
# 2. offload 策略：CPU offload vs NVMe offload
# 3. 显存碎片化：enforce_eager=True 可以减少碎片

# 显存组成（单卡，3B 模型）：
# - 模型参数（分片后）：~750 MB
# - 优化器状态（分片后）：~3 GB（offload 到 CPU）
# - 梯度（分片后）：~750 MB
# - 激活值：~2-4 GB（gradient checkpointing 后）
# - vLLM KV Cache：~4-6 GB
# - 其他开销：~2-3 GB
# 总计：~13-18 GB（可控）
```

### 2.2 训练配置增量知识

#### 2.2.1 Group Size 对训练的影响

**新发现**：
```python
# 论文用 group_size=8，3090 可能需要降到 4

# 增量知识点：
# 1. Group size 影响 GRPO 优势估计的准确性
# 2. 太小（如 2）：方差大，训练不稳定
# 3. 太大（如 16）：计算开销线性增长

# 实验设计：
# - group_size=2：快速验证，但结果可能不准
# - group_size=4：平衡选择
# - group_size=8：论文默认，需要更多显存

# 代码验证点（core_algos.py:113-174）：
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    # index 用于标识同一 group 的样本
    # group_size 越大，均值/方差估计越准
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    # ...
```

#### 2.2.2 序列长度与训练效率

**新发现**：
```python
# 论文用 max_prompt_length=2048, max_response_length=512
# 3090 可能需要缩短

# 增量知识点：
# 1. 序列长度直接影响显存占用（O(n²) 注意力）
# 2. 多轮 Agent 的 prompt 包含历史观察，可能很长
# 3. response 通常较短（单步动作）

# 优化策略：
# 1. 减小 history_length（环境配置）
# 2. 截断过长的 prompt（truncation='left'）
# 3. 使用 remove_padding 减少无效计算

# 代码位置（rollout_loop.py:168-180）：
if len(raw_prompt_ids) > self.config.data.max_prompt_length:
    if self.config.data.truncation == "left":
        raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length:]
    elif self.config.data.truncation == "right":
        raw_prompt_ids = raw_prompt_ids[:self.config.data.max_prompt_length]
```

#### 2.2.3 vLLM 显存管理

**新发现**：
```python
# gpu_memory_utilization 控制 vLLM 可用显存比例

# 增量知识点：
# 1. vLLM 显存 = 模型权重 + KV Cache
# 2. gpu_memory_utilization=0.6 意味着 60% 显存给 vLLM
# 3. 太小会导致 OOM，太大会挤占训练显存

# 3090 配置建议：
# - 3B 模型：gpu_memory_utilization=0.5
# - 7B 模型：gpu_memory_utilization=0.4

# 代码位置（ppo_trainer.yaml:115）：
# gpu_memory_utilization: 0.5

# 调试方法：
import torch
for i in range(torch.cuda.device_count()):
    total = torch.cuda.get_device_properties(i).total_mem / 1e9
    used = torch.cuda.memory_allocated(i) / 1e9
    print(f"GPU {i}: {used:.1f}/{total:.1f} GB used")
```

### 2.3 算法实现增量知识

#### 2.3.1 SDAR Loss 的数值稳定性

**新发现**：
```python
# sdar_utils.py 中的实现需要注意数值稳定性

# 增量知识点：
# 1. log_prob 可能是 -inf（概率为 0 的 token）
# 2. sigmoid 输入过大/过小会饱和
# 3. 需要 clamp 避免数值问题

# 代码细节（sdar_utils.py:42-52）：
teacher_log_probs = teacher_log_probs.detach()
delta_t = teacher_log_probs - student_log_probs.detach()  # 可能很大
gate = torch.sigmoid(gate_beta * delta_t).detach()  # beta=5 可能导致饱和

# 潜在问题：
# - delta_t 很大时，sigmoid 饱和，梯度消失
# - 需要监控 delta_t 的范围

# 调试代码：
print(f"delta_t range: [{delta_t.min():.3f}, {delta_t.max():.3f}]")
print(f"gate range: [{gate.min():.3f}, {gate.max():.3f}]")
print(f"gate mean: {gate.mean():.3f}")
```

#### 2.3.2 Teacher-Student Gap 的动态变化

**新发现**：
```python
# 训练过程中，teacher-student gap 会动态变化

# 增量知识点：
# 1. 初期：gap 大部分为负（teacher 不比 student 更自信）
# 2. 中期：gap 逐渐向 0 收敛
# 3. 后期：部分 token gap 为正（teacher 提供有益信号）

# 监控指标（skillsd_ray_trainer.py:222-230）：
metrics["skillsd/teacher_student_gap_mean"]  # 应该从负值向 0 收敛
metrics["skillsd/gate_active_ratio"]  # 应该逐渐上升

# 调试代码：
# 在训练循环中添加
if global_steps % 10 == 0:
    print(f"Step {global_steps}:")
    print(f"  Gap mean: {metrics['skillsd/teacher_student_gap_mean']:.4f}")
    print(f"  Gate active: {metrics['skillsd/gate_active_ratio']:.2%}")
    print(f"  SDAR loss: {metrics['sdar/loss']:.4f}")
```

#### 2.3.3 Skill 检索的实现细节

**新发现**：
```python
# Skill 检索有多种策略，影响训练效果

# 增量知识点：
# 1. Keyword Matching：简单但有效
# 2. UCB：需要在线更新，实现复杂
# 3. Random：用于消融实验

# 代码位置（rlsd_utils.py:75-110）：
def get_privileged_info_from_prompt(self, prompt_text: str) -> str:
    # 基于关键词匹配
    text_lower = prompt_text.lower()
    matched_tasks = []
    for task_type, keywords in self.task_keywords.items():
        if any(kw in text_lower for kw in keywords):
            matched_tasks.append(task_type)
    # ...

# 调试方法：
# 打印 skill 检索结果
skill_provider = SkillProvider(skills_dir="skills/alfworld")
test_prompt = "put the apple on the table"
skill = skill_provider.get_privileged_info_from_prompt(test_prompt)
print(f"Prompt: {test_prompt}")
print(f"Retrieved skill:\n{skill[:200]}...")
```

### 2.4 分布式训练增量知识

#### 2.4.1 FSDP 通信开销

**新发现**：
```python
# PCIe 带宽限制会影响 FSDP 性能

# 增量知识点：
# 1. FSDP 在 forward/backward 时需要 all-gather 参数
# 2. PCIe 4.0 x16 带宽 ~32 GB/s，NVLink ~900 GB/s
# 3. 3090 的通信开销比 H800 大很多

# 优化策略：
# 1. 增大 micro_batch_size，减少通信频率
# 2. 使用 gradient accumulation
# 3. 调整 FSDP 分片策略

# 监控方法：
# 使用 torch profiler 分析通信开销
import torch.profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # 训练一步
    pass
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

#### 2.4.2 Ray 集群配置

**新发现**：
```python
# Ray 用于分布式任务调度

# 增量知识点：
# 1. Ray 默认使用所有 CPU 核心
# 2. 在共享服务器上需要限制资源
# 3. 环境 worker 也需要 CPU 资源

# 配置（ppo_trainer.yaml:289）：
# ray_init:
#   num_cpus: null  # null 表示使用所有 CPU

# 3090 配置建议：
ray_init:
  num_cpus: 32  # 限制 CPU 使用，留资源给环境
env:
  resources_per_worker:
    num_cpus: 0.1  # 每个环境 worker 用 0.1 个 CPU
```

### 2.5 环境交互增量知识

#### 2.5.1 ALFWorld 环境特性

**新发现**：
```python
# ALFWorld 是文本游戏环境

# 增量知识点：
# 1. 观察是文本描述
# 2. 动作是文本命令
# 3. 有可行动作列表（admissible_actions）
# 4. 任务类型：Pick, Look, Clean, Heat, Cool, Pick2

# 环境管理器（base.py:63-97）：
def step(self, text_actions: List[str]):
    actions, valids = self.projection_f(text_actions)  # 文本 -> 动作
    next_obs, rewards, dones, infos = self.envs.step(actions)
    # ...

# 调试方法：
# 打印环境交互
obs, infos = envs.reset()
print(f"Initial observation: {obs['text'][0][:200]}")
print(f"Admissible actions: {infos[0].get('admissible_actions', [])}")
```

#### 2.5.2 奖励函数设计

**新发现**：
```python
# 奖励函数直接影响训练效果

# 增量知识点：
# 1. ALFWorld：任务成功奖励 1，失败奖励 0
# 2. 可选：invalid action penalty
# 3. 可选：步数惩罚

# 代码位置（episode.py:72-79）：
episode_rewards = data_item.non_tensor_batch['episode_rewards']
episode_lengths = data_item.non_tensor_batch['episode_lengths']

if self.normalize_by_length:
    score = episode_rewards / episode_lengths
else:
    score = episode_rewards

# 调试方法：
# 打印奖励分布
print(f"Rewards: {episode_rewards}")
print(f"Lengths: {episode_lengths}")
print(f"Success rate: {(episode_rewards > 0).mean():.2%}")
```

### 2.6 监控与调试增量知识

#### 2.6.1 WandB 集成

**新发现**：
```python
# WandB 用于实验跟踪

# 增量知识点：
# 1. 需要设置 WANDB_API_KEY
# 2. 可以离线运行（WANDB_MODE=offline）
# 3. 关键指标需要手动记录

# 配置（run_alfworld_3b.sh:17）：
export WANDB_API_KEY=your_key_here

# 代码位置（skillsd_ray_trainer.py:73-78）：
from verl.utils.tracking import Tracking
logger = Tracking(
    project_name=self.config.trainer.project_name,
    experiment_name=self.config.trainer.experiment_name,
    default_backend=self.config.trainer.logger,  # ['console', 'wandb']
    config=OmegaConf.to_container(self.config, resolve=True),
)

# 关键指标：
metrics_to_log = {
    "episode/success_rate": success_rate,
    "sdar/gate_mean": gate_mean,
    "sdar/teacher_student_gap_mean": gap_mean,
    "perf/throughput": throughput,
}
logger.log(data=metrics_to_log, step=global_steps)
```

#### 2.6.2 调试工具

**新发现**：
```python
# 项目内置了调试工具

# 增量知识点：
# 1. SAVE_SDAR_DEBUG=1 启用详细日志
# 2. 每隔 test_freq 步保存 per-token gap 数据
# 3. 可以分析 teacher-student gap 分布

# 代码位置（skillsd_ray_trainer.py:232-276）：
if os.environ.get("SAVE_SDAR_DEBUG", "0") == "1":
    # 保存 per-token 数据
    save_path = os.path.join(save_dir, f"step_{self.global_steps}.jsonl")
    with open(save_path, "w") as f:
        for i in range(bs):
            record = {
                "global_step": self.global_steps,
                "tokens": tokens_i,
                "gaps": gaps_i,
                "teacher_log_probs": teacher_lps_i,
                "student_log_probs": student_lps_i,
            }
            f.write(json.dumps(record) + "\n")

# 使用方法：
export SAVE_SDAR_DEBUG=1
export SAVE_SDAR_DEBUG_DIR=outputs/debug
bash run_alfworld_3b_3090.sh

# 分析结果：
import json
with open("outputs/debug/step_50.jsonl") as f:
    data = [json.loads(line) for line in f]
    
# 分析 gap 分布
gaps = [g for sample in data for g in sample["gaps"]]
print(f"Gap mean: {np.mean(gaps):.4f}")
print(f"Gap std: {np.std(gaps):.4f}")
print(f"Positive gap ratio: {(np.array(gaps) > 0).mean():.2%}")
```

---

## 三、实验结果对比

### 3.1 预期 vs 实际

| 指标 | 论文结果 (H800) | 预期结果 (3090) | 实际结果 | 差异分析 |
|------|----------------|----------------|---------|---------|
| ALFWorld-3B | 84.4% | ~78-82% | TBD | group_size 减小 |
| 训练时间 | ~4h | ~12-24h | TBD | 算力差异 |
| 显存占用 | ~60GB/卡 | ~20GB/卡 | TBD | offload 策略 |

### 3.2 消融实验对比

| 配置 | 预期成功率 | 实际成功率 | 差异原因 |
|------|-----------|-----------|---------|
| β=0 (无门控) | ~60% | TBD | 预期崩溃 |
| β=5 (默认) | ~80% | TBD | 最优配置 |
| λ=0.001 | ~75% | TBD | 蒸馏信号弱 |
| λ=0.1 | ~70% | TBD | 蒸馏主导 |

---

## 四、关键代码修改

### 4.1 适配 3090 的配置修改

```bash
# 原始配置（run_alfworld_3b.sh）
# group_size=8
# max_prompt_length=2048
# max_response_length=512
# ppo_micro_batch_size_per_gpu=32
# param_offload=False
# optimizer_offload=False

# 3090 适配配置
group_size=4
max_prompt_length=1536
max_response_length=384
ppo_micro_batch_size_per_gpu=8
param_offload=True
optimizer_offload=True
```

### 4.2 添加的调试代码

```python
# 在 skillsd_ray_trainer.py 的 fit() 方法中添加

# 1. 显存监控
def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_mem / 1e9
        used = torch.cuda.memory_allocated(i) / 1e9
        cached = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: Used {used:.1f}GB, Cached {cached:.1f}GB, Total {total:.1f}GB")

# 2. Gap 分布监控
def log_gap_distribution(delta_t, response_mask):
    valid_gaps = delta_t[response_mask.bool()].cpu().numpy()
    print(f"Gap stats: mean={valid_gaps.mean():.4f}, std={valid_gaps.std():.4f}")
    print(f"Positive ratio: {(valid_gaps > 0).mean():.2%}")

# 3. 训练速度监控
import time
step_start = time.time()
# ... 训练代码 ...
step_time = time.time() - step_start
print(f"Step time: {step_time:.2f}s")
```

---

## 五、问题排查记录

### 5.1 OOM 问题

**问题描述**：训练启动时 OOM

**排查过程**：
```bash
# 1. 检查显存使用
nvidia-smi

# 2. 逐步降低配置
# group_size: 8 -> 4 -> 2
# batch_size: 32 -> 16 -> 8 -> 4
# seq_length: 2048 -> 1536 -> 1024

# 3. 启用 offload
param_offload=True
optimizer_offload=True
```

**解决方案**：
```bash
# 最终配置
group_size=4
ppo_micro_batch_size_per_gpu=8
param_offload=True
optimizer_offload=True
gpu_memory_utilization=0.5
```

### 5.2 训练速度慢

**问题描述**：每步训练时间 > 5 分钟

**排查过程**：
```python
# 1. 分析时间分布
# 生成 (gen): ~40%
# Teacher forward: ~20%
# Ref forward: ~15%
# Advantage: ~10%
# Update: ~15%

# 2. 优化生成速度
# - 使用 vLLM
# - 调整 gpu_memory_utilization
# - 启用 chunked prefill

# 3. 减少前向传播次数
# - 合并 teacher 和 ref 的前向
# - 使用 gradient accumulation
```

**解决方案**：
```bash
# 优化配置
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.rollout.enable_chunked_prefill=True
```

### 5.3 训练不稳定

**问题描述**：Loss 突然飙升

**排查过程**：
```python
# 1. 检查梯度
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm: {grad_norm:.4f}")

# 2. 检查 teacher-student gap
print(f"Gap mean: {metrics['skillsd/teacher_student_gap_mean']:.4f}")
print(f"Gap std: {metrics['skillsd/teacher_student_gap_std']:.4f}")

# 3. 检查 gate 分布
print(f"Gate mean: {metrics['sdar/gate_mean']:.4f}")
print(f"Gate active ratio: {metrics['sdar/gate_active_ratio']:.2%}")
```

**解决方案**：
```bash
# 增加 KL 惩罚
actor_rollout_ref.actor.kl_loss_coef=0.02  # 从 0.01 增加到 0.02

# 降低学习率
actor_rollout_ref.actor.optim.lr=5e-7  # 从 1e-6 降低

# 增加 invalid action penalty
actor_rollout_ref.actor.invalid_action_penalty_coef=0.2  # 从 0.1 增加
```

---

## 六、学习心得总结

### 6.1 理论 vs 实践差距

**心得**：
```
1. 论文配置假设充足资源，实际需要大量适配
2. 超参敏感度比论文描述的更高
3. 调试和监控比训练本身更重要
4. 数值稳定性在实践中很关键
```

### 6.2 关键收获

**技术层面**：
```
1. FSDP 分片和 offload 的实际效果
2. vLLM 显存管理的细节
3. 多轮 Agent 的 rollout 实现
4. Token-level 门控的数值稳定性
```

**工程层面**：
```
1. 分布式训练的调试方法
2. 显存优化的多种策略
3. 训练监控的最佳实践
4. 问题排查的系统方法
```

### 6.3 后续学习方向

```
1. 尝试 7B 模型训练（可能需要 LoRA）
2. 在其他环境（WebShop, Search）复现
3. 尝试改进门控机制
4. 研究其他 RL+蒸馏方法
```

---

## 七、参考资料

### 7.1 代码位置索引

| 功能 | 文件路径 | 关键行号 |
|------|---------|---------|
| SDAR Loss | `verl/trainer/ppo/sdar_utils.py` | 14-67 |
| 训练循环 | `verl/trainer/ppo/skillsd_ray_trainer.py` | 64-333 |
| 多轮 Rollout | `agent_system/multi_turn_rollout/rollout_loop.py` | 300-433 |
| Skill 管理 | `verl/trainer/ppo/rlsd_utils.py` | 33-143 |
| 环境管理 | `agent_system/environments/base.py` | 34-174 |
| 奖励计算 | `agent_system/reward_manager/episode.py` | 20-96 |
| GRPO 算法 | `verl/trainer/ppo/core_algos.py` | 113-174 |

### 7.2 配置文件索引

| 配置 | 文件路径 |
|------|---------|
| 默认配置 | `verl/trainer/config/ppo_trainer.yaml` |
| ALFWorld 训练 | `examples/sdar_trainer/run_alfworld_3b.sh` |
| WebShop 训练 | `examples/sdar_trainer/run_webshop_3b.sh` |
| Search 训练 | `examples/sdar_trainer/run_search_3b.sh` |
| Skill 定义 | `skills/alfworld/skill_mapping.json` |

---

*文档创建时间：2026-06-27*
*基于 8x RTX 3090 复现计划*
*持续更新中...*
