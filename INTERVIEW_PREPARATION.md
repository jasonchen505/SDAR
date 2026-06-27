# SDAR 项目面试准备文档

> 本文档为 LLM Agent 后训练方向实习面试准备，基于 SDAR (Self-Distilled Agentic Reinforcement Learning) 项目深度分析。

---

## 一、项目概述与核心思想

### 1.1 项目定位

SDAR 是一个用于**多轮 LLM Agent 强化学习后训练**的框架，解决的核心问题是：

> 如何将 RL 的任务级奖励信号与 OPSD (On-Policy Self-Distillation) 的 token 级密集监督信号有效结合，实现多轮 Agent 的稳定训练？

### 1.2 核心贡献

1. **发现两个关键观察**：
   - **多轮 OPSD 不稳定性**：一旦 student 偏离 teacher 支持的轨迹，token 级监督变得不可靠，导致 KL 散度飙升和性能崩溃
   - **特权指导的非对称信任**：teacher 不是独立更强的模型，而是相同策略+特权上下文（如 skills），其信号质量不对称

2. **提出 SDAR 方法**：
   - RL (GRPO) 作为主优化骨架
   - OPSD 作为门控辅助目标
   - 通过 sigmoid 门控机制自适应调节每个 token 的蒸馏强度

3. **核心公式**：
   ```
   L_SDAR = Agg( g_t * (log π_teacher(y_t|s_t+) - log π_student(y_t|s_t)) )
   其中 g_t = σ(β * Δ_t), Δ_t = log π_teacher - log π_student (detach)
   ```

### 1.3 关键创新点

- **Token-level 门控**：让每个 token 自主决定监督强度
- **非对称处理**：正 gap (teacher 更自信) 增强蒸馏，负 gap 软衰减
- **Stop-gradient 门控**：g_t detach，梯度仅通过 student log-prob 流动，避免自引用耦合

---

## 二、技术架构与关键组件

### 2.1 整体架构

```
SDAR/
├── agent_system/           # Agent 系统核心
│   ├── environments/       # 环境管理器 (ALFWorld, WebShop, Search)
│   ├── memory/             # 记忆管理 (历史轨迹存储)
│   ├── multi_turn_rollout/ # 多轮 rollout 循环
│   └── reward_manager/     # 奖励计算
├── verl/                   # veRL 框架扩展
│   └── trainer/ppo/        # 训练器核心
│       ├── core_algos.py   # 基础算法 (GRPO, GAE, KL penalty)
│       ├── sdar_utils.py   # SDAR 损失计算
│       ├── rlsd_utils.py   # Skill 管理与 RLSD 工具
│       └── skillsd_ray_trainer.py  # SkillSD/SDAR 训练器
├── skills/                 # 技能库 (ALFWorld, WebShop, Search)
└── examples/               # 训练脚本
```

### 2.2 核心组件解析

#### 2.2.1 多轮 Rollout 循环 (`agent_system/multi_turn_rollout/rollout_loop.py`)

```python
class TrajectoryCollector:
    def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
        # 1. 环境重置获取初始观察
        obs, infos = envs.reset()
        
        # 2. 多步交互循环
        for _step in range(max_steps):
            # 预处理观察 -> 模型输入
            batch = self.preprocess_batch(gen_batch, obs)
            
            # Actor 生成响应
            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            
            # 解码动作并执行
            text_actions = tokenizer.batch_decode(batch_output['responses'])
            next_obs, rewards, dones, infos = envs.step(text_actions)
            
            # 收集轨迹数据
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
            
            if is_done.all():
                break
        
        return total_batch_list, episode_rewards, ...
```

**关键设计点**：
- 支持动态采样 (DAPO 风格)，过滤全成功/全失败的 group
- `active_masks` 标记已完成环境，避免无效计算
- `traj_uid` 保证同轨迹数据一致性

#### 2.2.2 环境管理器 (`agent_system/environments/base.py`)

```python
class EnvironmentManagerBase:
    def step(self, text_actions: List[str]):
        # 1. 文本动作 -> 环境动作 (projection)
        actions, valids = self.projection_f(text_actions)
        
        # 2. 执行环境步进
        next_obs, rewards, dones, infos = self.envs.step(actions)
        
        # 3. 构建观察字典
        next_observations = {
            'text': None,
            'image': next_obs,
            'anchor': None  # GiGPO 用的锚点观察
        }
        
        return next_observations, rewards, dones, infos
```

**支持的环境**：
- **ALFWorld**：文本游戏，6 类家务任务
- **WebShop**：网购场景，产品搜索与购买
- **Search-QA**：搜索增强问答 (NQ, TriviaQA, HotpotQA 等)

#### 2.2.3 Skill 管理 (`verl/trainer/ppo/rlsd_utils.py`)

```python
class SkillProvider:
    def __init__(self, skills_dir, skill_all=False):
        # 加载 skill_mapping.json
        self.skill_mapping = load_skill_mapping(skills_dir)
        self.skill_contents = load_skill_content(skills_dir, self.skill_mapping)
        self.task_to_skill = self.skill_mapping["task_to_skill"]
        self.task_keywords = self.skill_mapping.get("task_keywords", {})
    
    def get_privileged_info_from_prompt(self, prompt_text: str) -> str:
        # 基于关键词匹配任务类型
        matched_tasks = []
        for task_type, keywords in self.task_keywords.items():
            if any(kw in text_lower for kw in keywords):
                matched_tasks.append(task_type)
        
        # 组装 general_skills + task_specific_skills
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for task_type in matched_tasks:
            mapped_name = self.task_to_skill.get(task_type)
            parts.append(self.skill_contents[mapped_name])
        
        return "\n\n".join(parts)
```

**Skill 检索策略** (论文中提到 4 种)：
1. **UCB Retrieval**：多臂老虎机，探索-利用权衡
2. **Keyword Matching (KM)**：规则匹配，简单高效
3. **Full Retrieval**：检索所有技能
4. **Random Retrieval**：随机检索，用于消融实验

#### 2.2.4 SDAR 损失计算 (`verl/trainer/ppo/sdar_utils.py`)

```python
def compute_sdar_loss(
    student_log_probs: torch.Tensor,    # (bs, response_length)
    teacher_log_probs: torch.Tensor,    # (bs, response_length)
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    
    # 1. Detach teacher 信号
    teacher_log_probs = teacher_log_probs.detach()
    
    # 2. 计算 token 级 gap (detach，仅用于门控)
    delta_t = teacher_log_probs - student_log_probs.detach()
    
    # 3. Sigmoid 门控 (detach，避免梯度流过门控)
    gate = torch.sigmoid(gate_beta * delta_t).detach()
    
    # 4. 蒸馏损失 (梯度仅通过 student_log_probs)
    kl_per_token = teacher_log_probs - student_log_probs
    gated_kl = gate * kl_per_token
    
    # 5. 聚合损失
    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    # 6. 统计指标
    metrics = {
        "sdar/gate_mean": gate_mean.item(),
        "sdar/gate_active_ratio": gate_active.item(),  # g_t > 0.5 的比例
        "sdar/teacher_gap_mean": gap_mean.item(),
        "sdar/loss": loss.detach().item(),
    }
    
    return loss, metrics
```

**数学等价性** (论文 Proposition 1)：
```
L_SDAR = C - Agg( g_t * log π_θ(y_t|s_t) )
其中 C = Agg( g_t * log π_T(y_t|s_t+) ) 是常数
```
等价于 token 加权最大似然。

---

## 三、算法深度解析

### 3.1 GRPO (Group Relative Policy Optimization)

**核心思想**：组内相对优势估计，无需 critic 网络

```python
# 1. 采样 G 个响应
{y^(1), ..., y^(G)} ~ π_θ(·|x)

# 2. 计算组内归一化优势
A^(i) = (R(x,y^(i)) - μ_G) / σ_G

# 3. Clipped surrogate loss
L_GRPO = -E[ min(r_t * A, clip(r_t, 1-ε, 1+ε) * A) ]
其中 r_t = π_θ(y_t|s_t) / π_θ_old(y_t|s_t)
```

**关键特点**：
- 序列级优势，所有 token 共享同一 A 值
- 无需 critic，减少内存开销
- 但监督信号稀疏（credit assignment 难题）

### 3.2 OPSD (On-Policy Self-Distillation)

**核心思想**：teacher 分支看到特权上下文 (skills)，提供 token 级密集监督

```python
# Teacher 上下文：s_t+ = (x, c+, y_<t)  # c+ 是特权信息 (skills)
# Student 上下文：s_t = (x, y_<t)

# Token 级 KL 散度
D_RKL^(t) = D_KL( π_θ(·|s_t) || π_T(·|s_t+) )

# 单样本估计 (student 采样的 token)
Δ_t = log π_T(y_t|s_t+) - log π_θ(y_t|s_t)
```

**问题**：
- 直接用于多轮 Agent 会崩溃（compounding drift）
- 负 gap tokens 超过 50%（teacher 未必更优）

### 3.3 SDAR 的解决方案

**总损失**：
```
L(θ) = L_GRPO(θ) + λ * L_SDAR(θ)
```

**门控机制**：
```python
# Gap gating (默认)
g_t = σ(β * Δ_t)  # Δ_t = log π_T - log π_θ (detach)

# 正 gap (teacher 更自信) → g_t > 0.5 → 增强蒸馏
# 负 gap (teacher 不确定) → g_t < 0.5 → 软衰减
# β 控制门控锐度，β=5.0 是最优值
```

**梯度分析** (论文 Proposition 2)：
```
∇_θ L_SDAR = - Agg( g_t * ∇_θ log π_θ(y_t|s_t) )
```
- g_t 是 bounded scalar (0,1)
- 不会放大梯度爆炸 (Proposition 4)
- Stop-gradient 避免自引用耦合 (Proposition 5)

### 3.4 与其他方法对比

| 方法 | 优势估计 | 辅助损失 | Token 自适应 |
|------|---------|---------|-------------|
| GRPO | 序列级 A | 无 | 否 |
| RLSD | Token 级 Â_t | 无 | 是 (re-weight) |
| Skill-SD | 序列级 A | K3 散度 | 是 |
| GRPO+OPSD | 序列级 A | KL 散度 | 否 |
| **SDAR** | 序列级 A | 门控 KL | **是 (gated)** |

**SDAR 优势**：
1. RL 优势无偏（不修改 advantage）
2. 门控平滑有界（避免梯度爆炸）
3. 仅作用于 student 采样 token（计算高效）

---

## 四、代码实现细节

### 4.1 训练流程 (`verl/trainer/ppo/skillsd_ray_trainer.py`)

```python
class SkillSDRayTrainer(RLSDRayTrainer):
    def fit(self):
        for epoch in range(total_epochs):
            for batch_dict in self.train_dataloader:
                # 1. 多轮 Rollout
                gen_batch_output = self.traj_collector.multi_turn_loop(...)
                
                # 2. 计算 old_log_prob
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                
                # 3. Teacher 前向传播 (看到 skills)
                teacher_log_probs = self._compute_teacher_log_probs(batch)
                batch.batch["teacher_log_probs"] = teacher_log_probs
                
                # 4. 计算优势 (GRPO 风格)
                batch = compute_advantage(batch, adv_estimator="grpo", ...)
                
                # 5. Actor 更新 (内部计算 SDAR loss)
                actor_output = self.actor_rollout_wg.update_actor(batch)
```

### 4.2 Teacher 前向传播

```python
def _compute_teacher_log_probs(self, batch):
    # 构造 teacher 输入：原始 prompt + skill context
    teacher_batch = build_teacher_batch(batch, self.skill_provider)
    
    # Teacher 模型前向 (共享 actor 权重，但输入包含 skills)
    teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
    
    return teacher_output.batch["old_log_probs"]
```

### 4.3 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sdar_coef` (λ) | 0.01 | 蒸馏损失权重 |
| `gate_beta` (β) | 5.0 | Sigmoid 门控锐度 |
| `kl_loss_coef` | 0.01 | KL 惩罚系数 |
| `group_size` (G) | 8 | GRPO 组大小 |
| `max_steps` | 50 | 最大交互步数 |
| `learning_rate` | 1e-6 | 学习率 |

### 4.4 训练配置示例

```bash
python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    +algorithm.sdar.sdar_coef=0.01 \
    +algorithm.sdar.gate_beta=5.0 \
    +algorithm.sdar.skills_dir=skills/alfworld \
    env.env_name=alfworld/AlfredTWEnv \
    env.rollout.n=8 \  # group size
    trainer.total_epochs=150
```

---

## 五、面试考察点与深挖问题

### 5.1 基础概念类

#### Q1: 解释 RLHF/RLAIF 与 Agentic RL 的区别

**参考答案**：
- **RLHF/RLAIF**：单轮生成，奖励来自人类/AI 偏好，监督信号在序列末尾
- **Agentic RL**：多轮交互，奖励来自环境反馈，需要处理延迟奖励和 credit assignment
- **核心挑战**：多轮 Agent 的轨迹更长，token 重要性高度不均匀，直接应用 OPSD 会崩溃

#### Q2: 什么是 Credit Assignment 问题？SDAR 如何解决？

**参考答案**：
- **问题**：序列级奖励无法区分哪些 token 真正决定了任务成功
- **SDAR 方案**：
  1. GRPO 提供任务级优化方向
  2. OPSD 提供 token 级密集监督
  3. 门控机制自适应调节每个 token 的监督强度
- **关键洞察**：teacher 不总是可靠的，需要非对称处理正/负 gap

#### Q3: 解释 GRPO 的优势估计

**参考答案**：
```python
# 组内归一化
A^(i) = (R(x,y^(i)) - mean(R)) / std(R)

# 优点：
# 1. 无需 critic 网络，节省内存
# 2. 组内相对比较，自动适应奖励尺度

# 缺点：
# 1. 序列级信号，所有 token 共享同一 advantage
# 2. 需要足够大的 group size 才能准确估计
```

### 5.2 技术细节类

#### Q4: 为什么 SDAR 要对门控 g_t 做 stop-gradient？

**参考答案**：
```python
# 如果不 detach：
g_t = σ(β * Δ_t)  # Δ_t = log π_T - log π_θ

# 梯度会包含自引用项：
∇_θ L̃_t = -(g_t + β * Δ_t * g_t * (1-g_t)) * ∇_θ log π_θ

# 问题：
# 1. 引入不稳定的自引用耦合
# 2. 梯度可能发散（Δ_t 大时）
# 3. 优化目标变得复杂且不可控

# Detach 后：
∇_θ L_SDAR = - Agg( g_t * ∇_θ log π_θ )
# g_t 仅作为 bounded scalar 权重，梯度稳定
```

#### Q5: 为什么选择 Reverse KL 而不是 Forward KL？

**参考答案**：
```python
# Reverse KL: D_KL(π_θ || π_T)
# - Mode-seeking：student 集中概率到 teacher 支持的模式
# - 对 teacher 低概率 token 自然降权
# - 适合"弱 teacher"场景（teacher 不总是对的）

# Forward KL: D_KL(π_T || π_θ)
# - Mode-covering：student 覆盖 teacher 所有模式
# - 会强制 student 学习 teacher 的错误信号
# - 在多轮 Agent 中容易崩溃

# JSD: 对称折中，但继承了 forward KL 的 mode-covering 缺陷
```

#### Q6: β 参数如何影响训练？

**参考答案**：
```python
# β = 0: 无门控，等价于 naive GRPO+OPSD → 不稳定
# β = 5: 最优值，平滑区分正/负 gap
# β = 10: 过于尖锐，门控接近二值化，丢失细粒度信息

# 直觉：
# β 小 → 所有 token 都被蒸馏（包括噪声）
# β 大 → 只有明确正/负的 token 被处理，边界 token 被忽略
```

#### Q7: Skill Retrieval 的四种策略对比

**参考答案**：
```python
# 1. UCB Retrieval (Upper Confidence Bound)
#    score(e) = r̄(e) + c * sqrt(ln N / n(e))
#    - 探索-利用权衡
#    - 在线更新，自适应最优 skill
#    - 缺点：需要足够训练步数收敛

# 2. Keyword Matching (KM)
#    - 规则匹配 task description 中的关键词
#    - 简单高效，无需训练
#    - 在 WebShop 上甚至优于 UCB

# 3. Full Retrieval
#    - 检索所有 skills
#    - 信息最全，但可能引入噪声

# 4. Random Retrieval
#    - 随机选择 skill
#    - 用于消融实验证明门控机制的有效性
#    - 即使随机 skill，SDAR 仍优于 GRPO baseline
```

### 5.3 系统设计类

#### Q8: 多轮 Rollout 如何实现高效的并行采样？

**参考答案**：
```python
# 1. 向量化环境：batch_size 个环境并行执行
# 2. Active masks：标记已完成环境，避免无效计算
# 3. Padding 处理：pad 到 GPU 数的整除倍数
# 4. 动态采样：过滤全成功/全失败的 group

# 关键代码：
batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, world_size)
batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
```

#### Q9: 如何处理不同长度的轨迹？

**参考答案**：
```python
# 问题：同一 batch 中轨迹长度不同
# 解决方案：
# 1. response_mask 标记有效 token
# 2. 损失聚合时使用 masked_mean
# 3. 优势计算时按 traj_uid 分组

# 代码：
def agg_loss(loss_mat, loss_mask, loss_agg_mode):
    if loss_agg_mode == "token-mean":
        return masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.mean(seq_losses)
```

#### Q10: Teacher 和 Student 共享权重吗？为什么？

**参考答案**：
```python
# 是的，共享 actor 权重
# Teacher 只是"看到更多上下文的同一模型"

# 设计原因：
# 1. 节省内存（无需额外 teacher 模型）
# 2. Self-distillation：从自身提取知识
# 3. 渐进学习：student 逐渐接近 teacher

# 实现：
teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
# teacher_batch 包含 skill context，但模型权重相同
```

### 5.4 实验分析类

#### Q11: 为什么 SDAR 在小模型 (1.7B) 上优势更明显？

**参考答案**：
```python
# 小模型更难利用 retrieved skills
# - Skill-GRPO 在 1.7B 上严重退化 (21.1% vs GRPO 46.1%)
# - 因为小模型无法有效 grounding skills → 有害的分布偏移

# SDAR 的优势：
# 1. 门控机制自动过滤不可靠的 skill 信号
# 2. 仅蒸馏 teacher 更自信的 token
# 3. 避免强制学习无法理解的 skills

# 结果：Qwen3-1.7B 上 SDAR 53.9% vs GRPO 46.1%
```

#### Q12: Gate Active Ratio 的训练动态说明什么？

**参考答案**：
```python
# 训练初期：gate_active_ratio < 0.5
# - 大部分 token 的 gap 为负（teacher 不比 student 更自信）
# - 门控正确抑制这些 token 的蒸馏

# 训练后期：gate_active_ratio 逐渐上升
# - Student 策略改进，更多 token 进入"teacher 有帮助"区间
# - 门控逐步增强蒸馏强度

# 这证明了：
# 1. 自适应课程学习（token 级）
# 2. 非对称处理的必要性
# 3. 门控机制的有效过滤
```

#### Q13: Random Retrieval 仍优于 GRPO 说明什么？

**参考答案**：
```python
# 说明 SDAR 的鲁棒性来自门控机制，而非 skill 质量

# 即使 skill 完全无关：
# 1. 门控会过滤掉大部分负 gap token
# 2. 偶然的正 gap token 仍提供有益信号
# 3. 整体上避免了 naive OPSD 的崩溃

# 实际意义：
# - 不需要完美 skill retrieval
# - 门控机制是 noise-robust 的
# - 降低了 skill 工程的门槛
```

### 5.5 开放讨论类

#### Q14: SDAR 的局限性是什么？如何改进？

**参考答案**：
```python
# 局限性：
# 1. 仍依赖预定义的 skill 库
# 2. 门控仅基于单 token gap，可能忽略上下文
# 3. λ 和 β 需要调参

# 可能改进：
# 1. 动态 skill 生成（而非检索）
# 2. 多尺度门控（token + turn + episode）
# 3. 自适应 λ（根据训练阶段调整）
# 4. 结合 process reward model 提供更细粒度信号
```

#### Q15: 如何将 SDAR 应用到你的研究场景？

**参考答案**：
```python
# 适用场景：
# 1. 多轮对话系统（需要维护上下文一致性）
# 2. 代码生成 Agent（需要工具调用）
# 3. GUI 自动化（需要视觉 grounding）

# 实施步骤：
# 1. 定义环境和奖励函数
# 2. 构建 skill 库（领域知识）
# 3. 集成 SDAR 训练流程
# 4. 调参 λ, β, group_size

# 关键考虑：
# - Skill 质量 vs 门控鲁棒性的权衡
# - 计算开销（teacher 前向传播）
# - 与现有 RL 方法的兼容性
```

---

## 六、项目亮点与技术深度

### 6.1 可讨论的技术亮点

1. **理论分析完备**：
   - Proposition 1-5 证明了门控的数学性质
   - 梯度有界性、单调性、平滑性

2. **工程实现优雅**：
   - 复用 veRL 框架，最小侵入性修改
   - 支持 FSDP/Megatron 分布式策略
   - 灵活的 skill 管理系统

3. **实验设计全面**：
   - 3 个环境 × 3 个模型规模
   - 4 种 skill retrieval 消融
   - 训练动态可视化

4. **代码质量高**：
   - 清晰的模块划分
   - 完整的配置系统
   - 详细的注释和文档

### 6.2 面试中的展示要点

1. **理解深度**：不只停留在公式，能解释设计动机
2. **代码能力**：能指出关键实现细节（如 stop-gradient）
3. **批判思维**：知道局限性和改进方向
4. **工程视角**：理解分布式训练、内存优化等实际问题

---

## 七、快速复习清单

### 核心公式
```
L = L_GRPO + λ * L_SDAR
L_SDAR = Agg( g_t * (log π_T - log π_θ) )
g_t = σ(β * Δ_t).detach()
Δ_t = (log π_T - log π_θ).detach()
```

### 关键数字
- ALFWorld: +9.4% over GRPO (3B)
- Search-QA: +7.0% over GRPO (3B)
- WebShop-Acc: +10.2% over GRPO (7B)
- β = 5.0, λ = 0.01

### 论文引用
```bibtex
@misc{lu2026sdar,
    title={Self-Distilled Agentic Reinforcement Learning},
    author={Zhengxi Lu and Zhiyuan Yao and ...},
    year={2026},
    eprint={2605.15155},
}
```

### 代码入口
- 训练：`verl/trainer/main_sdar.py`
- 损失：`verl/trainer/ppo/sdar_utils.py`
- Rollout：`agent_system/multi_turn_rollout/rollout_loop.py`
- Skills：`verl/trainer/ppo/rlsd_utils.py`

---

## 八、延伸阅读

### 相关论文
1. **GRPO**: DeepSeekMath (Shao et al., 2024)
2. **GiGPO**: Step-level Group Relative Policy Optimization (Feng et al., 2025)
3. **RLSD**: Reinforcement Learning with Self-Distillation (Yang et al., 2026)
4. **SkillRL**: Skill-Augmented Agents (Xia et al., 2026)
5. **Search-R1**: Search-augmented QA (Jin et al., 2025)

### 技术栈
- **veRL**: Volcengine RL framework
- **vLLM**: 高效推理引擎
- **Ray**: 分布式计算框架
- **FSDP**: 全分片数据并行

---

*文档生成时间：2026-06-27*
*基于 SDAR 项目 commit: 最新版本*
