# SDAR 深入学习文档

- 阅读日期：2026-10-14
- 上游基准：`ZJU-REAL/SDAR`，默认分支 `master`，HEAD `29dcb12a`（2026-10-06，`Add recent works powered by SDAR to README`）
- Fork 状态：`jasonchen505/SDAR` 的 `master` 为 `aa183362`，即上游 `9f2ce6a8`（2026-06-23）+ 1 个个人笔记提交；相对上游 HEAD 落后 41 个提交
- 本文档所在分支：`notes/learning`（从 fork `master` 建出，个人学习笔记，不进上游 PR）
- 阅读方式：只读 API 逐文件核对源码 + arXiv 摘要页；未做 GPU 训练，未跑通端到端复现
- 约束提醒：个人笔记绝不进上游 PR diff；未来贡献分支必须从上游 `master` 建

---

## TL;DR

- SDAR = Self-Distilled Agentic RL。核心只有一句话：**RL 仍是主优化目标，OPSD 被降格为一个带门控的 token 级辅助损失**。教师不是外部大模型，而是同一个策略模型在“技能提示词”加持下的前向（self-teacher）。
- 门控是全文的关键：`g = sigmoid(beta * (logp_teacher - logp_student))`，beta 默认 5.0。教师更自信的 token 蒸馏权重趋近 1，教师被学生反超的 token 权重被压向 0。梯度只流向学生，教师和门控全部 detach。
- 仓库不是“一个算法”，而是**一族方法 + 三个环境 + 一套以 verl 为骨架的 vendored 训练栈**：SDAR、SkillSD、RLSD、OPD、GRPO+OPD、GRPO+OPD+RetireOPD、GRPO、DAPO、GiGPO 的入口和脚本都在 `examples/` / `verl/trainer/` 里。读代码时必须先分清当前看的是哪条训练器链路，否则会把不同教师来源混为一谈。
- 最容易踩的坑：`examples/opsd_trainer/` 与 `examples/grpo_opsd_trainer/` 这两个脚本**并不**走独立教师模型的 OPD/GRPO+OPD 训练器，而是复用 `main_sdar`，分别用 `pg_loss_coef=0` 和 `gate_beta=0` 模拟。真正的独立教师 OPD 入口是 `main_opd_trainer` / `main_grpo_opd_trainer`。这是上游 issue #61 正在质疑的点，截至 2026-10-14 仍 open、无 maintainer 回复。
- 论文宣称的数字（相对 GRPO：ALFWorld +9.4%，Search-QA +7.0%，WebShop-Acc +10.2%）是论文/ README 口径。本人未复现；仓库 issue 区有多个独立复现报告结果低于论文或与 GRPO 持平（#57、#59、#58、#43 等），复现可信度应打折扣，见“诚实结论”。
- 已有贡献：PR #62（`Weight validation success rates by sample count`，fixes #60）仍 open，0 评论 0 review，0 check runs。它只修了 #60 的第一半（按样本数加权），第二半（`test_score` 按 step 展开行平均）未修。

---

## 一句话定位

SDAR 把“技能提示词加持下的同模型教师”产生的 token 级信号，用一个基于教师-学生 logp gap 的 sigmoid 门控筛过之后，作为 GRPO 之外的辅助损失加进 Agentic RL；目标是给长程多轮 agent 补一个稠密的 token 级监督，同时不让不可靠的教师信号把 RL 训练带偏。

---

## 1. 现状核查（2026-10-14 实测）

### 1.1 上游

| 项目 | 实测值 |
|---|---|
| 仓库 | `ZJU-REAL/SDAR`，`Official code for "Self-Distilled Agentic Reinforcement Learning"` |
| Stars | **366**（API 实测；任务背景里说的约 398 与本次实测不符，可能为不同时刻计数或缓存，本文以实测为准） |
| Forks | 23 |
| 默认分支 | `master`（不是 `main`） |
| master HEAD | `29dcb12a`，2026-10-06T09:23:56Z，`Add recent works powered by SDAR to README` |
| open_issues_count（API 字段） | 9（= 3 个 issue + 6 个 open PR，含 5 个 dependabot PR；不是 9 个 issue） |
| 真实 open issue | 3 个：#59、#60、#61 |
| 真实 open PR | 6 个：#62（本人的）+ 5 个 dependabot 依赖升级（#48、#49、#50、#54、#55） |
| 最近一次 master 提交 | 2026-10-06。2026-10-14 当天在 master 提交 / issue / PR 层面查不到新活动 |
| repo `updated_at` 字段 | 2026-10-14T07:39:55Z，但无法对应到任何可见的提交/issue/PR 变更；**不要把“今天更新”理解为今天有代码提交** |

关于“今天更新”的判断：上游 master 自 2026-10-06 之后无新提交；2026-10-02 到 10-06 的 4 个提交全部是 README 层面的（加 RetireOPD/PR-OPD/UniOPSD 项目、引用格式）。也就是说，**近期的“更新”全是 README 生态位宣传，没有算法代码变更**。上游真正的代码开发在 2026-09-17（RetireOPD 训练代码落地）之后基本停滞。

### 1.2 Fork

| 分支 | HEAD | 说明 |
|---|---|---|
| `master` | `aa183362`（2026-06-27，`Add interview prep notes and paper references`） | 上游 `9f2ce6a8` + 1 个个人笔记提交；落后上游 HEAD **41** 个提交；绝不动 |
| `contrib/60-weighted-val-success-rate` | `49c5278c` | PR #62 的分支，基底已是上游 `master` 最新 `29dcb12a`，含 2 个提交（见 1.3） |
| `notes/learning` | 建于本次学习（基底 fork `master` `aa183362`） | 本文档所在分支 |

### 1.3 PR #62 最新状态（2026-10-14 实测）

- 状态：**open，未合并**，`mergeable_state: unstable`，`merged: false`
- Head：`jasonchen505:contrib/60-weighted-val-success-rate` @ `49c5278c`；Base：上游 `master` @ `29dcb12a`
- 规模：2 个提交，2 个文件，+66 / -1
- 评论：`get_comments` 返回空；Review：`get_reviews` 返回空；Checks：commit status 查询返回 `state: pending`，`total_count: 0`，`statuses: []`，即 **0 个 check runs**
- 两个提交的 Signed-off-by 均为 `Jason Chen <90548657+jasonchen505@users.noreply.github.com>`（noreply 邮箱；这是 PR #62 提交时的既有状态，记录于此）
- 结论：自 2026-10-07 提交后，maintainer 无任何互动。结合 issue #59/#60/#61 同样无人回复，maintainer 近期对外部 issue/PR 的响应基本停滞。

---

## 2. 论文速览

### 2.1 SDAR 主论文

- 标题：Self-Distilled Agentic Reinforcement Learning
- arXiv：`arXiv:2605.15155`，提交于 2026-05-14，https://arxiv.org/abs/2605.15155
- 作者（arXiv 页顺序）：Zhengxi Lu、Zhiyuan Yao、Zhuowen Han、Zi-Han Wang、Jinyang Wu、Qi Gu、Xunliang Cai、Weiming Lu、Jun Xiao、Yueting Zhuang、Yongliang Shen
- 组织：仓库归属 `ZJU-REAL`（浙大 REAL 组）；各作者具体单位未在摘要页逐一核对，本文不展开
- 核心 claim（意译，非引用）：纯 RL 的轨迹级奖励对长程交互监督太稀疏；OPSD 能补 token 级稠密监督，但直接搬到多轮 agent 上会出两个问题，一是多轮不稳定性放大监督噪声，二是技能条件特权指导对负样本（教师拒绝）需要不对称处理，因为拒绝可能源于技能检索/利用不完善而非学生真的错了。SDAR 的解法是把 OPSD 做成带门控的辅助目标，RL 仍是主目标；门控把 detached 的 token 级信号映射成 sigmoid 权重，对教师认可的正 gap token 加强蒸馏，对教师负向拒绝做软衰减。论文报告在 Qwen2.5 / Qwen3 系列上，ALFWorld、WebShop、Search-QA 均显著优于 GRPO，且避免了 naive GRPO+OPSD 的不稳定。
- 可信度判断：方法描述与代码实现能对上（门控公式、detach、GRPO 主损失都在代码里）；**效果数字的可信度低于方法本身**，理由见第 7 节（多个独立复现报告不达标；验证指标本身在 #60/#61 中被发现有聚合缺陷）。

### 2.2 RetireOPD（同仓库的第二篇工作）

- 标题：RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning
- arXiv：`arXiv:2609.20784`，v1 2026-09-17，v2 2026-09-27，https://arxiv.org/abs/2609.20784
- 作者（arXiv 页顺序）：Yan Yu、Zhengxi Lu、Yizhou Liu、Yichen Pan、Aozhe Wang、Qipeng Chen、Hua Yang、Wenqi Zhang、Qianglong Chen、Yongliang Shen
- 核心 claim（意译）：OPD 的教师监督收益是阶段依赖的，且特权信息本身不保证教师可靠。RetireOPD 先用环境奖励单独训一个 skill-conditioned 教师，再用 RL+OPD 训一个无技能学生；不按预设 schedule 退场，而是当师生差距停止缩小、且学生达到教师成功率的目标比例时，学生自行“退休”教师，之后只用 RL。论文报告在 Qwen2.5 1.5B–7B 上，ALFWorld 成功率相对 RL 基线提升 14.1%–18.8%，WebShop 准确率提升 11.8%–19.0%，且在所有设置下超过其自身的 skill-conditioned 教师。
- README 里的 RetireOPD 结果表（README 口径，未经本人验证）：ALFWorld Qwen2.5-1.5B/3B/7B 为 89.8% / 93.8% / 95.3%；WebShop 为 75.8% / 77.3% / 84.4%。
- 与 SDAR 的关系：RetireOPD 不是 SDAR 的门控损失，而是 GRPO+OPD（SDL 路径）+ 退休调度器；教师是**独立训练出的冻结模型**，不是 SDAR 的同模型自教师。两者共享技能/教师 batch 构造等基础设施，但教师来源和退场机制完全不同。

### 2.3 生态位：SDAR 已成为一个“派生工作集线器”

README 的 `Awesome Work Powered by SDAR` 列出 21 个后续工作（按时间倒序），包括 AlignOPSD、PR-OPD、UniOPSD、RetireOPD、TASPO、AHEAD、ICSD、BCSD、AgentOPSD、OCSD、ADRS、PCSD、GRSD、OVCSD、MAPD、SEED、UCOB、CRAFT、ATOD、OPID、StepOPSD。README 自称 SDAR 是“第一个把 Agentic RL 与 OP(S)D 统一起来的开源框架”。这个定位基本属实：它确实提供了一套可复用的教师 batch 构造 + 技能 + 多训练器骨架，后续工作大多是在这套骨架上改门控/调度/信用分配。但也要注意，**这 21 个工作绝大多数不是本仓库的代码**，只是引用关系；不要把生态列表误读为本仓库的功能清单。

---

## 3. 方法拆解

### 3.1 问题设定

Agent 在多轮环境交互中产生一条轨迹，环境只在轨迹结束时给一个标量奖励（成功/失败）。GRPO 这类方法把序列级 advantage 广播到所有 token，监督信号稀疏，且无法区分“哪一步真正错了”。OPSD 的思路是：给同一个学生一个“特权版本”的输入（加上技能提示词），用特权教师的 token 分布提供稠密监督。

### 3.2 SDAR 的教师/学生到底是什么

关键事实（代码证实，不是比喻）：

- **学生**：当前策略模型，输入为普通 prompt（无技能），`logp_S = log pi_theta(y_t | x, y_<t)`，带梯度。
- **教师**：**同一个策略模型**，输入为“技能提示词 + 普通 prompt”，`logp_T = log pi_theta(y_t | x + r, y_<t)`，冻结、无梯度。`r` 是技能文本（见 3.4）。
- “Self-Distilled” 的 self 就在这里：蒸馏源不是更大的外部教师，而是**同模型在特权输入下的自己**。这既是方法的优雅之处（零额外教师训练成本），也是它的根本风险（教师和学生共享同一个模型的缺陷，特权输入不总能带来可靠提升——RetireOPD 论文的出发点正是这一点）。

教师 logp 的产生链路（`verl/trainer/ppo/rlsd_ray_trainer.py`）：

```
普通 batch
  -> build_teacher_batch()（L39）：按 gamefile / data_source / prompt 关键词取技能文本，
     以 "[Privileged Skill Information]\n<技能>\n\n" 为前缀拼到 prompt 前，重新 tokenize
  -> 同一个 actor 的 compute_log_prob() 对教师 batch 做前向（L422-439）
  -> 得到 teacher_log_probs，塞进 batch.batch["teacher_log_probs"]
  -> dp_actor 在计算 policy loss 时取出来用
```

### 3.3 门控损失（SDAR 唯一的算法增量）

`verl/trainer/ppo/sdar_utils.py`，`compute_sdar_loss()`（L14 起）：

```
delta_t = logp_T - stopgrad(logp_S)          # L44，教师-学生 gap
g_t     = stopgrad(sigmoid(beta * delta_t))  # L46，门控，beta 默认 5.0
L_SDAR  = agg(g_t * (stopgrad(logp_T) - logp_S))  # L52，经 agg_loss 按 response_mask 聚合
```

机制逐条读：

- `delta_t > 0`（教师更认可当前 token）：`g_t -> 1`，蒸馏拉满，学生被推向教师分布。
- `delta_t < 0`（教师反而更不认可当前 token）：`g_t -> 0`，蒸馏被软衰减。论文称之为对“教师负向拒绝”的不对称处理：拒绝可能源于技能检索/利用不完善，不应像正样本那样硬惩罚学生。
- `delta_t = 0`：`g_t = 0.5`，中等权重。
- 梯度：`dL/d(logp_S) = -g_t`（在 token-mean 聚合下，忽略聚合常数）。所以门控本质上是**逐 token 的学习率调节器**，而不是一个 KL 距离。这一点和 SkillSD 的 K3 式 SDL、RLSD 的 advantage 重加权都不同。
- beta 的作用：beta 越大，sigmoid 越陡，门控越接近 0/1 硬开关；beta = 0 时门控恒为 0.5，教师信息只剩一个与 `logp_T` 无关的常数梯度（见 5.3，issue #61 的核心论证）。
- 上报指标：`sdar/gate_mean`、`sdar/gate_active_ratio`（`g > 0.5` 的 token 比例）、`sdar/teacher_gap_mean`、`sdar/loss`。

### 3.4 技能（特权信息）从哪来

`skills/` 下按环境分三个目录：`alfworld`、`search`、`webshop`，每个目录一个 `skill_mapping.json` + 若干技能 md 文件。`verl/trainer/ppo/rlsd_utils.py` 的 `SkillProvider` 是唯一入口：

- `load_skill_mapping` / `load_skill_content`：读 `skill_mapping.json` 的 `skill_files`，把 md 全文读进内存字典。
- `get_privileged_info(gamefile)`：任务类型作为子串出现在 gamefile 路径里时取对应技能；ALFWorld 用这条。
- `get_privileged_info_from_data_source(data_source, prompt)`：Search 用这条，含硬编码规则：`popqa -> entity_attribute_lookup`，`nq/triviaqa -> direct_retrieval`，`hotpotqa -> multi_hop_reasoning`，`2wikimultihopqa/musique/bamboogle -> multi_hop_reasoning`，prompt 含 `which` + `or`（且无 `for`）时判为 `compare`。
- `get_privileged_info_from_prompt(prompt)`：关键词回退路径。
- `skill_all=True` 时直接拼全部技能（general + 全部任务技能），跳过匹配。

技能匹配是规则系统，不是检索模型。ALFWorld 的 `skill_mapping.json` 里 `task_keywords` 有 6 类任务；WebShop 有 7 类商品类别关键词表（`shirt/dress/...`、`shoe/boot/...` 等）。规则简单，意味着**技能检索错误是常态而非例外**，这正是论文用“软衰减”而非“硬过滤”处理负拒绝的工程理由，也是 #35（ALFWorld 技能加载不全导致性能低于论文，2026-06-23 已修）这类 bug 能直接打穿方法效果的原因。

### 3.5 总损失构成

`verl/workers/actor/dp_actor.py`，L412–458（`compute_log_prob` 后的 policy loss 组装）：

```
policy_loss = pg_loss * pg_loss_coef          # GRPO/PPO 主损失，默认 pg_loss_coef=1
            - entropy * entropy_coeff          # 可选熵项
            + kl_loss * kl_loss_coef            # 可选 KL（SDAR 脚本里 kl_loss_coef=0.01/0.001）
            + sdl_loss * sdl_coef               # SkillSD 路径（use_sdl_loss 时）
            + sdar_loss * sdar_coef             # SDAR 路径（use_sdar_loss 时，L445-458）
```

SDAR 脚本的实际取值（`examples/sdar_trainer/run_*_3b.sh`）：`sdar_coef=0.01`，`gate_beta=5.0`。注意代码默认值与脚本不一致：`main_sdar.py` 里 `sdar_coef` 默认 0.1，`dp_actor.py` 里 `sdar_loss_coef` 默认 0.1、`sdar_gate_beta` 默认 5.0。**读代码时以脚本为准，默认 0.1 是个容易误读的陷阱**——0.01 与 0.1 相差一个数量级，辅助损失的权重直接决定教师信号是“轻推”还是“主导”。

---

## 4. 代码地图

### 4.1 顶层结构

```
SDAR/
├── verl/                  # vendored 的 verl（不是 pip 依赖）：训练器、worker、utils 全在仓库内
│   ├── trainer/           # 各方法入口 main_*.py + ppo/ 核心
│   ├── workers/           # actor/critic/rollout worker（含 dp_actor.py）
│   └── ...                # 其余 verl 模块（dataset、utils、single_controller 等）
├── agent_system/          # agent 侧：环境、多轮 rollout、奖励
│   ├── environments/      # env_manager.py（三种环境的 _process_batch）+ env_package（webshop/search 等）
│   ├── multi_turn_rollout/ # rollout_loop.py（TrajectoryCollector）
│   └── reward_manager/    # episode.py（EpisodeRewardManager）
├── skills/                # 特权技能：alfworld / search / webshop
├── examples/              # 各方法的启动脚本（按训练器分子目录）+ 数据预处理 + 检索服务
├── tests/                 # verl 继承的测试树（tests/trainer/ 等）
├── scripts/               # model_merger.py 等
├── recipe/ gigpo/ docs/ docker/  # 附加内容（recipe/gigpo 未深挖）
├── pyproject.toml / setup.py    # 包名 sdar；版本读 verl/version/version
└── .github/workflows/    # checkpoint_converter / dataset / doc / e2e_dapo 等；无针对 PR 的通用 CI
```

关键结构判断：**verl 是 vendored，不是依赖**。`pip install -e .` 装的是仓库内这份 verl，`setup.py` 的 `install_requires`（ray 2.41–2.50、transformers ≤4.57.3、vllm 0.8.5–0.11.0 等）是 verl 继承来的依赖表。这意味着：第一，改 `verl/` 下任何文件都会影响全部方法；第二，上游 verl 的演进不会自动同步进来，这份 verl 是 2026-05-15 前后冻结的快照；第三，贡献时要分清“SDAR 新增代码”与“verl 原有代码”，后者的修改需要更谨慎的 blast-radius 判断。

### 4.2 训练器入口矩阵（必须分清的六条链路）

| 入口 | 训练器 | 教师来源 | 辅助损失 | 用途 |
|---|---|---|---|---|
| `verl/trainer/main_sdar.py` | `SkillSDRayTrainer`（继承 `RLSDRayTrainer`） | 同模型自教师（skill-augmented prompt 前向） | 门控 SDAR loss（`compute_sdar_loss`）+ GRPO 主损失 | **SDAR 本体**；`examples/sdar_trainer/` |
| `verl/trainer/main_opd_trainer.py` | `OPDTrainer` | 独立冻结教师模型（`teacher_model_path`，`Role.TeacherPolicy`） | K3-IS 式 OPD（`use_sdl_loss=False`，走 opd 路径） | 纯 OPD |
| `verl/trainer/main_grpo_opd_trainer.py` | `GRPOOPDTrainer` | 独立冻结教师模型 | SDL loss（`use_sdl_loss=True`，固定系数）+ GRPO | GRPO+OPD（独立教师版） |
| `verl/trainer/main_retire_opd_trainer.py` | `RetireOPDTrainer`（继承 `GRPOOPDTrainer`） | 独立冻结教师模型，可退休 | SDL loss，系数由退休调度器控制，退休后归零并跳过教师前向 | RetireOPD |
| `verl/trainer/main_rlsd.py` | RLSD 训练器 | 同模型自教师 | 无辅助损失；教师 gap 调制 token 级 advantage | RLSD |
| `verl/trainer/main_skillsd.py` | `SkillSDRayTrainer` | 同模型自教师 | SDL loss（K3 式，`compute_sdl_loss`） | SkillSD |

以及两个**容易误认**的脚本目录：

| 脚本目录 | 实际入口 | 实际行为 |
|---|---|---|
| `examples/opsd_trainer/` | `main_sdar`，`pg_loss_coef=0`，`gate_beta=5.0` | SDAR 门控损失单独跑（无 GRPO 主损失），不是 `main_opd_trainer` 的纯 OPD |
| `examples/grpo_opsd_trainer/` | `main_sdar`，`gate_beta=0.0` | SDAR 损失 + GRPO，但门控恒 0.5（见 5.3） |

### 4.3 训练 / 验证数据链路

```
数据（parquet）
  -> create_rl_dataset / create_rl_sampler（main_ppo.py）
  -> TrajectoryCollector.multi_turn_loop()（rollout_loop.py）
     -> envs（EnvironmentManagerBase）：每步 agent 动作 -> 环境反馈
     -> 轨迹结束：env_manager._process_batch() 产出每条轨迹的 won / success_rate
     -> gather_rollout_data()（L246）：把 episode_rewards / success_rate 复制到每一步的行上
  -> RayPPOTrainer.fit()：rollout -> teacher 前向 -> advantage -> dp_actor 更新
  -> _validate()（ray_trainer.py L691）：验证集多轮 rollout -> val_reward_fn -> 指标聚合
```

注意 `gather_rollout_data` 的复制语义（L275–292）：`episode_rewards[bs]` 被写到该轨迹**每一步**的行上。这就是 #60 第二半问题（`test_score` 按 step 展开行平均）的源头：验证聚合时若不按 `traj_uid` 去重，长轨迹的权重会被放大。训练侧 `EpisodeRewardManager` 默认 `normalize_by_length=False`，同样依赖调用方正确处理。

---

## 5. 关键机制深挖

### 5.1 门控损失：SDAR 的唯一算法增量（已在 3.3 展开）

定位：`verl/trainer/ppo/sdar_utils.py:14-67`，`compute_sdar_loss`；调用点 `verl/workers/actor/dp_actor.py:445-458`；开关 `verl/trainer/main_sdar.py`（`use_sdar_loss=True`，`sdar_loss_coef` / `sdar_gate_beta` 由 `algorithm.sdar` 配置注入）。

值得深挖的一点：门控的 detach 位置。`delta_t` 用了 `student_log_probs.detach()`，`gate` 整体 detach，`teacher_log_probs` detach。这意味着**教师信号只通过 `g_t` 这个标量权重和 `logp_T` 这个常数目标影响梯度**，不存在任何让学生“反向影响教师”的通道。教师和学生是同一组参数，但前向是两次独立的 `compute_log_prob`，参数共享只体现在优化器更新上。这就是 self-distillation 在实现上不塌缩的原因。

### 5.2 教师 batch 构造：左截断会切掉技能前缀

`verl/trainer/ppo/rlsd_ray_trainer.py:39-161`，`build_teacher_batch()`。

流程：解码学生 prompt 文本 → 取技能文本 → `skill_prefix + prompt_text` 拼成教师 prompt → 重新 tokenize → 拼 response → 计算 position_ids。

关键细节在 L117–119：

```python
# Truncate if needed (left truncation to keep the end of prompt)
if len(teacher_prompt_ids) > max_prompt_length:
    teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]
```

技能前缀在教师 prompt 的**最前面**，左截断保留的是**末尾**的 token。也就是说，当 `技能 + prompt` 超长时，被切掉的正是技能前缀本身——教师退化为普通学生，`delta_t` 趋近 0，门控趋近 0.5，SDAR 损失退化为一个常数权重的 logp 拉升。代码注释写的是“keep the end of prompt”，但没有处理“技能前缀被切”的后果，也没有上报截断发生频率的指标。SDAR 脚本里 ALFWorld 的 `max_prompt_length=2048`、Search 的 4096，技能 md 通常不长，实践中可能很少触发；但这是一个**未被度量的静默退化路径**，值得在复现时加一个截断计数器验证。GRPO+OPD 的 `GRPOOPDTrainer._compute_teacher_log_probs` 同样调用 `build_teacher_batch`，同一问题同样适用。

### 5.3 `gate_beta=0` 的 GRPO+OPSD 脚本：教师无关梯度（issue #61）

`examples/grpo_opsd_trainer/run_webshop_3b.sh`（ALFWorld/Search 同理）设置 `gate_beta=0.0`。代入 3.3 的公式：

```
g = sigmoid(0 * delta) = 0.5（对所有有效 token）
L_aux = masked_mean(0.5 * (stopgrad(logp_T) - logp_S))
dL_aux/d(logp_S) = -0.5（与 logp_T 无关）
```

即辅助项的**梯度**与教师完全无关；改技能、改教师概率只能改变上报的标量 loss 值，不能改变这次更新的方向。独立的 reference-model KL 正则化不改变这个结论（它约束的是学生与 reference 的距离，不是教师信号）。#61 的论证在数学上是成立的（本人核对了 `compute_sdar_loss` 的 detach 位置与公式，与 #61 的推导一致）。

同理，`examples/opsd_trainer/`（`pg_loss_coef=0`，`gate_beta=5.0`）是“SDAR 门控目标去掉 GRPO 主损失”，教师通过门控权重 `g` 影响更新，所以 #61 明确表示**不**主张这个运行忽略教师；它质疑的是：这个运行是“SDAR 的消融”，而不是论文附录 B 算法 3 描述的 plain OPSD。Table 1 的 OPSD 对比是否由这些脚本产生，截至 2026-10-14 仍是未回答的问题。

### 5.4 `_validate()` 的两处聚合缺陷与 PR #62 的位置（issue #60）

`verl/trainer/ppo/ray_trainer.py:691-828`，`RayPPOTrainer._validate()`。

缺陷一（PR #62 已修）：L778–793。每批验证 batch 产生一个 success rate，存入 `success_rate_dict[k]`，最后 `np.mean(v)` 等权平均。尾巴 batch 只有 1 个 Bamboogle 问题时，它占 50% 权重。PR #62 的修复是记录每批样本数，用 `np.average(..., weights=sample_counts)` 加权（新辅助函数 `_weighted_mean_success_rate`，测试 `tests/trainer/ppo/test_success_rate_aggregation.py`，4 个测试）。

缺陷二（PR #62 **未修**，仍在代码里）：L795–818。`val/{data_source}/test_score` 对 `reward_tensor` 按行平均，而 `gather_rollout_data` 已把 episode reward 复制到每一步的行上。`tool_call_count` 在 L806 用了 `np.unique(traj_uids)` 去重，但 reward 聚合没有。#60 给出的例子：两条轨迹 correctness `[1, 0]`、长度 `[1, 4]` 时 `test_score = 0.2`，长度 `[4, 1]` 时为 `0.8`，而问题级准确率都是 0.5。**比较“用了不同搜索步数的策略”时，这个指标会系统性偏向短轨迹策略**。PR #62 的标题和正文只承诺修复缺陷一；缺陷二仍是 open 的工作。

另注意 L784–785 的 assert：它要求同一 batch 内所有行的 success rate 相同。这个 assert 在验证 batch 内部是合理的（同一 batch 的 success_rate 是 `gather_rollout_data` 里按 batch 平均后复制的），但它掩盖了跨 batch 的权重问题——assert 通过不代表聚合正确。

### 5.5 RetireOPD 调度器：退休条件比看起来严格

`verl/trainer/ppo/retire_opd.py`，`RetireOPDScheduler`；训练器 `verl/trainer/ppo/retire_opd_trainer.py:31-81`。

- `observe_gap(step, gap_mean)`：把 `k_step = -gap_mean` 缓存进窗口，窗口大小默认 5（`window_size`）。
- `observe_validation(step, student_performance)`：取下一个已完成的窗口，算两个量：
  - `alignment_progress = (prev_window_mean - current_window_mean) / prev_window_mean`（分母接近 0 时跳过，`denominator_epsilon=1e-8`）
  - `competence_ratio = (performance + prev_performance) / (2 * teacher_performance)`
- 退休条件（三个同时成立）：`alignment_progress <= alignment_threshold`（默认 0.0，即对齐无进展甚至倒退）**且** `competence_ratio >= competence_threshold`（默认 0.9）**且** `criterion_valid`（需要至少两个窗口 + 两次验证）。
- 退休后：`opd_coef = 0.0`；教师前向被跳过，返回 `old_log_probs` 的 placeholder（`retire_opd_trainer.py:52-56`），并上报 `retire_opd/teacher_forward_skipped=1.0`；退休状态随 checkpoint 保存/恢复（`retire_opd_state.json`，`load_state_dict` 要求 config 完全匹配，否则报错）。
- 值得注意的耦合：`main_retire_opd_trainer.py` 要求 `trainer.test_freq <= retirement_window_size`，因为退休判断依赖验证指标来推进窗口。`run_alfworld_3b.sh` 里 `test_freq = retirement_window_size = 5`，正好卡在边界上。
- 潜在脆弱点（观察，未验证）：`competence_ratio` 用了**两次验证性能的平均**除以教师性能；如果验证指标本身有 #60 式的聚合偏差，退休时机会被带偏。以及 `TEACHER_PERFORMANCE` 是用户手动传入的标量（脚本要求 `TEACHER_PERFORMANCE=0.50` 这样的形式），填错直接改变退休阈值，没有任何代码侧校验能发现。

### 5.6 SkillSD 与 RLSD：同一教师，三种用法

三者共享 `build_teacher_batch` + `SkillProvider`，区别只在怎么用 `teacher_log_probs`：

| 方法 | 教师信号用法 | 代码位置 |
|---|---|---|
| SDAR | 门控损失 `agg(g * (logp_T - logp_S))`，GRPO 主损失保留 | `sdar_utils.py:14`，`dp_actor.py:445` |
| SkillSD | K3 式 SDL：`k3 = exp(clamp(-ell)) - 1 + ell`，其中 `ell = logp_S - logp_T`，再乘 on-policy 比率 `rho_on = exp(clamp(logp_S - logp_old))` | `skillsd_utils.py:compute_sdl_loss`，`dp_actor.py:431` |
| RLSD | 无辅助损失；教师 gap 调制 token 级 advantage：`A_t = A_seq * ((1-lambda) + lambda * clamp(exp(sign(A_seq) * delta_t)))` | `rlsd_utils.py:compute_rlsd_token_advantage` |

三者的对比是理解 SDAR 设计取舍的最佳参照：RLSD 把教师信号放在 advantage 里（影响 PG 梯度方向），SkillSD 把教师信号放在 K3 损失里（无门控，教师错误会全额传递），SDAR 把教师信号放在门控损失里（教师错误被门控衰减）。SDAR 的论文叙事——“RL 为主、OPSD 为辅、门控防教师错误”——正好对应这张表。

### 5.7 SkillProvider 的文档与实现不一致（观察）

`rlsd_utils.py` 中 `SkillProvider` 类的 docstring 称关键词匹配是“ALL keywords appear 才命中，第一个匹配的任务类型获胜”。但 `get_privileged_info_from_prompt` 的实现是 `any(kw in text_lower)`，且多个任务类型匹配时会**拼接全部**对应技能文本。文档说 ALL + 首个获胜，实现做 ANY + 全部拼接。这是文档/实现不一致，至少应修文档。影响评估：ANY 匹配会让教师 prompt 更容易混入无关技能（例如 ALFWorld 里 `put` 这个关键词会匹配 `pick_and_place`，而几乎所有取放任务都含 `put`），在 `skill_all=False` 的默认配置下仍可能发生。

---

## 6. 与 verl / GRPO 的关系

- **SDAR 对 verl 的关系是“vendored + 扩展”，不是“依赖”**。`verl/` 目录是仓库内的一份 verl 快照，`SkillSDRayTrainer` 继承 `RLSDRayTrainer`，`RLSDRayTrainer` 继承 `RayPPOTrainer`。SDAR 新增的训练器代码集中在 `verl/trainer/main_sdar.py`、`verl/trainer/ppo/sdar_utils.py`、`verl/trainer/ppo/skillsd_ray_trainer.py`、`verl/trainer/ppo/rlsd_ray_trainer.py`、`verl/trainer/ppo/rlsd_utils.py`、`verl/trainer/ppo/skillsd_utils.py`、`verl/trainer/ppo/retire_opd*.py`、`verl/trainer/ppo/opd_trainer.py`、`verl/trainer/ppo/grpo_opd_trainer.py`。
- **与 GRPO 的关系**：GRPO 是 SDAR 的主损失（`algorithm.adv_estimator=grpo`，序列级 advantage 广播到 token）。SDAR 没有改 GRPO 本身，只是在它外面加了一个 token 级门控辅助损失。`pg_loss_coef` 是 GRPO 主损失的开关：SDAR 脚本里为 1（默认），`opsd_trainer` 脚本里为 0。
- **与 DAPO / ProRL 的关系**：DAPO 在仓库里只是一个 `examples/dapo_trainer/` 目录（继承 verl 的 DAPO 训练器），ProRL 在仓库里没有对应实现。SDAR 与 DAPO/ProRL 是**同一代 RL 方法的不同分支**，不是上下游：DAPO 解决的是长 CoT 的 Clip-Higher / 动态采样 / token 级 PG 损失问题，ProRL 解决的是长程 rollout 的截断/恢复与奖励塑形问题，SDAR 解决的是“给 agent 补 token 级教师监督”的问题。三者正交，可以组合，但本仓库没有提供 SDAR+DAPO 的联合配置。
- **与 GiGPO 的关系**：`examples/gigpo_trainer/`、`gigpo/` 目录存在（GiGPO 是同期另一种 agent RL 方法），但本次未深挖其代码，不展开。

---

## 7. 复现与门槛

### 7.1 最小跑通命令（README 口径，未经本人执行）

```bash
conda create -n sdar python==3.12 -y && conda activate sdar
pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
bash examples/sdar_trainer/run_alfworld_3b.sh
```

ALFWorld 还需 `pip install alfworld` + `alfworld-download -f`；WebShop 需单独的 Python 3.10 环境（`conda create -n verl-webshop python==3.10`）+ `./setup.sh -d all`；Search 需单独的 retriever 环境（faiss-gpu 1.8.0、约 6GB GPU 显存/GPU）+ 下载 e5 索引并启动 `retrieval_launch.sh`。

### 7.2 资源门槛（脚本口径）

| 任务 | 脚本 GPU 需求 | 关键超参 | 训练规模 |
|---|---|---|---|
| ALFWorld 3B | `trainer.n_gpus_per_node=8`，`tensor_model_parallel_size=2` | lr 1e-6，`sdar_coef=0.01`，`gate_beta=5.0`，`group_size=8`，`max_steps=50` | `total_epochs=150`，`test_freq=5` |
| WebShop 3B | `n_gpus_per_node=2`，`tensor_model_parallel_size=2` | 同上，KL 系数 0.01 | `total_epochs=150`，`test_freq=5` |
| Search 3B | `n_gpus_per_node=4`，`tensor_model_parallel_size=1` | KL 系数 0.001，`max_steps=4`，`history_length=4` | `total_training_steps=150`，`test_freq=150` |

另有 7B 与 Qwen3 变体脚本（`run_*_7b.sh`、`run_*_qwen3.sh`）。按脚本口径，单卡复现不可行；4–8 卡是起步。Search 还要额外为检索服务留显存。

### 7.3 诚实结论：已验证 vs 未验证

**已验证（本人逐文件核对）：**

- 门控公式、detach 位置、总损失构成、教师 batch 构造、技能匹配规则、六条训练器链路的教师来源区别、`_validate` 的两处聚合缺陷、RetireOPD 调度器逻辑、PR #62 的状态与范围、fork/upstream 的分支状态。
- #61 关于 `gate_beta=0` 时辅助梯度与教师无关的推导，与 `compute_sdar_loss` 的实现一致。
- #60 缺陷一的数学例子（124+1 Bamboogle）与 `_validate` 的 `np.mean` 实现一致。

**论文宣称、未在代码里验证：**

- 全部效果数字（SDAR 相对 GRPO 的 +9.4% / +7.0% / +10.2%；RetireOPD 的 14.1%–18.8% / 11.8%–19.0%）。仓库没有提供论文的完整实验配置与 checkpoint，README 的结果表没有随机种子、评估次数、方差。
- “避免 naive GRPO+OPSD 的不稳定性”——代码里没有 naive GRPO+OPSD 的对照实现（`grpo_opsd_trainer` 脚本是 `gate_beta=0` 的 SDAR，不是论文里的 naive 基线），无法从代码侧确认对照的公平性。
- 技能 md 文件的内容质量（技能文本本身是否正确、是否覆盖任务）未逐条评审；技能是手写规则文本，其质量直接决定教师信号质量。

**复现风险（issue 区实证）：**

- #59（open）：Search 上 Table 1 复现失败。
- #58（closed）：固定验证下 SDAR 在 ALFWorld 上增益小于论文，且在 WebShop 3B 上反转落后于 GRPO。
- #57（closed）、#43（closed）：多位用户报告 ALFWorld / WebShop 上 GRPO 与 SDAR 均低于论文，或 GRPO 反超 SDAR。
- #52（closed）：WebShop 复现成功率系统性偏高。
- 这些报告与 #60/#61 的指标缺陷叠加：即使跑通，**验证指标本身在修复前就不可靠**，论文数字与复现数字的比较需要先对齐指标实现。
- 依赖风险：vendored verl 快照 + 严格的版本钉选（ray 2.41–2.50、transformers ≤4.57.3、tensordict 0.8–0.10、vllm 0.8.5–0.11.0）；dependabot 的 5 个 open PR（pyarrow / transformers / ray / sglang / packaging 升级）无人处理，说明依赖升级无人维护。

---

## 8. 贡献视角（只观察，不认领）

### 8.1 已做：PR #62 的位置

PR #62 修的是验证指标的“按样本数加权”一半，位于 `verl/trainer/ppo/ray_trainer.py:793`（`_validate` 的 success_rate 聚合）。它的价值不在于代码量，而在于**它修复的是论文数字的测量工具**：在修复前，Bamboogle 这类小数据集的验证成功率会被尾巴 batch 主导，任何基于该指标的结论（包括“SDAR 优于 GRPO”）都有测量偏差。PR 正文里“4 个单元测试通过、ruff 通过”的验证是 CPU 侧的，合理；GPU 端到端验证未做（仓库无 CI，maintainer 也未要求）。

### 8.2 未来值得看的方向（观察清单，不构成认领）

按“与方法核心的贴近度 × 独立可验证性”排序：

1. **#60 缺陷二：`test_score` 按 `traj_uid` 去重**。位置 `ray_trainer.py:795-818`，隔壁 `tool_call_count` 已有 `np.unique(traj_uids)` 的去重范式可照抄。验证方式与 PR #62 同款（CPU 侧合成数据 + 单元测试），不依赖 GPU。这是最自然的 PR #62 续作。
2. **#61 的文档澄清**：`examples/opsd_trainer/` 与 `examples/grpo_opsd_trainer/` 的脚本注释与 README 需要说明“这两个脚本是 main_sdar 的参数化变体，不是独立教师 OPD 实现；独立教师入口是 `main_opd_trainer` / `main_grpo_opd_trainer`”。纯文档改动，零 blast radius，但需要 maintainer 确认 Table 1 的实际产生方式才能写准。
3. **教师 batch 左截断的度量**：`rlsd_ray_trainer.py:117-119` 的左截断会静默切掉技能前缀。加一个截断计数指标（或在技能被切时告警）是小而可验证的改进；改截断策略本身（例如优先保留技能前缀）是行为变更，需谨慎。
4. **`SkillProvider` 文档与实现不一致**：`rlsd_utils.py` 的类 docstring（ALL + 首个获胜）与 `get_privileged_info_from_prompt` 的实现（ANY + 全部拼接）矛盾。修文档是零风险；改实现为 ALL 是行为变更。
5. **RetireOPD 的 `TEACHER_PERFORMANCE` 手动输入**：脚本要求用户手动填教师验证成功率，填错直接改变退休阈值。能否从 Phase 1 checkpoint 或验证日志自动读取，是一个可用性改进，但需要先确认 Phase 1 的验证指标与 Phase 2 的 `success_metric` 是否为同一口径。
6. **Dependabot 依赖升级**：5 个 open PR 无人处理。逐个验证升级是否破坏训练需要 GPU 环境，不适合作为轻量贡献。

### 8.3 贡献环境判断

- **Maintainer 响应**：2026-06 的 #35/#36（ALFWorld 技能 bug）曾在 1 天内修复并合并；但 2026-09 下旬以来的 #59/#60/#61 与 PR #62 均无 maintainer 互动。响应速度已明显放缓，提交前应预期较长的等待期。
- **CI**：仓库无针对 PR 的通用 CI（`.github/workflows/` 只有 checkpoint_converter / dataset / doc / e2e_dapo 等专项 workflow）。PR #62 的 0 check runs 与此一致。贡献的验证需要自己在 PR 正文里写清（PR #62 的做法：CPU 侧单元测试 + ruff）。
- **PR 模板 / CONTRIBUTING**：本次未在仓库里发现 PR 模板与 CONTRIBUTING 文件（未逐目录穷举；如有遗漏需在提交前再确认）。PR #62 的正文结构（What / High-Level Design / Specific Changes / API / Test / Checklist）可作为既有范式参考。
- **分支规范**：贡献分支必须从上游 `master` 建（PR #62 的分支基底已是上游 `29dcb12a`，符合）；个人笔记绝不进 PR diff。

---

## 9. 开放问题

1. Table 1 的 OPSD / GRPO+OPSD 数字到底由哪套脚本产生？（#61 未回答）这决定了论文基线对比是否可信。
2. `gate_beta` 与 `sdar_coef` 是否在不同任务/模型规模下做过系统 sweep？脚本里三个任务、三个模型变体全部使用 `0.01 / 5.0`，论文未提供敏感性分析，代码里也没有 sweep 记录。
3. 技能 md 文件的质量如何保证？技能是手写文本，检索是关键词规则；技能错误会直接污染教师信号，而门控只能衰减、不能纠错。
4. RetireOPD 的 Phase 1 教师训练成本是否计入总成本？RetireOPD 需要先训一个完整的 Skill-GRPO 教师，再训学生；论文报告的收益是否扣除了这部分成本，未在 README 中说明。
5. 验证指标修复（#60、#62）之后，论文 Table 1 的数字是否需要重测？如果验证指标本身有偏差，历史数字的复核是一个 open 问题。
6. vendored verl 快照的同步计划是什么？上游 verl 在演进，本仓库的 verl 冻结在 2026-05 附近；长期看，同步成本会越来越高。

---

## 附录：关键文件索引（均核对自上游 master `29dcb12a`）

| 文件 | 关键位置 | 作用 |
|---|---|---|
| `verl/trainer/ppo/sdar_utils.py` | L14 `compute_sdar_loss` | SDAR 门控损失 |
| `verl/workers/actor/dp_actor.py` | L329 教师 logp 取数；L445–458 SDAR 分支；L431–443 SDL 分支 | 总损失组装 |
| `verl/trainer/main_sdar.py` | 全文 | SDAR 入口，注入 `use_sdar_loss` 等配置 |
| `verl/trainer/ppo/rlsd_ray_trainer.py` | L39 `build_teacher_batch`；L422 `_compute_teacher_log_probs` | 同模型自教师 batch 构造与前向 |
| `verl/trainer/ppo/skillsd_ray_trainer.py` | L61 `_get_training_teacher_log_probs`；L173–176 教师前向 | SkillSD 训练器（SDAR 的父类链） |
| `verl/trainer/ppo/skillsd_utils.py` | `compute_sdl_loss` | SkillSD 的 K3 式 SDL |
| `verl/trainer/ppo/rlsd_utils.py` | `SkillProvider`；`compute_rlsd_token_advantage` | 技能加载/匹配；RLSD token advantage |
| `verl/trainer/ppo/ray_trainer.py` | L691 `_validate`；L793 success_rate 聚合；L818 test_score | 验证指标聚合（#60/#62 位置） |
| `agent_system/multi_turn_rollout/rollout_loop.py` | L246 `gather_rollout_data` | 轨迹数据收集，episode reward 复制到每步 |
| `agent_system/environments/env_manager.py` | L119 / L214 / L508 三个 `_process_batch` | 三环境各自的 won / success_rate 产出 |
| `agent_system/reward_manager/episode.py` | `EpisodeRewardManager` | Episode 奖励管理（`normalize_by_length=False`） |
| `verl/trainer/main_opd_trainer.py` | `configure_opd_trainer` | 独立教师纯 OPD 入口 |
| `verl/trainer/main_grpo_opd_trainer.py` | `configure_grpo_opd_trainer` | 独立教师 GRPO+OPD 入口 |
| `verl/trainer/main_retire_opd_trainer.py` | `configure_retire_opd_trainer` | RetireOPD 入口与输入校验 |
| `verl/trainer/ppo/retire_opd.py` | `RetireOPDScheduler` | 退休调度器（窗口 + 对齐 + 胜任力） |
| `verl/trainer/ppo/retire_opd_trainer.py` | L49–65 退休钩子 | 退休后跳过教师前向、系数归零 |
| `verl/trainer/ppo/grpo_opd_trainer.py` | `_compute_teacher_log_probs` | 独立教师前向（`TeacherPolicy` worker） |
| `examples/sdar_trainer/run_*_3b.sh` | `sdar_coef=0.01`，`gate_beta=5.0` | SDAR 启动脚本（ALFWorld/WebShop/Search） |
| `examples/opsd_trainer/run_webshop_3b.sh` | `pg_loss_coef=0` | 名为 OPSD、实为 SDAR-无-GRPO 的脚本（#61） |
| `examples/grpo_opsd_trainer/run_webshop_3b.sh` | `gate_beta=0.0` | 名为 GRPO+OPSD、实为门控恒 0.5 的脚本（#61） |
| `examples/retireopd_trainer/run_alfworld_3b.sh` | `TEACHER_MODEL_PATH` / `TEACHER_PERFORMANCE` 必填 | RetireOPD Phase 2 脚本 |
| `skills/*/skill_mapping.json` | 各环境 | 技能文件映射 + 任务-技能映射 + 关键词表 |
