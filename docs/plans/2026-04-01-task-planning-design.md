# RoboBrain-3DGS 任务规划与约束生成方案

> 日期: 2026-04-01
> 状态: 设计完成，框架改动已实现，标注 pipeline 待启动

## 1. 目标

将 RoboBrain-3DGS 从**单帧 affordance 预测**扩展为 **pi0.5 式高层决策模块**：

- 接收抽象任务指令 + 场景 RGB-D → 输出操作原语序列 + 每阶段约束条件
- 约束条件用于运行时判断 π0 执行器是否需要轨迹修正或重规划
- 不直接生成动作，与 VLA 执行器（π0）通过语言接口解耦

## 2. 系统架构

```
用户指令: "把桌上的杯子放进柜子"
               │
               ▼
┌──────────────────────────────────────────────┐
│  RoboBrain-3DGS (高层决策 + 约束监控)          │
│                                              │
│  RGB-D ──→ DepthToGaussian ──→ 3D Tokens     │
│  RGB   ──→ Qwen3-VL ViT    ──→ 2D Tokens     │
│                                              │
│  [3D Tokens + 2D Tokens + Task Text]          │
│         ↓                                    │
│  Qwen3-VL LLM (8B, LoRA)                     │
│         ↓                                    │
│  输出: 操作原语序列 + 每阶段约束               │
└──────────────────┬───────────────────────────┘
                   │
    ┌──────────────┼──────────────────┐
    │              │                  │
    ▼              ▼                  ▼
 子任务语言     几何约束          完成/失败条件
    │              │                  │
    ▼              │                  │
┌──────────┐       │                  │
│ π0 执行器  │       │                  │
│ (端到端)  │       │                  │
└────┬─────┘       │                  │
     │ 执行动作     │                  │
     ▼              ▼                  ▼
  机器人 ───→ 新观测 ───→ RoboBrain 约束验证
                              │
                   ┌──────────┼──────────┐
                   │          │          │
                CONTINUE    CORRECT    REPLAN
                (不干预)   (修正语言)  (重新分解)
```

### 与 pi0.5 的核心差异

| | pi0.5 | 本方案 |
|---|---|---|
| 高层模型 | Gemma (纯语言 LM) | RoboBrain-3DGS (语言 + 3D 几何) |
| 场景理解 | 2D 图像描述 | 3D Gaussian Splatting 场景重建 |
| 约束形式 | 隐式(语言) | 显式(可量化验证的几何谓词) |
| 执行监控 | 语言判断"做完没" | 3D token 差异 + 约束值验证 |
| 修正信号 | "try again" | "cup tilting 23 degrees, keep upright" |

## 3. 操作原语设计

### 粒度原则

每个阶段对应一个**操作原语**(manipulation primitive)。粒度选择标准：

1. **语义完整性**: 对应一个可命名的操作动词 (reach/grasp/transport/place/pour/release)
2. **状态可区分**: 完成前后场景有明显可观测变化
3. **VLA 可泛化**: 同一原语在不同场景中动作不同但语义相同

一个任务通常拆分为 **2-5 个原语**，不超过 5 个。

### 不同粒度的对比

```
过细 (退化为轨迹回放):          过粗 (VLA 无法执行):
  Stage 1: 移动到 x=0.3         Stage 1: 收拾桌子
  Stage 2: 下降到 z=0.1
  Stage 3: 闭合夹爪
  ...8-10 个阶段

正确 (操作原语级别):
  Stage 1: reach(cup_handle)    — VLA 自行规划接近路径
  Stage 2: grasp(cup_handle)    — VLA 自行选择抓取姿态
  Stage 3: transport(cup, shelf) — VLA 自行控制搬运
  Stage 4: place(cup, shelf)    — VLA 自行放置
```

## 4. 约束条件设计

### 每阶段三类约束

```yaml
subtask: "grasp the red cup"

# 1. 进度约束 — 判断是否在正确推进
progress_constraints:
  - gripper_to_target_distance: decreasing   # 越来越近 = 正常

# 2. 安全约束 — 违反则立即修正
safety_constraints:
  - cup_tilt: "< 15 degrees"       # 杯子不能倒
  - collision: "gripper not in bowl" # 不撞到碗

# 3. 完成约束 — 满足则进入下一子任务
completion_constraints:
  - gripper_state: closed
  - gripper_holding: cup
  - cup_height: "> table + 3cm"
```

### 触发逻辑

```
每个控制周期 (观测新帧):

  if completion_constraints 全部满足:
      -> NEXT (发送下一个子任务给 pi0)

  elif safety_constraints 任一违反:
      -> CORRECT (生成修正语言，pi0 用新 prompt 重新生成 actions)

  elif progress_constraints 停滞超过 N 步:
      -> REPLAN (RoboBrain 重新分解任务)

  else:
      -> CONTINUE (不干预 pi0)
```

### 约束验证 — 基于 3D Scene Tokens

约束验证复用 RoboBrain 的 3D branch，加轻量 MLP heads：

```python
class ConstraintVerifier:
    # 从 3D scene tokens 回归各项指标
    distance_head = nn.Linear(512, 1)    # 物体间距离
    holding_head  = nn.Linear(512, 1)    # 是否抓住
    tilt_head     = nn.Linear(512, 1)    # 倾斜角度

    def verify(self, scene_tokens, constraints):
        # 检查每个约束是否满足
        # 返回: CONTINUE / CORRECT / REPLAN / NEXT
```

训练数据来源：episode 帧序列天然提供正样本，帧间 3D 变化可自监督学习。

## 5. 训练数据格式

### plan.json 结构

每个 episode 目录下放一个 GPT 生成的 `plan.json`：

```json
{
  "task": "close the black jar",
  "num_steps": 3,
  "steps": [
    {
      "step": 1,
      "action": "reach",
      "target": "jar lid",
      "affordance": [0.52, 0.38],
      "approach": [0.0, 0.0, -1.0],
      "gripper": "open",
      "done_when": "gripper_near(jar_lid) AND gripper_is(open)"
    },
    {
      "step": 2,
      "action": "grasp",
      "target": "jar lid",
      "affordance": [0.52, 0.38],
      "approach": [0.0, 0.0, -1.0],
      "gripper": "closed, width=0.06",
      "done_when": "gripper_is(closed) AND holding(jar_lid)"
    },
    {
      "step": 3,
      "action": "place",
      "target": "jar opening",
      "affordance": [0.50, 0.40],
      "approach": [0.0, 0.0, -1.0],
      "gripper": "open",
      "done_when": "lid_on(jar) AND gripper_is(open)"
    }
  ]
}
```

### 训练 target 文本格式

LLM 学习输出的文本（由 `unified_loader.py` 从 plan.json 转换）：

```
Step 1: reach(jar lid)
  affordance: [0.52, 0.38], approach: [0.00, 0.00, -1.00], gripper: open
  done_when: gripper_near(jar_lid) AND gripper_is(open)
Step 2: grasp(jar lid)
  affordance: [0.52, 0.38], approach: [0.00, 0.00, -1.00], gripper: closed, width=0.06
  done_when: gripper_is(closed) AND holding(jar_lid)
Step 3: place(jar opening)
  affordance: [0.50, 0.40], approach: [0.00, 0.00, -1.00], gripper: open
  done_when: lid_on(jar) AND gripper_is(open)
```

## 6. 框架改动 (已完成)

### 改动清单

| 文件 | 改动内容 | 状态 |
|------|---------|------|
| `utils/prompt_utils.py` | 新增 `"planning"` prompt 模板 | 已完成 |
| `utils/prompt_utils.py` | 新增 `parse_planning_output()` 解析器 | 已完成 |
| `data/unified_loader.py` | 新建，读 processed data + plan.json | 已完成 |
| `train.py` | 注册 UnifiedDataset，支持 `unified_root` 配置 | 已完成 |

### 未改动的模块 (无需改动)

| 模块 | 原因 |
|------|------|
| `models/robobrain_vlm.py` | LLM 生成文本的能力不变，只是 target 变了 |
| `models/depth_to_gaussian.py` | 3D 重建能力仍需保留，提供空间约束基础 |
| `models/gs_encoder.py` | Token 编码不变 |
| `models/gs_renderer.py` | 渲染损失仍用于监督 3D branch 学好场景几何 |
| `models/cross_modal_fusion.py` | 3D-2D 融合不变 |
| 训练损失 `L_lm + 0.1 * L_render` | LM 损失自动适配新 target，渲染损失不变 |

### 配置文件新增字段

```yaml
# config/train_lora.yaml 中新增
data:
  unified_root: "data/processed"
  unified_datasets: ["rlbench", "taco_play", "bridge"]  # 可选过滤
  unified_task_type: "planning"  # "affordance" 或 "planning"
```

## 7. 标注 Pipeline (待实施)

### GPT 标注流程

```
Step 1: 准备输入
  - 每个 episode 的 rgb_0.png (场景首帧)
  - meta.json 中的 task 字段

Step 2: 调用 GPT-4o API
  - 输入: 首帧图片 + 任务描述
  - 输出: plan.json (操作原语 + 约束)

Step 3: 保存到 episode 目录
  data/processed/{dataset}/episode_{N}/plan.json

Step 4: 质量验证
  - 抽样检查步骤数 (2-5)
  - 检查 affordance 坐标合理性
  - 检查 done_when 是否可执行
```

### 标注优先级

| 优先级 | 数据集 | Episodes | 原因 |
|--------|--------|----------|------|
| P0 | rlbench | 1,800 | 有精确任务描述，仿真数据稳定 |
| P1 | taco_play | 3,242 | 原生深度，任务多样 |
| P1 | nyu_franka_play | 365 | 原生深度 |
| P2 | bridge | 25,446 | 量大，真实场景 |
| P2 | furniture_bench | 5,100 | 复杂装配任务 |
| P3 | fractal | 86,586 | 量最大，但任务相对简单 |

### 成本估算

- GPT-4o API: 约 $0.01-0.03/episode (图片 + 文本)
- P0 (1,800 ep): 约 $30-50
- P0+P1 (5,407 ep): 约 $80-160
- 全量 (160K ep): 约 $2,000-5,000

## 8. 与 pi0 执行器的对接

### pi0 端微调

pi0 保持端到端训练，但语言输入从完整任务变为阶段子目标：

```
原始: "pick up the cup and place it on the shelf"
改为: "grasp the cup handle from the right side"  (来自 RoboBrain 的 Step N)
```

微调数据: 原始 LeRobot 连续轨迹 + 阶段级子目标标注

### 运行时交互

```python
# 伪代码
plan = robobrain.plan(scene_rgbd, "put the cup in the cabinet")
# plan = [Step(reach, cup), Step(grasp, cup), Step(transport, ...), Step(place, ...)]

for step in plan:
    pi0.set_language(step.to_language())  # "grasp the cup handle"

    while True:
        action = pi0.act(current_obs)
        robot.execute(action)

        result = robobrain.verify(current_obs, step.constraints)

        if result == NEXT:
            break
        elif result == CORRECT:
            pi0.set_language(result.correction)  # corrective language
        elif result == REPLAN:
            plan = robobrain.replan(current_obs, remaining_task)
            break
```

## 9. 训练计划

### Phase 1: 验证 (1-2 周)

- [ ] 设计 GPT 标注 prompt，在 RLBench 10 个 episode 上手动验证输出质量
- [ ] 批量标注 RLBench 1,800 episodes
- [ ] 用 planning task_type 训练 LoRA，验证模型能生成合理的任务分解
- [ ] 评估: ROUGE/BLEU vs GPT 标注, 人工评估步骤合理性

### Phase 2: 扩展 (2-3 周)

- [ ] 扩展标注到 taco_play + nyu_franka_play (有原生深度的数据集优先)
- [ ] 混合训练: 50% affordance + 50% planning (多任务)
- [ ] 添加 ConstraintVerifier heads，用帧序列自监督训练
- [ ] 评估: 约束验证准确率 (用 episode 帧序列构造正负样本)

### Phase 3: 端到端联调 (3-4 周)

- [ ] pi0 微调: 用阶段子目标标注改写 LeRobot 训练数据
- [ ] RoboBrain + pi0 联合测试 (仿真环境 RLBench)
- [ ] 闭环测试: 注入干扰，验证 CORRECT/REPLAN 触发
- [ ] 真实机器人部署测试

## 10. 数据 Pipeline 修复记录

在设计本方案过程中，修复了以下数据 pipeline 问题：

| 修复 | 影响 |
|------|------|
| `taco_play` 原生深度提取 (3,242 ep) | OXE `depth_static` 字段未被读取，已修复 |
| `nyu_franka_play` 原生深度提取 (365 ep) | OXE `depth` 字段未被读取，已修复 |
| `run_pipeline.py` OUTPUT_DIR 路径错误 | 指向不存在路径，已修正 |
| DROID resume 支持 | 无断点续传，已加 `start_chunk` + skip-existing |
| CACHE_DIR `/tmp` 空间不足 | 改为主磁盘 `.cache/pipeline` 目录 |
| RH20T `gdown` 未安装 | Google Drive 下载依赖缺失，已安装 |
| RH20T HF fallback bug | GDrive 成功后 fallback 仍运行，已修复 |

### 数据集当前状态

| 数据集 | Episodes | 深度类型 | 状态 |
|--------|----------|---------|------|
| rlbench | 1,800 | native | 完成 |
| taco_play | 3,242 | native (已修复) | 完成 |
| nyu_franka_play | 365 | native (已修复) | 完成 |
| bridge | 25,446 | pseudo | 完成 |
| fractal | 86,586 | pseudo | 完成 |
| furniture_bench | 5,100 | pseudo | 完成 |
| jaco_play | 976 | pseudo | 完成 |
| berkeley_cable | 1,482 | pseudo | 完成 |
| utokyo_xarm | 64 | pseudo | 完成 |
| aloha | 20 | pseudo | 完成 |
| rh20t | 30 | pseudo | 完成 (GDrive 深度受限) |
| droid | 34,404 | pseudo | 处理中断，待修复 |
