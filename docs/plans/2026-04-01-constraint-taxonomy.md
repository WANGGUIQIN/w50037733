# 操控约束条件分类体系

> 日期: 2026-04-01
> 基于: 2026-04-01-task-planning-design.md
> 状态: 设计阶段

## 1. 设计动机

原方案按**触发时机**将约束分为 progress / safety / completion 三类。
这适合运行时调度，但存在两个问题：

1. **约束语义模糊** — `cup_tilt < 15°` 到底是姿态约束还是安全约束？同一个物理量在不同阶段可能扮演不同角色
2. **无法系统化生成** — GPT 标注时缺乏结构化的约束维度清单，容易遗漏关键约束

本文档将约束按**物理语义**分为五类，每类定义清晰的谓词集合和量化方式。
运行时触发逻辑（progress/safety/completion）作为每条约束的**属性标签**，
而非顶层分类。

```
                  ┌────────────────────────────────┐
                  │     每条约束 = 物理语义 + 角色   │
                  │                                │
                  │  物理语义:  接触 / 空间 / 姿态   │
                  │            / 方向 / 安全        │
                  │                                │
                  │  角色标签:  completion / safety  │
                  │            / progress           │
                  └────────────────────────────────┘
```

## 2. 五类约束定义

### 2.1 接触约束 (Contact Constraints)

描述**末端执行器或物体之间是否存在物理接触**，以及接触的状态。

| 谓词 | 参数 | 量化方式 | 示例 |
|------|------|---------|------|
| `gripper_contact(obj)` | 目标物体 | 夹爪力传感器 > 阈值 / 3D距离 < ε | 夹爪接触壶柄 |
| `gripper_state(state)` | open / closed | 夹爪开合量 | 夹爪闭合 |
| `gripper_width(w)` | 宽度 (m) | 夹爪开口 = w ± δ | 开口 0.06m |
| `holding(obj)` | 目标物体 | 闭合 + 接触 + 相对位移 < ε | 稳定抓握茶壶 |
| `released(obj)` | 目标物体 | !holding(obj) | 松开茶壶 |
| `surface_contact(A, B)` | 两物体 | A 底面与 B 的距离 < ε | 茶壶底部接触桌面 |
| `liquid_contact(liquid, container)` | 液体, 容器 | 液体在容器内检测 | 水流入杯中 |

**3D Token 验证方式**: 从 3D scene tokens 提取夹爪区域与物体区域的最近点距离;
`holding` 进一步要求两区域跨帧保持刚性绑定。

### 2.2 空间关系约束 (Spatial Relation Constraints)

描述**物体之间的相对位置和距离关系**。

| 谓词 | 参数 | 量化方式 | 示例 |
|------|------|---------|------|
| `distance(A, B, op, d)` | 物体A, B, 比较符, 阈值 | 3D 中心距 | 夹爪距壶柄 < 3cm |
| `above(A, B, margin)` | A 在 B 上方 | z_A > z_B + margin | 壶嘴在杯口上方 5cm |
| `inside(A, B)` | A 在 B 内部 | A 的 bbox 被 B 包含 | 水在杯子内 |
| `aligned_xy(A, B, tol)` | A、B 水平对齐 | \|x_A - x_B\| + \|y_A - y_B\| < tol | 壶嘴对准杯口 |
| `height(A, op, h)` | 物体高度 | z_A op h | 茶壶高于桌面 10cm |
| `on_surface(A, surface)` | A 在表面上 | surface_contact + 静止 | 茶壶放在桌上 |
| `clear_path(A, B)` | 无障碍路径 | A→B 射线无遮挡 | 搬运路径畅通 |

**3D Token 验证方式**: DepthToGaussian 重建的点云直接提供 3D 坐标，
两物体 Gaussian 中心的欧氏距离即 `distance`，z 坐标差即 `above`。

### 2.3 姿态约束 (Pose Constraints)

描述**物体或末端执行器的朝向和旋转状态**。

| 谓词 | 参数 | 量化方式 | 示例 |
|------|------|---------|------|
| `tilt(obj, op, angle)` | 物体倾斜角 | 主轴与重力方向夹角 | 茶壶倾斜 < 10° (搬运中) |
| `upright(obj, tol)` | 物体竖直 | tilt < tol | 茶壶竖直放下 |
| `tilted(obj, axis, angle)` | 沿指定轴倾倒 | 绕 axis 旋转 angle° | 壶身绕 Y 轴倾斜 80° |
| `orientation(obj, quat)` | 目标姿态 | 四元数距离 < ε | 壶嘴朝向杯子 |
| `stable(obj)` | 静态稳定 | 角速度 + 线速度 < ε | 放置后不摇晃 |
| `level(obj, tol)` | 水平放置 | roll, pitch < tol | 茶壶水平放回 |

**3D Token 验证方式**: GaussianEncoder 的 PointNet++ 特征隐式编码了物体的
主方向，tilt_head MLP 从 token 回归主轴与垂直方向的夹角。

### 2.4 方向约束 (Direction Constraints)

描述**末端执行器的接近方向、运动方向和力的施加方向**。

| 谓词 | 参数 | 量化方式 | 示例 |
|------|------|---------|------|
| `approach_dir(vec)` | 接近方向 [x,y,z] | 末端 z 轴与 vec 夹角 < θ | 从上方接近壶柄 |
| `grasp_axis(axis)` | 抓握主轴 | 夹爪开合方向 ∥ axis | 沿壶柄纵向抓握 |
| `motion_dir(vec)` | 运动方向 | 帧间位移方向与 vec 夹角 < θ | 向上提起 |
| `pour_dir(src, dst)` | 倾倒朝向 | 壶嘴投影指向杯口 | 壶嘴朝杯子方向倾倒 |
| `retreat_dir(vec)` | 撤离方向 | 松手后末端向 vec 方向移开 | 放下后向上撤离 |

**3D Token 验证方式**: approach 向量与 3D Token 特征拼接后过 MLP 判断一致性;
motion_dir 通过相邻帧 3D Token 差分计算。

### 2.5 安全约束 (Safety Constraints)

描述**不可违反的物理安全边界**，违反则立即触发修正。

| 谓词 | 参数 | 量化方式 | 示例 |
|------|------|---------|------|
| `no_collision(A, B)` | 两物体不碰撞 | distance(A, B) > d_safe | 茶壶不撞杯子 |
| `within_workspace(obj)` | 在操作空间内 | position ∈ bounds | 茶壶不越界 |
| `no_spill(liquid, container)` | 液体不溢出 | 液面 < 容器边缘 | 搬运中水不洒 |
| `force_limit(obj, F_max)` | 力不超限 | 接触力 < F_max | 不捏碎茶壶 |
| `no_drop(obj)` | 不掉落 | holding(obj) 在搬运期间持续为真 | 搬运中不松手 |
| `thermal_safe(obj)` | 温度安全 | 物体温度在操作范围内 | (热茶壶 - 本例忽略) |
| `support_stable(obj)` | 放置后稳定 | 放下后 5 帧内 stable(obj) | 放下后不倒 |

**3D Token 验证方式**: 安全约束大多复合其他四类谓词;
`no_spill` = `tilt(container) < θ_spill` + `above(liquid_surface, rim) = false`;
violation 检测优先级最高。

---

## 3. 完整案例：端茶壶倒水

### 任务描述

```
"Pick up the teapot from the table and pour water into the cup"
```

### 场景假设

```
桌面场景:
  - 茶壶 (teapot): 位于桌面左侧，内有水，有侧柄
  - 茶杯 (cup): 位于桌面右侧，空杯
  - 桌面 (table): 平面支撑
  - 壶嘴 (spout): 茶壶出水口
  - 壶柄 (handle): 茶壶侧面把手
```

### 阶段分解 (5 个操作原语)

---

#### Stage 1: reach(teapot_handle) — 接近壶柄

**子任务语言**: *"Move the gripper to approach the teapot handle from the side"*

| 约束类别 | 约束条件 | 谓词表达 | 角色 |
|---------|---------|---------|------|
| 接触 | 夹爪张开待抓取 | `gripper_state(open)` | progress |
| 接触 | 尚未接触茶壶 | `!gripper_contact(teapot)` | progress |
| 空间 | 夹爪逐渐接近壶柄 | `distance(gripper, handle) → decreasing` | progress |
| 空间 | 夹爪到达壶柄附近 | `distance(gripper, handle) < 0.03` | **completion** |
| 方向 | 从侧面水平接近 | `approach_dir([1, 0, 0])` | safety |
| 安全 | 不碰到茶壶壶身 | `no_collision(gripper, teapot_body)` | safety |
| 安全 | 不碰到茶杯 | `no_collision(gripper, cup)` | safety |

```yaml
# plan.json Stage 1
step: 1
action: reach
target: teapot_handle
affordance: [0.32, 0.45]
constraints:
  contact:
    - {pred: gripper_state, args: [open], role: progress}
  spatial:
    - {pred: distance, args: [gripper, handle, "<", 0.03], role: completion}
  direction:
    - {pred: approach_dir, args: [1, 0, 0], role: safety}
  safety:
    - {pred: no_collision, args: [gripper, teapot_body]}
    - {pred: no_collision, args: [gripper, cup]}
done_when: "distance(gripper, handle) < 0.03 AND gripper_state(open)"
```

---

#### Stage 2: grasp(teapot_handle) — 抓握壶柄

**子任务语言**: *"Close the gripper to firmly grasp the teapot handle"*

| 约束类别 | 约束条件 | 谓词表达 | 角色 |
|---------|---------|---------|------|
| 接触 | 夹爪闭合到壶柄宽度 | `gripper_width(0.04)` | progress |
| 接触 | 夹爪与壶柄有接触力 | `gripper_contact(handle)` | **completion** |
| 接触 | 稳定抓握 | `holding(teapot)` | **completion** |
| 空间 | 抓握点在壶柄中段 | `distance(grasp_point, handle_center) < 0.02` | safety |
| 姿态 | 茶壶未被推倒 | `upright(teapot, 10°)` | safety |
| 方向 | 沿壶柄纵轴抓握 | `grasp_axis([0, 1, 0])` | safety |
| 安全 | 抓握力不超限 | `force_limit(handle, 20N)` | safety |

```yaml
# plan.json Stage 2
step: 2
action: grasp
target: teapot_handle
affordance: [0.32, 0.45]
constraints:
  contact:
    - {pred: gripper_width, args: [0.04], role: progress}
    - {pred: gripper_contact, args: [handle], role: completion}
    - {pred: holding, args: [teapot], role: completion}
  spatial:
    - {pred: distance, args: [grasp_point, handle_center, "<", 0.02], role: safety}
  pose:
    - {pred: upright, args: [teapot, 10], role: safety}
  direction:
    - {pred: grasp_axis, args: [0, 1, 0], role: safety}
  safety:
    - {pred: force_limit, args: [handle, 20]}
done_when: "holding(teapot) AND gripper_contact(handle)"
```

---

#### Stage 3: lift_and_transport(teapot → above cup) — 提起并搬运至杯上方

**子任务语言**: *"Lift the teapot and move it above the cup, keeping it upright"*

| 约束类别 | 约束条件 | 谓词表达 | 角色 |
|---------|---------|---------|------|
| 接触 | 持续抓握 | `holding(teapot)` | safety |
| 空间 | 茶壶离开桌面 | `height(teapot, ">", table + 0.10)` | progress |
| 空间 | 壶嘴到达杯口上方 | `above(spout, cup_rim, 0.05)` | **completion** |
| 空间 | 壶嘴对准杯口 | `aligned_xy(spout, cup_center, 0.03)` | **completion** |
| 空间 | 搬运路径畅通 | `clear_path(teapot, cup)` | progress |
| 姿态 | 保持竖直（防洒水） | `upright(teapot, 10°)` | safety |
| 方向 | 先向上再水平移动 | `motion_dir([0, 0, 1])` → `motion_dir([1, 0, 0])` | progress |
| 安全 | 不洒水 | `no_spill(water, teapot)` | safety |
| 安全 | 不掉落 | `no_drop(teapot)` | safety |
| 安全 | 不碰杯子 | `no_collision(teapot, cup)` | safety |

```yaml
# plan.json Stage 3
step: 3
action: transport
target: teapot
destination: above_cup
affordance: [0.65, 0.40]
constraints:
  contact:
    - {pred: holding, args: [teapot], role: safety}
  spatial:
    - {pred: height, args: [teapot, ">", "table + 0.10"], role: progress}
    - {pred: above, args: [spout, cup_rim, 0.05], role: completion}
    - {pred: aligned_xy, args: [spout, cup_center, 0.03], role: completion}
  pose:
    - {pred: upright, args: [teapot, 10], role: safety}
  direction:
    - {pred: motion_dir, args: [0, 0, 1], phase: "first_half", role: progress}
  safety:
    - {pred: no_spill, args: [water, teapot]}
    - {pred: no_drop, args: [teapot]}
    - {pred: no_collision, args: [teapot, cup]}
done_when: "above(spout, cup_rim, 0.05) AND aligned_xy(spout, cup_center, 0.03)"
```

---

#### Stage 4: pour(water → cup) — 倾倒注水

**子任务语言**: *"Tilt the teapot to pour water into the cup through the spout"*

| 约束类别 | 约束条件 | 谓词表达 | 角色 |
|---------|---------|---------|------|
| 接触 | 持续抓握 | `holding(teapot)` | safety |
| 接触 | 水流入杯中 | `liquid_contact(water, cup)` | progress |
| 空间 | 壶嘴保持在杯口上方 | `above(spout, cup_rim, 0.02)` | safety |
| 空间 | 壶嘴与杯口对齐 | `aligned_xy(spout, cup_center, 0.03)` | safety |
| 姿态 | 壶身沿 Y 轴倾斜 60°-90° | `tilted(teapot, Y, 60~90°)` | progress |
| 姿态 | 倾斜角逐渐增大 | `tilt(teapot) → increasing` | progress |
| 方向 | 壶嘴朝向杯子 | `pour_dir(spout, cup)` | safety |
| 安全 | 水不溢出杯子 | `no_spill(water, cup)` | safety |
| 安全 | 不掉落茶壶 | `no_drop(teapot)` | safety |

**完成判据**: 倒水量达到预设值，或持续倾倒 N 秒。
（在视觉上可近似为: 水流停止 + 壶内水面下降）

```yaml
# plan.json Stage 4
step: 4
action: pour
target: water
destination: cup
affordance: [0.65, 0.40]
constraints:
  contact:
    - {pred: holding, args: [teapot], role: safety}
    - {pred: liquid_contact, args: [water, cup], role: progress}
  spatial:
    - {pred: above, args: [spout, cup_rim, 0.02], role: safety}
    - {pred: aligned_xy, args: [spout, cup_center, 0.03], role: safety}
  pose:
    - {pred: tilted, args: [teapot, Y, "60~90"], role: progress}
  direction:
    - {pred: pour_dir, args: [spout, cup], role: safety}
  safety:
    - {pred: no_spill, args: [water, cup]}
    - {pred: no_drop, args: [teapot]}
done_when: "pour_duration > 3s OR water_level(cup) > 0.7"
```

---

#### Stage 5: upright_and_place(teapot → table) — 扶正并放回桌面

**子任务语言**: *"Tilt the teapot back upright and place it on the table"*

| 约束类别 | 约束条件 | 谓词表达 | 角色 |
|---------|---------|---------|------|
| 接触 | 持续抓握直到放下 | `holding(teapot)` | progress |
| 接触 | 茶壶底面接触桌面 | `surface_contact(teapot_base, table)` | **completion** |
| 接触 | 松开夹爪 | `released(teapot)` | **completion** |
| 空间 | 茶壶回到桌面高度 | `height(teapot_base, "<", table + 0.01)` | completion |
| 空间 | 放置在安全区域（不碰杯子） | `distance(teapot, cup) > 0.10` | safety |
| 姿态 | 恢复竖直 | `upright(teapot, 5°)` | **completion** |
| 姿态 | 放下后稳定 | `stable(teapot)` | **completion** |
| 方向 | 先扶正再向下放置 | `motion_dir([0, 0, -1])` | progress |
| 方向 | 松手后向上撤离 | `retreat_dir([0, 0, 1])` | progress |
| 安全 | 扶正过程不洒水 | `no_spill(water, teapot)` | safety |
| 安全 | 放置后不倒 | `support_stable(teapot)` | safety |

```yaml
# plan.json Stage 5
step: 5
action: place
target: teapot
destination: table
affordance: [0.30, 0.50]
constraints:
  contact:
    - {pred: surface_contact, args: [teapot_base, table], role: completion}
    - {pred: released, args: [teapot], role: completion}
  spatial:
    - {pred: height, args: [teapot_base, "<", "table + 0.01"], role: completion}
    - {pred: distance, args: [teapot, cup, ">", 0.10], role: safety}
  pose:
    - {pred: upright, args: [teapot, 5], role: completion}
    - {pred: stable, args: [teapot], role: completion}
  direction:
    - {pred: motion_dir, args: [0, 0, -1], role: progress}
    - {pred: retreat_dir, args: [0, 0, 1], role: progress}
  safety:
    - {pred: no_spill, args: [water, teapot]}
    - {pred: support_stable, args: [teapot]}
done_when: "surface_contact(teapot_base, table) AND upright(teapot, 5) AND released(teapot)"
```

---

## 4. 约束全景矩阵

### 4.1 约束类型 × 阶段 热力图

```
              Stage1    Stage2    Stage3    Stage4    Stage5
              reach     grasp     transport pour      place
  ─────────────────────────────────────────────────────────
  接触         ●○        ●●●       ●         ●●        ●●●
  空间关系     ●●        ●         ●●●●      ●●        ●●
  姿态         -         ●         ●         ●●        ●●●
  方向         ●         ●         ●●        ●         ●●
  安全         ●●        ●         ●●●●      ●●●       ●●
  ─────────────────────────────────────────────────────────
  约束总数     7         7         10        9         11
              ●=约束数   ○=否定约束
```

### 4.2 跨阶段持续约束

某些约束跨越多个阶段持续生效：

| 持续约束 | 生效阶段 | 说明 |
|---------|---------|------|
| `holding(teapot)` | Stage 2→5 | 抓握后全程持有，直到 Stage 5 放下 |
| `no_drop(teapot)` | Stage 3→4 | 搬运和倾倒阶段不能松手 |
| `no_spill(water, teapot)` | Stage 3, 5 | 竖直搬运和扶正阶段不洒水 |
| `no_spill(water, cup)` | Stage 4 | 倾倒阶段水不溢出杯子 |
| `no_collision(teapot, cup)` | Stage 1, 3 | 接近和搬运阶段不碰杯子 |

### 4.3 约束角色分布

| 角色 | 约束数量 | 占比 | 用途 |
|------|---------|------|------|
| **safety** | 21 | 47.7% | 违反 → CORRECT |
| **completion** | 13 | 29.5% | 全部满足 → NEXT |
| **progress** | 10 | 22.7% | 停滞 → REPLAN |
| **合计** | 44 | 100% | 5 个阶段总约束数 |

---

## 5. 约束与 ConstraintVerifier 的对应

### 5.1 所需 MLP Heads

从 3D Scene Tokens (512-dim) 回归各约束值：

```python
class ConstraintVerifier(nn.Module):
    """从 3D scene tokens 验证五类约束。"""

    def __init__(self, token_dim=512):
        super().__init__()

        # ---- 接触约束 heads ----
        self.contact_head = nn.Sequential(        # 二分类: 是否接触
            nn.Linear(token_dim * 2, 256),        # 拼接两物体 tokens
            nn.GELU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
        self.holding_head = nn.Sequential(        # 二分类: 是否抓握
            nn.Linear(token_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
        self.gripper_width_head = nn.Linear(token_dim, 1)  # 回归: 夹爪宽度

        # ---- 空间关系约束 heads ----
        self.distance_head = nn.Sequential(       # 回归: 两物体距离
            nn.Linear(token_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1), nn.ReLU(),
        )
        self.relative_pos_head = nn.Sequential(   # 回归: 相对位置 [dx, dy, dz]
            nn.Linear(token_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

        # ---- 姿态约束 heads ----
        self.tilt_head = nn.Sequential(           # 回归: 倾斜角度 (degrees)
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1), nn.ReLU(),
        )
        self.stability_head = nn.Sequential(      # 二分类: 是否稳定
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

        # ---- 方向约束 heads ----
        self.direction_head = nn.Sequential(      # 回归: 运动/接近方向 [x,y,z]
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),                    # 输出后 normalize
        )

        # ---- 安全约束 (复合) ----
        # no_collision = distance > threshold (复用 distance_head)
        # no_spill = tilt < threshold (复用 tilt_head)
        # no_drop = holding == True (复用 holding_head)
```

### 5.2 训练数据来源

| 约束类型 | 训练信号来源 | 标注成本 |
|---------|------------|---------|
| 接触 (contact, holding) | Episode 帧序列中夹爪状态变化点 | 低 (自动提取) |
| 空间 (distance, above) | 3D 点云直接计算 | 零 (几何计算) |
| 姿态 (tilt, upright) | 物体 Gaussian 主轴 vs 重力方向 | 零 (几何计算) |
| 方向 (approach, motion) | 帧间末端位移差分 | 零 (几何计算) |
| 安全 (复合) | 组合上述基础约束 | 零 (组合逻辑) |

关键发现: **大多数约束可从 3D 重建几何自监督学习**，不需要人工标注。
仅 `holding` 和 `liquid_contact` 需要轨迹级弱标注。

---

## 6. 对 plan.json 格式的升级建议

### 现有格式 (原方案)

```json
{
  "step": 1,
  "action": "reach",
  "target": "jar lid",
  "affordance": [0.52, 0.38],
  "approach": [0.0, 0.0, -1.0],
  "gripper": "open",
  "done_when": "gripper_near(jar_lid) AND gripper_is(open)"
}
```

问题: `approach` 和 `gripper` 是约束但没有归类; `done_when` 是纯字符串,
缺少安全约束和姿态约束的表达。

### 升级格式

```json
{
  "step": 3,
  "action": "transport",
  "target": "teapot",
  "destination": "above_cup",
  "affordance": [0.65, 0.40],

  "constraints": {
    "contact": [
      {"pred": "holding", "args": ["teapot"], "role": "safety"}
    ],
    "spatial": [
      {"pred": "above", "args": ["spout", "cup_rim", 0.05], "role": "completion"},
      {"pred": "aligned_xy", "args": ["spout", "cup_center", 0.03], "role": "completion"}
    ],
    "pose": [
      {"pred": "upright", "args": ["teapot", 10], "role": "safety"}
    ],
    "direction": [
      {"pred": "motion_dir", "args": [0, 0, 1], "role": "progress"}
    ],
    "safety": [
      {"pred": "no_spill", "args": ["water", "teapot"]},
      {"pred": "no_drop", "args": ["teapot"]},
      {"pred": "no_collision", "args": ["teapot", "cup"]}
    ]
  },

  "done_when": "above(spout, cup_rim, 0.05) AND aligned_xy(spout, cup_center, 0.03)"
}
```

### LLM 训练 target 文本格式 (升级后)

```
Step 3: transport(teapot → above_cup)
  affordance: [0.65, 0.40]
  contact: holding(teapot) [safety]
  spatial: above(spout, cup_rim, 0.05) [completion]; aligned_xy(spout, cup_center, 0.03) [completion]
  pose: upright(teapot, 10°) [safety]
  direction: motion_dir([0, 0, 1]) [progress]
  safety: no_spill(water, teapot); no_drop(teapot); no_collision(teapot, cup)
  done_when: above(spout, cup_rim, 0.05) AND aligned_xy(spout, cup_center, 0.03)
```

---

## 7. 与运行时调度的对接

五类约束 + 三种角色标签的运行时触发逻辑不变：

```python
def verify_step(scene_tokens, step_constraints):
    """每个控制周期调用一次。"""
    results = {}

    for category in ["contact", "spatial", "pose", "direction", "safety"]:
        for c in step_constraints.get(category, []):
            val = evaluate_predicate(scene_tokens, c["pred"], c["args"])
            results[c["pred"]] = val

    # 1. 安全约束: safety 类别 + role=safety → 任一违反则 CORRECT
    for c in all_constraints_with_role("safety"):
        if not results[c["pred"]]:
            correction = generate_correction(c)
            return Action.CORRECT, correction

    # 2. 完成约束: role=completion → 全部满足则 NEXT
    completion = [c for c in all_constraints() if c.get("role") == "completion"]
    if all(results[c["pred"]] for c in completion):
        return Action.NEXT, None

    # 3. 进度约束: role=progress → 停滞检测
    if progress_stalled(results, history, patience=30):
        return Action.REPLAN, None

    return Action.CONTINUE, None
```

注意: `safety` 类别的约束默认 role 就是 safety (可省略标注);
其他四类的约束需要显式标注 role。
