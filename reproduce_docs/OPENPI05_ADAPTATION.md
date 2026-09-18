<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenPI05 策略在 SimFoundry C 阶段评测的适配过程

> 用途：记录把 openpi05（`pi05_droid_jointpos_polaris`）接到 C 阶段
> `1_eval_policy_og_scene.py` 上、并让评测真正跑出成绩的完整过程——包括**走过的弯路**。
> 每条结论都标注了证据来源；**推断**与**实测**分开写。
> 最后更新：**2026-09-18**。
> 本轮新增：§3.6（端末执行器一换，夹爪信号静默反向）、§3.7（物体尺寸 vs 夹爪开度）、
> §9（视频帧率已改为派生；`workspace_bounds` 收窄并实测被拖拽比例）、§12（仓库外的关键产物与启动参数）。
> 相关代码：`scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py`、
> `simfoundry/policies/openpi.py`、`simfoundry/tasks/pick_place_task.py`、
> `scripts/cfg/real2sim_cfg.yaml`。

## 0. 结论速览

| 假设 | 判定 | 影响量级 |
|---|---|---|
| 远程网关缺 `Api-Key` 认证头 | ✅ 真实缺陷，已修 | 必要但不充分 |
| 远程 fork 的 `AbsoluteActions` 没生效 | ❌ **误判**（测了陈旧进程） | 0 |
| checkpoint 版本不对 | ❌ 排除（本地官方权重同样失败） | 0 |
| 场景每次初始化不同 | ✅ 真实问题，已修（`init_states_path`） | 可复现性，非分数 |
| 外部相机几何（180×320 vs 720×1280） | ❌ 无显著影响 | 0 |
| 夹爪型号（Panda → Robotiq） | ⚠️ 必要但**不是**根因 | 中 |
| 桌面高度需要抬升 | ❌ 排除（模型输入里没有世界 z） | 0 |
| **夹爪约定（状态 + 动作）** | ✅ **根因** | **决定性** |
| **腕部相机分辨率** | ✅ 第二大因素 | **大** |
| prompt 措辞 | ❌ 无显著影响（所有检验 p≥0.44） | 0 |
| `execute_horizon` 改成 24 | ❌ 不建议（那是 gr00t 配方） | 0 |
| 加真实桌面背景（`droid_v1` 扫描） | ⚠️ **未验证成功**（渲染与定位都有问题） | 未知 |
| **端末执行器被换掉（robotiq → panda hand）** | ✅ **本轮新发现；夹爪信号静默反向** | **0/5 ↔ 5/5** |
| **物体大于夹爪开度** | ✅ 真实物理瓶颈（苹果 110 mm vs 开度 100 mm） | **决定性** |
| A 阶段深度分辨率导致球状物体被放大 1.5–1.7× | ✅ 根因（见 `A_STAGE_OBJECT_SIZE_BIAS.md`） | 大 |
| 提高相机分辨率以「对齐训练」 | ❌ 无意义（模型恒 `resize_with_pad` 到 224×224） | 0 |
| 用 task 定义阻止「抓错物体」 | ❌ 不可能（task 只管判据，不管动作） | 0 |
| 把白地板平面关掉 | ⚠️ 无背景时桌子会悬空，更难看 | 0（纯观感）|

**净效果**：官方精选场景从 **0/5（平均里程碑 5%）→ 5/5（100%）**；
我们自己的重建场景从 **任何里程碑从未达成 → 3/5 成功、90% 里程碑**；
加了桌子 + 缩小水果后：**5/5 成功、100% 里程碑**（`H3_5ep`，详见 §11）。

---

## 1. 起点症状

2026-09-16 之前，C 阶段 stage 1 用 openpi 评测 `Data/fruits` 场景，
5 轮全部失败，且**任何一个里程碑在任何时刻都从未达成**（`rollouts.hdf5` 的
`milestones` 逐帧全 0，reward 恒 0）。视频里机械臂表现为：手指几乎不动、动作幅度极小。

关键线索：动作向量前 7 维的数值**接近 0**（≈ 关节零位），而 `1_eval` 用的是
`JointController(use_delta_commands=False)`——**绝对位置控制**。
所以"接近 0"不是"不动"，而是**命令机械臂回到全零位**（竖直朝上、离开画面）。
这说明问题在**观测或动作的编码**，不在策略本身。

---

## 2. 排查过的假设（按时间顺序，含被否掉的）

### 2.1 任务为空导致"假成功"（有效，已修）

第一版用默认 `task=load_scene`。它的 YAML 是 `goal_predicates_all: null`，
**空谓词集合恒为真**，于是 `env.task.success` 在第 0 步就为 True，评测 1 步就"成功"结束，
成功率 1.0 毫无意义。

**修**：改用为本场景写的任务 `task=droid/droid_desk_serve_fruits`。

### 2.2 远程 server 的认证（有效，已修）

远程 server 走的是带网关的 WSS 端点（`wss://<instance>.seetacloud.com:8443`），
网关要求 `Authorization: Api-Key <key>`，否则握手 401。

**修**：`simfoundry/policies/openpi.py` 的 `OpenPIClient.__init__` 增加 `api_key`，
转发给 `websocket_client_policy.WebsocketClientPolicy(host, port, api_key=...)`；
key 从环境变量 `SIMFOUNDRY_OPENPI_API_KEY` 读，**不写进任何 YAML**（避免共享文件带密钥），
也因此同一份配置既能连需要认证的远程 server，也能连本地无认证 server。

### 2.3 「远程 fork 的 AbsoluteActions 没生效」（❌ 误判，已撤回）

远程实例跑的是对方的 fork（`PbTfcLx/openpi`，HEAD `b4998d0`），
它把 `AbsoluteActions` **直接 patch 进了 `pi05_droid`**（`src/openpi/training/config.py`），
注释写着：*"注意顺序：AbsoluteActions 需要 data['state'] 和未裁剪的 actions，
而 DroidOutputs 会把它裁剪成只剩 {"actions": 前 8 维}。因此必须先做 AbsoluteActions，
否则推理时会 KeyError: 'state'。"*

我先在 15:53/16:00 测出"AbsoluteActions 未生效"（注入 0.7 rad 的状态偏移，输出只响应 0.135，
落在噪声量级 0.141 内）。**但这个结论是错的**：17:00 复测时响应是 **0.668 vs 注入 0.700**，
明显生效。原因是 `ps` 显示该进程 `etime` 只有 12:58——**server 在 ~16:41 重启过，
我第一次测的是一个陈旧进程**。

**教训**：测服务端行为前，先确认进程的启动时间与被测配置一致。
我正式撤回了这个结论，以及随后基于它得出的"模型对图像不敏感"
（在那种状态下模型的 delta 只有 ~0.05 rad，图像引起的 0.024 变化其实是 delta 的一半，
拿它跟"噪声"比是没有意义的）。

### 2.4 checkpoint 版本（❌ 排除）

涉及三个不同的权重，曾经混在一起：

| 权重 | 来源 | `AbsoluteActions` |
|---|---|---|
| `pi05_droid` | 远程 fork，被 patch | 手动 patch 进去 |
| `pi05_droid_jointpos_polaris` | 官方发布，本地 | 自带（`action_space=JOINT_POSITION` 时由 `DataConfig` 自动加 `DeltaActions`+`AbsoluteActions`） |

依据：`src/openpi/training/config.py:403-408`
`if self.action_space == JOINT_POSITION: data_transforms.push(inputs=[DeltaActions(mask)], outputs=[AbsoluteActions(mask)])`；
而 `pi05_droid` 用的是裸 `DroidInputs`/`DroidOutputs`，没有这个 transform。

**排除依据**：远程（打补丁的 `pi05_droid`）与本地（官方 `pi05_droid_jointpos_polaris`）
**都**是 0/5，动作轨迹相关性只有 0.04–0.31。两个权重在同一 harness 下表现一致 →
问题不在权重。

最终切到本地官方权重（**12.44 GB，26 个文件，从 GCS 下载并逐字节校验**），
用本地 server（`localhost:8000`）评测，消除了认证、网关、fork 差异三个变量。

### 2.5 场景初始状态不可复现（有效，已修）

`PickPlaceTask._reset_scene` 每次 `env.reset()` 都会重掷布局：
`group_xyz_randomization` 抖动、`group_z_rot_randomization` 旋转、
以及 `group_predicate_placement`——后者会**随机选一个空间谓词**
（`left_of`/`right_of`/`behind`/`in_front_of`）加随机间距，以香蕉为参照摆盘子。
所以两轮评测从来不是同一个起始状态，逐里程碑对比没有意义（早期甚至出现
"第 0 步就已经 `placed_apple_on_teal_plate`=True"）。

**修**：`s15_eval.init_states_path` 固定布局。注意它**会覆盖 `n_episodes`**
为文件里的条目数（文件里放 5 条相同状态 = 要 5 轮可比的评测）。机器人**故意不固定**：
它不被随机化，而 reset 后重新摆机器人会让控制器失步。

**验证**（而非假设）：`1_eval` 现在打印 `[LAYOUT/after_reset]` 与 `[LAYOUT/after_init_state]`，
实测 `after_reset` 每轮不同、`after_init_state` 跨轮完全一致。

### 2.6 相机几何 A/B（❌ 无显著影响）

`nv_franka_droid.yaml` 里外部相机是 `180×320`（旁边注释着 `#720`/`#1280`），
但 `1_eval` 曾**无条件**覆盖成 720×1280，且注释谎称"224x224 as expected by DROID/OpenPI"。

做了 A/B：A = 用 YAML 自己的 180×320；A+B = 强制 720×1280。
**两边都是 0/5、0 里程碑**。所以外部相机几何不是瓶颈。

顺带修掉：`1_eval` 里改 sensor 分辨率的位置是错的。
**env 建好之后再改 sensor 尺寸会让 observation space 失效**，
下一次 `env.reset()` 直接抛 `ValueError: Observation space does not match returned observations!`
然后 SIGSEGV（exit 139）。正确做法是把 `sensor_config` 在**构造 env 之前**注入场景 JSON。

### 2.7 夹爪型号：Panda → Robotiq（⚠️ 必要，但不是根因）

DROID 的硬件描述是统一的："a Franka Panda 7DoF robot arm, two adjustable Zed 2 stereo
cameras, a wristmounted Zed Mini stereo camera"。变的是场景（564 个场景、1417 个视角），
不是本体。所以"本体相似度"是强杠杆，"场景相似度"是弱杠杆。

仓库自己的证据：**所有** DROID 任务配置都用 `end_effector: robotiq`
（`PutCupInBowl` / `PutMarkerInCup` / `ServeFruitsOnGreenPlate` / `cluttered_scene` /
`serve_banana` / `stack_dishware` / `stack_dishware_easy`），
而 `real2sim_cfg.yaml` 是唯一的例外（`gripper`）。
`simfoundry/tasks/macros.py` 也已经带 `GAINS["franka_robotiq"]`，手指限位 π/4。

**结论**：换 Robotiq 是对的（后来 HF 上那个官方精选场景自身也用 `end_effector: robotiq`，
独立佐证），但**单独换它并没有让成绩从 0 变好**。真正的问题是下面这一条。

> **2026-09-18 补充**：本节写于「换回 robotiq 是修 bug」这个认识之前。现在回头看，
> **换端末本身就是决定性的**，只是当时是在 robotiq 场景上修编码约定，没意识到
> 「换了端末会让编码反向」。反过来做了对照：把 `end_effector` 从 robotiq 改回
> `gripper`（panda hand），成绩从 5/5 变成 **0/5、里程碑全 0**。详见 §3.6。

### 2.8 桌面高度（❌ 排除）

一度怀疑"重建不含桌面/地面，物体都放在 z=0 平面上"是问题。
**排除依据**：策略的输入只有 2 张图 + 7 个关节角 + 夹爪 + 语言，
**没有任何世界坐标的 z**。整场景竖直平移对模型是不可观测的。
真正的要求是"机器人基座与物体共面"，而这一点本来就满足。

### 2.9 夹爪约定（✅ **根因**）

见 §3。

### 2.10 prompt 措辞 / `execute_horizon` / 背景（详见 §5、§6）

---

## 3. 决定性 bug：夹爪约定整个反了

### 3.1 权威依据

- **openpi 固定了策略侧**（`docs/norm_stats.md:66`）：
  *"Gripper positions are in [0.0, 1.0], with 0.0 corresponding to fully open and
  1.0 corresponding to fully closed."*
- **OmniGibson 固定了仿真侧**：`manipulation_robot.py:1617-1619` 明确写了
  `MultiFingerGripperController` 的语义：
  ```
  # - Non-inverted: closed = lower   - Inverted: closed = upper
  grasp_non_inverted = control < upper_limits
  grasp_inverted     = control > lower_limits
  ```
  `1_eval` 给夹爪控制器传的是 `inverted=True` ⇒ **闭合端 = 关节上限**。

### 3.2 两个夹爪方向相反，不能用同一套编码

| 夹爪 | 关节类型与量程 | 张开 | 闭合 |
|---|---|---|---|
| `franka_panda` | 平动指 `[0, 0.04] m` | 0.04 | 0 |
| `franka_robotiq` | 转动铰 `[0, 0.7854] rad` | **0** | **0.7854** |

而代码里对**两个夹爪都**写死了 `gripper_np = 1.0 - gripper_norm`——
对 Panda 正确，**对 Robotiq 整个反向**：夹爪大张时告诉模型"完全闭合"。

### 3.3 动作侧也反了，而且**不抵消**

从 rollout 里实测（当时配置是 `gripper_invert=false`）：
策略指令 **0 → 铰到 +0.7854（闭合）**；指令 **1 → 铰到 0.0（张开）**。两个都是反的。

合成效果是：**模型永远打不开夹爪**。数据完全吻合——夹爪 85% 的时间停在下限，
而模型 90% 的时间在喊 "open"。

### 3.4 修复

**旧写法（已被 §3.6 取代）**：
```python
GRIPPER_GRASPS_POSITIVE = True   # robotiq；Panda 用 False
gripper_np = gripper_norm if GRIPPER_GRASPS_POSITIVE else 1.0 - gripper_norm
```
这个常量就是§3.6 那个坑的根源：它把「哪个方向算闭合」当成了全文件唯一的常量，
而这是**端末执行器的属性**。现在改成从关节几何推导（见 §3.6）。

**动作侧**必须保留 `gripper_invert` 的默认值 `true`——
之前把 `inverted=True` 的控制器配上 `1-action[7]` 判断为"两次翻转抵消"是**错的**，
改成 `false` 是回归，已撤销。

### 3.5 效果

| | 修前 | 修后 |
|---|---|---|
| 官方精选场景 | 0/5，平均里程碑 **5%** | **5/5，100%**，197–503 步完成 |
| 我们的 `Data/fruits` + robotiq | 0/5，**里程碑从未达成** | 0/5，平均里程碑 **50%**（抓蕉 100%、放蕉 80%、抓苹 20%） |

### 3.6 ⚠️ 端末执行器一换，夹爪信号就静默反向（2026-09-18 新发现）

**症状**：同一场景、同一水果、同一 prompt、同一 h=8，只是把端末从 robotiq 换成
`s14_og.robot_config.end_effector=gripper`（Franka 原装平行夹爪），成绩从 0% 变成
**0/5、四个里程碑全 0**，而**工作空间检查全部通过**（三目标 base-frame x = 0.519/0.723/0.356，
全在界内）—— 所以既不是够不着，也不是任务判据问题。

**指纹（一眼可辨）**：

| | robotiq 场景 | panda hand 场景 |
|---|---|---|
| `joint_pos` 维度 | 15 | 9 |
| DOF 第 8+ | `left/right_outer_knuckle_joint` | `panda_finger_joint1/2` |
| 行程 | `[0, 0.7854]` rad，**0 = 张开** | `[0, 0.04]` m，**0 = 闭合** |
| 初始 `panda_joint7` | 0.0 | 0.75 |
| 实测送出的 `gripper_position` | `[0.]` → 读作全开 ✓ | `[1.]` → 读作全闭 ✗ |

**根因**：`gripper_norm = qpos[gripper_idx] / gripper_limit` —— 对 panda hand，
手指全开`= 0.04 = 上限`，所以 `gripper_norm = 1.0`，而 DROID 约定 `1.0 = 完全闭合`。
**手指物理上大张，却告诉模型「已经夹死了」**。而 `GRIPPER_GRASPS_POSITIVE = True`
是在 robotiq 上标定的（代码注释自己写着 `Measured on franka_robotiq from a rollout`）。

**同样的事实还有两处在 HEAD 里，也都是 robotiq 标定值，也必须一致**：
`controller_cfg["gripper_*"]["inverted"] = True`、`s15_eval.gripper_invert` 默认 `True`。
**三处必须一起对**——只改一处会变成「观测对、动作反」。

**修法（配置优先）**：不再用常量，改为按关节几何推导：
```python
def finger_dofs_are_prismatic(robot, arm):
    """平动手指（panda，限位是 m 且下界 0）还是转动铰（robotiq，限位是 rad）。"""
    dof_idx = [int(i) for i in robot.gripper_control_idx[arm]]   # 存的是 0 维 tensor，不是 int
    lo = ...joint_lower_limits[dof_idx]...
    hi = ...joint_upper_limits[dof_idx]...
    return len(dof_idx) == 2 and all(lo≈0) and all(0.005 < hi < 0.2)

positive_grasps = not finger_dofs_are_prismatic(robot, robot.default_arm)
```
robotiq → `True`（与旧常量一致，**已有运行零影响**）；panda hand → `False`（修正）。
可用 `s15_eval.gripper_positive_grasps` 显式覆盖（默认 `null` = 推导）。

> **踩坑记录**：这个函数的第一版用 `list(...)` 拿到的是 **0 维 tensor**，
> 直接索引 1-D 的 `joint_lower_limits` 会抛 `IndexError: too many indices for tensor of
> dimension 1`。当时只做了「看代码推演」就当成已验证，结果把整轮评测打挂（exit 139）。
> 改成 `[int(i) for i in ...]` 后，用 `ast` 从源码抽出函数、喂伪造 robot 真跑才确认。
> **教训：这类 API 细节必须实跑，推演不算验证。**

### 3.7 物体尺寸必须小于夹爪开度（否则物理上抓不起来）

A 阶段把球状物体放大了：苹果 `112.5×110.2×113.9 mm`、橙子 `133.9×133.8×121.1 mm`
（根因是深度分辨率，见 `A_STAGE_OBJECT_SIZE_BIAS.md`）。而：

| 夹爪 | 最大开度 | 苹果 110 mm | 橙子 121 mm | 香蕉 39 mm |
|---|---|---|---|---|
| robotiq 2F-85（内侧指尖原点间距 99.7 mm；真实行程 85 mm） | ~100 mm | ✗ | ✗ | ✓ |
| panda hand（2 × 0.04 m） | 80 mm | ✗ | ✗ | ✓ |

**两个球状水果在两种夹爪下都夹不住 —— 没有任何策略能抓起它们。**
这解释了为什么 `picked_banana` 在 8 个配置里恒为 100%，而 `picked_apple` 在 40–100% 之间摆：
**香蕉是长条（39 mm），永远能抓；苹果是球（110 mm），永远抓不住。**

**修法（纯配置）**：改 `objects_info.init_info[<obj>].args.scale`，在 `Data/` 之外做场景变体：
```bash
PY=/root/autodl-tmp/simfoundry/conda_envs/simfoundry/bin/python
$PY /root/simfoundry_logs/make_scene_variant.py \
    --src Data/fruits/s14_og_table_robotiq/reconstructed_og_scene.json \
    --out /root/autodl-tmp/simfoundry/scene_variants/fruits_table_robotiq_smallfruit/reconstructed_og_scene.json \
    --target iter_3 --scale 0.667      # 苹果 112.5 -> 75.0 mm
# 再对 iter_4（橙子）跑一次 --scale 0.600 -> 80.3 mm
```

**已内建检查**：`1_eval` 现在会在跑第一集前打印一张表，并把结果写进
`eval_results.json["grasp_feasibility"]`：端末名、夹爪 DOF 名、**实测开度**、每个物体的
最窄边与余量。开度不假设任何方向约定（平动手指取行程和，转动铰取指节原点间距），
`fixed_base` 的布景（桌子）会被跳过。余量阈值是配置
`s15_eval.grasp_clearance`（默认 0）。

---

## 4. 方法学：「阳性对照」是关键手段

**问题**：0/5 到底说明"策略/harness 坏了"还是"我们的重建场景太差"？两者都解释得通。

**做法**：从 Hugging Face（`nadunRanawaka1/simfoundry-assets`）取仓库 README 里
自己推荐的示例场景 `assets/scenes/DROID/droid_desk_serve_fruits/`，
在**同一 harness、同一 server、同一 prompt/horizon** 下评测。

**这个场景是仓库为本任务准备的参考场景**，证据：`scripts/cfg/ServeFruitsOnGreenPlate.yaml`
把 `task: droid_desk_serve_fruits` 与 `scene_json: droid_desk_serve_fruits` 配在一起；
任务 YAML 的 `semantic_group_mapping` 明确同时覆盖两种命名
（`teal_plate: [teal_tray, teal_plate]`、`orange: [orange_fruit, orange]`，
注释写着 *"authored scenes say orange_fruit, the pipeline says orange"*）。

它由**真实 BEHAVIOR-1K 资产**（`zlgmzo`/`rofqtq` 这类 model id，各带 10–23 MB 贴图）
加一段**真实 DROID 桌面的摄影测量扫描**（`droid_v1.usdz`，14 MB）组成。

**价值**：它把"策略坏了"这一支排除掉，并把注意力从"场景"拉回到"夹爪约定"——
修好夹爪后对照直接跳到 5/5。

**注意**：该 YAML 的官方配方是 **gr00t**（`policy: gr00t`, port 5555,
checkpoint `n17_cs_er2`），openpi 那行是注释掉的。我用 openpi 跑它是为了**与我们的场景同模型**，
对照才成立。

---

## 5. 第二大因素：腕部相机分辨率

| 臂 | 腕部 | 成功 | 平均里程碑 | [抓蕉, 放蕉, 抓苹, 放苹] |
|---|---|---|---|---|
| p0w0 | 128×128 | 0/5 | 50% | [5, 4, 1, 0] |
| p0_640 | 640×640 | 2/5 | 70% | [5, 5, 2, 2] |
| **p0w1** | **1280×720** | **3/5** | **90%** | **[5, 5, 5, 3]** |
| **baked** | 场景自带 720 | **3/5** | **90%** | **[5, 5, 5, 3]** |

- `picked_apple`：128 时 1/10，720 时 8/10，**Fisher p = 0.0055（显著）**。
- 整体成功 0/10 → 4/10（p = 0.087，方向一致但样本不足）。
- **已固化**：`s14_og.robot_config.sensor_config` 现在带 720×1280，
  所以每个新重建的场景自带 DROID 16:9 腕部相机，不再落到 OmniGibson 的 128×128 默认值。
  `baked` 臂与 `p0w1` 里程碑**逐位相同**，证明"烙进产物"等价于"评测时覆盖"。
  在此之前，**每一个重建场景都是在被削弱的相机配置下评测的**。

**这个旋钮不是纯分辨率**（必须写明）：`1_eval` 固定
`horizontal_aperture=5.376`、`focal_length=2.8`，而 USD 由 height/width 推导
`verticalAperture`，所以 128×128 是**正方形视场**（约 87.6°），1280×720 是
**16:9 的窄竖视场**（约 56.8°）。

想分离"分辨率"与"取景"的尝试**没成功**（n=5 不够）：

| 对比 | p 值 |
|---|---|
| 128 vs 720 | 0.0055 显著 |
| 128 vs 640 | 0.35 不显著 |
| 640 vs 720 | 1.00 不显著 |

有个方向性信号：640×640 是 128×128 的 **25 倍像素**，若纯粹是"看得清"，
它本该接近 720 的成绩，实际 70% < 90% —— 暗示取景也有贡献。**但要判定需要 n≈20。**

---

## 6. 试过但**没有效果**的

### 6.1 prompt 措辞

对比两段式 `"put the yellow banana on the teal plate, then put the red apple on the teal plate"`
与并列句 `"put both the banana and the apple on the teal plate"`（结构对齐官方措辞）：

| 对比 | p 值 |
|---|---|
| 128 下对整体成功 | 1.00 |
| 128 下对 `picked_apple` | 1.00 |
| 720 下对整体成功 | 0.52 |
| 720 下对 `picked_apple` | 0.44 |

**全部不显著。保留两段式**；不要声称官方并列句更好。

### 6.2 `execute_horizon` 改成 24

`ServeFruitsOnGreenPlate.yaml:262` 写的 `execute_horizon: 24` 是 **gr00t 配方**，不能照抄给
openpi：openpi 自己的 DROID 参考部署用 `open_loop_horizon = 8`
（`openpi/examples/droid/main.py:42`），`real2sim_cfg.yaml:493` 本来就是 8。
而且 `n_actions_per_chunk = min(execute_horizon, len(chunk))`，chunk 恒为 15，
所以 24 实际等于"吃完整个 chunk"（15），并不是"horizon 24"。

### 6.3 加真实桌面背景（**未验证成功**，两条独立原因）

动机：重建场景没有任何桌面，外部相机看到的是"无限白色平面 + 纯色天空"，
而 DROID 的训练图是真实房间里的真实桌子——腕部修好之后，这是最大的剩余视觉域差距。

做法：把已经下载的 `droid_v1` 桌面扫描挂成 `mesh_background_0`。
位姿**不是猜的**：保留参考场景的背景**朝向**（它编码了扫描 Y-up → 世界 Z-up，
实测 `R @ 局部+Y = [0,0,1]`），只解平移；桌面高度由参考姿态反推得 0.9633，
解出 `pos_z = -0.9543`，与参考的 `-0.9632` 只差 **9 mm**（交叉验证通过）。

结果 1/5、70%，与不带背景的 3/5、90% 相比 p=0.63（不显著）。**但这一臂没有真正测到该假说**，
因为有两个独立缺陷：

1. **渲染冲突（用户从视频里看出来的）**：runner 传了 `s15_eval.floor_plane_visible=true`，
   于是 Isaac Sim 在 z≈0 渲染了一块**不透明白色地面平面**；而扫描的桌面也被解到 **z=0**。
   两个共面不透明面 **z-fighting** → 画面基本全白，只在掠射角露出木色桌面斑块。
   参考场景正是因为带网格背景才设 `floor_plane_visible: false`。
2. **横向定位不可靠**：定位用的平面横向跨 **5.7×4.0 m**，那是**房间尺度**的面，
   不是桌面，所以它的质心不能当桌面中心。

**下一版应该用的方法**（比"基座差平移"更好，见 §7）：
把参考场景**已经被人对齐好的**桌面局部中心直接转移过来——
`local = Rᵀ·(参考道具质心 − 参考背景pos)`，再 `pos = 我们道具质心 − R·local`。
这不需要任何几何估计，是精确转移。三种方法算出来相差 9–14 cm：
A 基座差 `[-0.309, -0.168, -0.963]`、B 转移 `[-0.258, -0.246, -0.963]`、
C 点云质心 `[-0.150, -0.339, -0.954]`（就是跑过的那个）。
**并且在投入 20 分钟评测之前，先只导一帧看对齐。**

**渲染冲突已在代码里修掉**（无需再靠人记得）：`1_eval` 现在会在读完场景后检查它是否带背景对象
（`mesh_background*` / `gs_background*`），带背景就**强制** `floor_plane_visible=False` 并打印原因。
这条规则本来就在 pipeline 里——`14_create_og_scene.py:192-193` 写的是
`floor_plane_visible = not include_gs`、`use_skybox = not include_gs`——
只是 `s15_eval.floor_plane_visible` 之前会无条件覆盖它。
（`use_skybox` 故意不动：GS 背景**需要**天空盒，否则只渲染出约 4% 亮度，要求正好相反。）

**两条背景路线的区别**：
- **pipeline 原生**：`s14_og.include_gs=true`，背景来自
  `A_reconstruction/stages/auto_bg_reconstruction/`（7 个阶段：VOID 深度 → seed ply →
  训练 bg splat → 桥接到 OG → 构建场景资产），是**训练出来的、自动对齐的**。成本高（要训 splat）。
- **外挂资产**：像精选样例场景那样挂一个网格房间。pipeline **没有**网格背景的路径
  （`grep -rln mesh_background scripts/pipeline/` 为空）；这条只有编辑器有，
  正规入口是 `light_editor/background_io.py` 的 `attach_background()`
  （纯增量、会写 `expected_file_hash`、已有背景时会拒绝）。
  手动往 JSON 里塞 `mesh_background_0` 能跑，但不是正规做法。

---

## 7. 为什么"基座差平移背景"这个想法价值有限

用户对这个提议的怀疑是对的，详细分析：

**提议内容**：`pos = 参考场景背景pos + (我们的机器人基座 − 参考的机器人基座)`，
即 `[-0.309, -0.168, -0.963]`。

**它的隐含前提**：我们场景里道具相对桌面的排布，与参考场景一致。
**这个前提不成立**：
- 参考场景是**人工编排**的（编辑器里摆的），道具位置是人的选择；
- 我们的场景是**从视频重建**出来的，道具位置来自 A 阶段的重建流程，
  与我们后来把机器人基座从 -1.0 改到 -0.6 这件事**没有任何关系**——
  那个 -0.6 是**我自己**为了让道具落在任务声明的 `workspace_bounds` 里而选的参数。

**所以"基座差"只是"两堆道具质心恰好差得差不多"的巧合式启发，没有机制支撑。**
实测它与精确转移法 B 相差 9.4 cm——而 9 cm 在这个任务尺度上就是"桌面边沿 vs 桌面上"。

**真正有意义的是方法 B**，因为它转移的是参考场景**已经验证过的一组对齐关系**，
完全不需要测量扫描几何。方法 C（点云质心）失败的原因也很明确：它找错了平面。

**但更重要的两点**：
1. 即使位姿完美，**渲染冲突必须先修**（`floor_plane_visible` 在带网格背景时必须是 false），
   否则测的是"两个共面不透明面 z-fighting 的画面"，不是"真桌面"。
2. 位姿和渲染都修好之后，这个假说**仍然未被检验**——它可能有用、也可能有害
   （真实房间的杂乱背景也可能是干扰）。所以它现在是一个**待做的实验**，不是结论。

---

## 8. 最终推荐配置

```
task                  = droid/droid_desk_serve_fruits
prompt                = "put the yellow banana on the teal plate, then put the red apple on the teal plate"
wrist camera          = 1280x720   (已由 s14_og.robot_config.sensor_config 固化进场景)
external cameras      = nv_franka_droid.yaml 自带的 180x320
execute_horizon       = 8          (real2sim_cfg.yaml 默认；= openpi 官方 DROID 参考部署)
action_freq           = 15
timeout_s             = 120        (n_steps = 1800；配方 ServeFruitsOnGreenPlate.yaml:261 也是 120)
gripper_invert        = 默认(true)  ← 不要覆盖
end_effector          = robotiq    ← 不要改成 gripper，见 §3.6
init_states_path      = 固定布局，跨轮可比
scene_json            = 水果缩小变体（苹果 0.667、橙子 0.600），见 §3.7
floor_plane_visible   = true       (场景无背景时留着；有 mesh 背景会被自动关掉)
```

| 配置 | 成功 | 平均里程碑 |
|---|---|---|
| 官方精选场景 | **5/5** | **100%** |
| 桌子 + robotiq + 真实尺寸水果（`H3_5ep`，§11.1） | **5/5** | **100%** |
| 同上但用 panda hand + 原尺寸水果（反面对照） | **0/5** | **0%** |

---

## 9. 遗留问题与已知坑

- **`rollouts.hdf5` 的 `state` 不是可用的姿态向量**：里混着 `21211758.0` 这类非姿态值，
  6 个物体里只有 1 个能用 (pos, ori) 匹配定位。要逐帧追踪物体，应该像现有的
  `dump_layout` 那样在 `1_eval` 里加一个小 dump，而不是解析它。
  机器人 15 个自由度的关节值在 `state[27:42]`（DOF 顺序）。
- **`1_eval` 里的临时诊断（已清理一部分）**：`[DIAG]` 块（一次/轮，保留 —— 它一眼
  区分 robotiq 与 panda hand）、`[Debug] payload keys`（一次/轮，保留 —— 它暴露了
  `gripper: [1.]`）。`dump_layout()` 及其两次调用**已删**（布局早已验证到 4 位小数）。
  日志体积主要来自 `tqdm` 的 `\r` 刷新，不是这几条。
- **背景命名的历史不一致**（已修，但记下来）：`pick_place_task.py` 有四处按字面比较
  `"gs_background"` / `"mesh_background"`，而 light editor 写的是 `mesh_background_<n>`
  （`light_editor/background_io.py:45`，`BACKGROUND_OBJECT_NAME = "mesh_background_0"`）。
  所以**编辑器产出的场景全都**打印 "No background found, setting to None"，
  包括那个真的带桌面扫描的精选场景。现统一走
  `simfoundry.tasks.pick_place_task.is_background_object()`（前缀匹配）。
- **`s15_eval.scene_json` 传裸名字**会解析到
  `<ASSET_DIR>/scenes/<name>/<name>_scene_state_latest.json`，而 HF 的资产布局多一层
  `DROID/`。传绝对路径。
- **任务的 `workspace_bounds` 会在每次 reset 钳制非机器人组**
  （`pick_place_task.py:822`）。band 是**基座系**（存于 `:254`），
  `_compute_world_workspace_bounds`（`:479`）把 8 个角点转到世界系取 AABB 才交给
  `randomize_object_pose`，后者按世界系算区间
  `[max(-max_offset, lo-pos), min(+max_offset, hi-pos)]`（`task_utils.py:37-38`）。
  **区间不含 0 时物体每次 reset 被整体拖向边界，而不是原地抖动** —— 这是未 pin 布局时
  「布局看起来是随机的、其实被 band 决定」的原因。

  但**只有 4 个道具的最终位姿由这条钳制决定**：`pear` / `banana` / `apple` / `orange`。
  另两个（`orange_square_plate`、`teal_tray`）在 XYZ 随机化**之后**又被
  `group_predicate_placement` 重摆到香蕉旁（gap 1–5 cm，
  `place_with_predicate(..., bounds=world_bounds)`，`:878`），**覆盖掉随机化结果**，
  所以「板子被拖 40 mm」只是中间态，不是最终布局。且 legacy 谓词
  **只用 bounds 采次级轴、主轴不 clamp**（`placement_utils.py:29-31`），
  band 对板子的约束本就是部分的。

  实测（`/root/simfoundry_logs/test_workspace_bounds.py`，直接采样真函数、含基座系→世界系
  转换）。只统计最终位姿受 band 支配的那 4 个：

  | band | 最终布局被拖拽 | 最终基座系 x > 855 mm（臂展） |
  |---|---|---|
  | HEAD `x≤0.80, y≤0.05` | **pear 125 mm、banana 73 mm、orange 110 mm** | 0.0% |
  | 暂存的 `x≤0.95` | 无 | 1.4% |
  | 现用 `x≤0.86` | 无 | **0.4%** |

  HEAD 的缺陷是**每次 reset 把这 3 个道具整体挪 7–12 cm**。y 必须放宽的原因：本场景 6 个
  道具的基座系 y ∈ [-0.313, +0.192]，HEAD 的 y∈[-0.05, 0.05] 一个都装不下；已发布参考
  场景的 `teal_tray`（y=+0.282）与 `orange_square_plate`（y=-0.211）同样越界 ——
  是 band 对任务太窄，不是这个场景特殊。x 上限取 0.86 而非 0.95：参考场景的
  `orange_square_plate` 重建在基座系 900 mm，**本身就超出** Franka 的 855 mm 臂展。

  ⚠️ **AABB 表达不了径向臂展**：角点 (0.86, ±0.25) 径向 895 mm 仍超臂展；残余的 0.4%
  要降到 0 需要径向约束（改代码）。
  这条 band **只在未 pin 布局时生效**（`init_states_path` 为 null，`real2sim_cfg.yaml:541`）；
  H1/H3 都 pin 了，从没走过这条路径。端到端复核：
  `/root/simfoundry_logs/run_bandcheck.sh`（1 轮未 pin，exit=0，Stage 1 用时 157 s）。
- **视频帧率已改为派生（旧的手工换算已过时）**：`s15_eval.video_fps` 默认 `null`，
  由 `action_freq / video_frame_stride` 算出。`n_steps = timeout_s * action_freq`，
  而每 `video_frame_stride` 步采一帧，所以想 1:1 就必须用这个比值：
  h=8、120 s → **1800 步 / 226 帧 @ 1.875 fps = 120.5 s** ✓。
  旧代码写死 `video_fps: 10`，把 h=8 的录像放快了 **5.33 倍**（帧是对的，时钟是错的）。
  帧也改成流式写盘 + 结束时改名（`stride=1` 时 1350 帧 3840×720 ≈ 11 GB 会炸内存）。
  **已存在的旧视频**用 `/root/simfoundry_logs/realtime_videos.py` 重定时 ——
  必须用 `setpts` 拉伸、**不能**用 `fps` 重采样（后者会丢掉 169 帧里的 138 帧）；
  输出 `<名>_realtime.mp4` 与原件同目录并存。

- **相机分辨率与「对齐训练」无关**：openpi 无论输入多少一律
  `resize_with_pad(image, 224, 224)`（`src/openpi/models/model.py:47,166`，训练推理同一套）。
  DROID 三个相机也都是 320×180（`examples/droid/convert_droid_data_to_lerobot.py:134`
  连**腕部**都缩到 `(320, 180)`），而我们三个相机也全是 16:9，所以长宽比已一致，
  分辨率只影响降采样后的锐度。经验数据：腕部 720×1280 比 128×128 好
  （`picked_apple` 100% vs 0%），所以**唯一的偏离（腕部 720p）恰好是有利的**。
  ⚠️ 真正可能偏差的是 **FOV**：外部 104.0°（`nv_franka_droid.yaml`
  `focal 2.1 / aperture 5.376`），腕部 87.7°（`1_eval:676-681` 运行时覆盖为
  `focal 2.8 / aperture 5.376`，旁边留着注释掉的备选值）。这两个数看起来是试出来的，
  没有从 DROID 标定拄来；要改先查 DROID 公开规格。

- **task 定义管不住策略的动作**：`goal_predicates_all` 只要求香蕉和苹果 `OnTop` 青盘，
  `goal_predicates_any` 只要求结束时机械臂不碰这两样。策略多抓一个橙子放上去
  （H2 那集实际就是 香蕉→橙子→苹果），`success` 依旧是 `True`。
  失败集的机制是：第二个抓的是**梨**（非目标），`picked_apple` 永远不触发，
  耗满预算。可用的配置杠杆只有 **prompt**（配方用单句合取
  `"put both the banana and the apple on the green plate"`，不是两句顺序式）和
  **`timeout_s`**；`semantic_group_mapping` 里梨/橙子是显式声明的组，删不掉它们对策略的可见性。
- **HF 下载**：`git sparse-checkout` 会撞 HF 503；用 HF 的 resolve 端点 + `curl` 稳定。
  资产本体是 LFS 指针，必须显式拉取。

---

## 10. 复现命令

环境（每个 shell）：
```bash
cd /root/workspace/SimFoundry
export PATH=/root/miniforge3/bin:$PATH MAMBA_ROOT_PREFIX=/root/miniforge3
unset http_proxy https_proxy          # 需要 GitHub/HF 时先 source /etc/network_turbo
```

本地 openpi server（官方 `pi05_droid_jointpos_polaris`）：
```bash
cd /root/workspace/openpi
export PATH=/root/miniforge3/bin:/root/.local/bin:$PATH
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
export UV_PROJECT_ENVIRONMENT=/root/autodl-tmp/openpi_venv
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
    --policy.config=pi05_droid_jointpos_polaris \
    --policy.dir=/root/autodl-tmp/openpi_cache/pi05_droid_jointpos
```

重建场景（把腕部相机固化进产物；旧目录不会被改写）：
```bash
cd /root/workspace/SimFoundry
bash scripts/pipeline/A_reconstruction/run.sh \
    --scene-name fruits \
    --include 14 \
    -- 's14_og.robot_config.position=[-0.6,0.0,0.0]' \
       's14_og.robot_config.end_effector=robotiq' \
       's14_og.out_dirname=s14_og_robotiq_w720'
```
腕部 720×1280 来自 `real2sim_cfg.yaml` 的 `s14_og.robot_config.sensor_config`（已在仓库里），
不需要额外 override。机器人在 −0.6 是为了让道具落进任务声明的 `workspace_bounds`。
约 2 分钟；产物是一个约 19 KB 的 JSON（网格在 s13，不在这个目录里），
所以**不需要归档产物，保留 recipe 即可**。

评测（`/root/simfoundry_logs/run_eval_fruits.sh` 是薄包装，
支持 `S14_DIR` / `OPENPI_HOST` / `OPENPI_PORT` / `OPENPI_CHECKPOINT` /
`PROMPT` / `WRIST_SENSOR_RES` / `EXT_SENSOR_RES` / `SCENE_JSON` / `LOG` 覆盖）：

```bash
LOG=/tmp/eval.log OPENPI_CHECKPOINT=my_run \
S14_DIR=$PWD/Data/fruits/s14_og_robotiq_w720 \
OPENPI_HOST=localhost OPENPI_PORT=8000 \
bash /root/simfoundry_logs/run_eval_fruits.sh
```

---

## 11. 附录：完整实验结果

| 臂 | 场景 | prompt | 腕部 | 成功 | 平均里程碑 | [抓蕉, 放蕉, 抓苹, 放苹] |
|---|---|---|---|---|---|---|
| 对照（修前） | 精选 | 官方并列句 | 720 | 0/5 | 5% | [1, 0, 0, 0] |
| **对照（修后）** | 精选 | 官方并列句 | 720 | **5/5** | **100%** | **[5,5,5,5]** |
| p0w0 | fruits | 两段式 | 128 | 0/5 | 50% | [5,4,1,0] |
| p1w0 | fruits | 并列句 | 128 | 0/5 | 50% | [5,5,0,0] |
| **p0w1** | fruits | 两段式 | **720** | **3/5** | **90%** | **[5,5,5,3]** |
| p1w1 | fruits | 并列句 | 720 | 1/5 | 70% | [5,5,3,1] |
| p0_640 | fruits | 两段式 | 640 | 2/5 | 70% | [5,5,2,2] |
| **baked** | fruits | 两段式 | 720（固化） | **3/5** | **90%** | **[5,5,5,3]** |
| bg_droid_v1 | fruits | 两段式 | 720 + 背景 | 1/5 | 70% | [5,5,3,1] |

每个臂 n=5（由 `init_states_path` 固定为同一布局），同一 server、同一 prompt/horizon。
里程碑定义（`scripts/cfg/task/droid/droid_desk_serve_fruits.yaml:41-69`）：
`picked_banana` / `placed_banana_on_teal_plate` / `picked_apple` / `placed_apple_on_teal_plate`。

### 11.1 2026-09-18 的 H 系列（桌子场景 + robotiq + 缩小水果）

| 臂 | 场景 | 端末 | 水果 | 预算 | 成功 | 平均里程碑 | 平均步数 |
|---|---|---|---|---|---|---|---|
| `T_table` | 桌子 | **panda hand** | 原尺寸 | 1350 | **0/5** | **0%** | 1350（全部超时）|
| `H1` | 桌子 | robotiq | 苹果 0.667 / 橙 0.600 | 1350 | **2/3** | **83.3%** | 737.7 |
| `H2` | 桌子（无白地板）| robotiq | 同上 | 1350 | **1/1** | **100%** | 409 |
| **`H3`** | 桌子 | robotiq | 同上 | **1800** | **5/5** | **100%** | **325.6** |

`T_table` → `H1` 之间只改了**两件事**：换回 robotiq（§3.6）与缩小两个球状水果（§3.7）。
`H1` → `H3` 只改了 `timeout_s: 90 → 120`。

⚠️ **不能把 `H3` 的 5/5 归因于 `timeout_s`**：两次的前 3 条 init state 完全相同
（最大位置差 `0.00e+00`），`H3` 三集只要 249/260/339 步，而 `H1` 同三集是 347/516/1350；
加长预算只会抬高上限，不可能让策略**更快**。剩下的差异是运行间方差 —— 这也是为什么
`H3` 要跑 5 集。更硬的结论需要重复跑（n≥20 才能分辨 <40 pp 的差异）。

`H3` 逐集：249 / 260 / 339 / 498 / 282 步，全 `success=True`、`progress=1.00`。

---

## 12. 仓库外的关键产物与启动参数

> 本节专门记录**不在 git 里**、但重跑实验必需的路径与参数。它们都在 `/root/` 下，
> 换机器/换实例需要找齐的三个根：`/root/workspace/openpi`、`/root/autodl-tmp/`、
> `/root/simfoundry_logs/`。

### 12.1 根目录与链接

```
/root/workspace/SimFoundry          # 本 repo（分支 junjie/work_branch）
  Data  -> /root/autodl-tmp/simfoundry/Data          # 流水线产物
  deps  -> /root/autodl-tmp/simfoundry/deps          # BEHAVIOR-1K / OmniGibson（不受版本控制）
  assets-> /root/autodl-tmp/simfoundry-assets/assets # 资产包
/root/workspace/openpi              # openpi 源码（server 就跑在这里）
/root/simfoundry_logs/              # 所有实验脚本、固定布局、日志、图片
```

⚠️ `/root/autodl-fs` 是**跨实例只读**，只能读，不要写。

### 12.2 openpi server（重跑必启）

```bash
cd /root/workspace/openpi
export PATH=/root/miniforge3/bin:/root/.local/bin:$PATH
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
export UV_PROJECT_ENVIRONMENT=/root/autodl-tmp/openpi_venv
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
    --policy.config=pi05_droid_jointpos_polaris \
    --policy.dir=/root/autodl-tmp/openpi_cache/pi05_droid_jointpos
```
- `--port` **必须在 `policy:checkpoint` 之前**（tyro 解析顺序）。
- 检查点：`/root/autodl-tmp/openpi_cache/pi05_droid_jointpos`（assets 在 `assets/droid/`）。
- 验证存活：`pgrep -af serve_policy.py`；预热一次推理（首次编译很慢）。
- ⚠️ `--policy.config` **静默决定动作空间**（这里必须是与权重配套的 `*_jointpos_polaris`），
  写错不报错、只会得分崩。

### 12.3 模拟器/工具环境（各阶段不同）

| 用途 | env | 启动方式 |
|---|---|---|
| A 阶段 / C 阶段评测（OmniGibson + Isaac Sim） | `simfoundry` | `mamba run -n simfoundry python ...` |
| 只读 USD（`pxr`）、light editor | `simfoundry-editor` | **无 matplotlib** |
| 点云/深度诊断（open3d、numpy） | `simfoundry` | `/root/autodl-tmp/simfoundry/conda_envs/simfoundry/bin/python`；**无可用 pxr** |
| DA3 深度 | `da3` | |

标准块：
```bash
export PATH=/root/miniforge3/bin:$PATH MAMBA_ROOT_PREFIX=/root/miniforge3
unset http_proxy https_proxy        # 需要外网时先 source /etc/network_turbo，用完必须 unset
export OMP_NUM_THREADS=8            # 不设 open3d 会报 nthreads must be a positive integer
```

### 12.4 场景变体（都在仓库外，`Data/**` 保持只读）

| 路径 | 内容 |
|---|---|
| `/root/autodl-tmp/simfoundry/scene_variants/fruits_table_robotiq_smallfruit/reconstructed_og_scene.json` | **主用**：桌子 + robotiq + 苹果 0.667/橙子 0.600 |
| `.../scene_variants/fruits_apple070/reconstructed_og_scene.json` | 下午的苹果 0.70 变体 |
| `.../scene_variants/bg_wood_table/bg_wood_table_scene_state_latest.json` | 木桌 mesh 背景（带背景时 `1_eval` 会自动关掉白地板）|
| `.../scene_variants/fruits_droid_v1_bg.json` | fruits + droid 桌面扫描背景（当时未验证成功）|
| `/root/workspace/SimFoundry/Data/fruits/s14_og_table_robotiq/` | **主用**桌子场景（`end_effector=robotiq`）|
| `/root/workspace/SimFoundry/Data/fruits/s14_og_robotiq_w720/` | 下午全部成功臂用的无桌子场景 |

### 12.5 固定布局（`init_states_path`，决定可比性）

| 文件 | 集数 | 用途 |
|---|---|---|
| `/root/simfoundry_logs/init_states_table_robotiq_5.json` | 5 | **H3 用的** |
| `/root/simfoundry_logs/init_states_table_robotiq_3.json` | 3 | H1 |
| `/root/simfoundry_logs/init_states_table_robotiq_1.json` | 1 | H2 |
| `/root/simfoundry_logs/init_states_fruits_recon_5.json` | 5 | 下午各臂 |
| `/root/simfoundry_logs/init_states_fruits_recon_5.robotnear_backup.json` | 5 | 上者的备份 |

生成：`mamba run -n simfoundry python /root/simfoundry_logs/make_init_states.py <场景JSON> --out <输出> --episodes N`

### 12.6 实验脚本（`/root/simfoundry_logs/`）

| 脚本 | 作用 |
|---|---|
| `run_eval_fruits.sh` | 评测薄包装；支持 `S14_DIR`/`OPENPI_HOST`/`OPENPI_PORT`/`OPENPI_CHECKPOINT`/`PROMPT`/`WRIST_SENSOR_RES`/`EXT_SENSOR_RES`/`SCENE_JSON`/`LOG` 覆盖。⚠️ 里面的 `INIT_STATES` 是硬赋值，**不能**用环境变量覆盖 |
| `run_eval_H1.sh` / `run_eval_H2_nofloor.sh` / `run_eval_H3.sh` | 本轮的 H 系列启动器（直接调 `run_application.py`，可传 `init_states_path`）|
| `make_init_states.py` | 从场景 JSON 生成固定布局 |
| `make_scene_variant.py` | 改单物体 `args.scale` 做变体 |
| `make_variant_init_states.py` | 布局变体（swap / move）|
| `realtime_videos.py` | 视频重定时为 1:1（`setpts`，不重采样），原件保留 |
| `summarize_eval.py` | 逐里程碑汇总 `eval_results.json` |
| `sheet.py` / `frames.py` / `grid.py` | 抽帧、拼图 |
| `test_finger_dofs.py` | 夹爪分类器的**真测试**（`ast` 抽源码 + 伪造 robot）|
| `diag2.py`–`diag6.py` | A 阶段尺寸偏差诊断（掩码/深度/置信度）|
| `check_reach.py` / `diff_scenes.py` | 可达性、场景逐项差异 |

日志：`eval_table.log`（0/5 那次）、`eval_H1_table_robotiq_smallfruit.log`、
`eval_H2_nofloor.log`、`eval_H3.log`。
图片：`floor_on_vs_off.png`（白地板严格对照）、`h1_ok_cam1.png`、`h2_cam1.png`。

### 12.7 评测结果目录命名

```
Data/fruits/s15_eval/openpi/<checkpoint>/results_<timestamp>/
    eval_results.json        # 含 grasp_feasibility、逐里程碑成功率
    rollouts.hdf5            # 逐帧 state/action/reward/milestones
    videos/episode_XXX_{success,fail}.mp4        # 3 相机并排，1:1 实时
    videos/cameras/{ext_0,ext_1,wrist}/...
```
