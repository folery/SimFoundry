<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gemini 模型替换方案与 A/B 测试设计

> 目标：把 pipeline A 中所有**远程 Pro 模型**调用替换为 **Flash**（降本），并给出可复现的 A/B 测试方法。
> 状态：**规划文档 + A/B 执行记录**。方案与测试设计见 §1–§7；**实际执行的变体与结果见 §9**。
> 撰写日期：2026-09-15
> 相关：[`RESUME_INSTALL.md`](RESUME_INSTALL.md)（环境部署与恢复手册，与本文件互补）
>
> 本目录 `reproduce_docs/` 收集**本仓库原本没有、为复现而新增**的文档 —— 与上游 repo 内容分开管理。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 一次默认 A 跑有几个远程 Pro 调用点？ | **8 个配置旋钮**（其中 1 个自动继承，实际需 6 个 override）|
| 换成 Flash 要改代码吗？ | **不需要**。`gemini-2.5-flash` / `gemini-2.5-flash-image` 已同时满足两道校验闸门 |
| 想用"更新的" flash 呢？ | **文本类**可零改动升到 `gemini-3-flash-preview`；**图像类**原无零改动选项，现已在 3 处注册 `gemini-3.1-flash-image` / `-flash-lite-image` 并验证通过（§9.7）|
| 图像类换 3.1 代 flash 的结论？ | ✅ **`gemini-3.1-flash-image` 可用**（2/2 无幻影、26m41s vs 基线 41m14s）；❌ **`-flash-lite-image` 不建议**（更慢 31m57s、回填最不守掩码）。⚠️ **但回填忠实度弱于 `pro-image`：抹除处偏暗 ~8 级**（§9.7⑪）|
| 最该先补的代码改动是什么？ | **token/成本统计** —— 仓库里完全没有，导致 A/B 无法回答"省了多少" |
| A/B 的基线选哪个？ | `put_cup_in_bowl_outline`（Pro + `outline` 提示词），**不是** `put_cup_in_bowl` |
| 最容易踩的坑 | 重跑上游会因 `frame_selection` 用 VLM 选帧而**换帧**，导致 A/B 不是同一输入 |

---

## 1. 现状审计：谁在用远程模型

### 1.1 审计方法

`Data/<scene>/<stage>/stage_info.json` **记录了每个 stage 解析后的完整配置**。因此"某次跑到底用了什么模型"可以事后从产物反推，不依赖运行日志：

```bash
python3 -c "
import json,glob,os
for f in sorted(glob.glob('Data/<scene>/*/stage_info.json')):
    d=json.load(open(f))
    stage=os.path.basename(os.path.dirname(f))
    for k in ('detection_model','removal_model','vlm_model','front_pick_model',
              'validity_model','shape_model','texture_model','model'):
        if k in d: print(f'{stage:14s} {k:20s} {d[k]}')
"
```

这也是 A/B 测试的自证机制（见 §4.4①）。

### 1.2 实际生效的旋钮（一次默认 A 跑）

| # | 配置键 | 默认值 | 代码位置 | 任务类型 | 每场景调用（实测）|
|---|---|---|---|---|---|
| 1 | `s3_ground.frame_selection.vlm_model` | `null` → 继承 #2 | `simfoundry/pipeline/frame_selection.py:572` | 图 → 文本 | ~1（`mode: hybrid`）|
| 2 | `s3_ground.detection_model` | `gemini-3.1-pro-preview` | `stages/3_segment_ground_plane.py:57` | 图 → 文本 | 1-3 |
| 3 | `s5_scene.detection_model` | `gemini-3.1-pro-preview` | `stages/5_decompose_scene.py:866` | 图 → **JSON** | **4** |
| 4 | `s5_scene.removal_model` | `gemini-3-pro-image` | `stages/5_decompose_scene.py:876,882` | **图 → 图** | **4**（1 源图 + 3 移除）|
| 5 | `s6_upsample.model` | `gemini-3-pro-image` | `stages/6_upsample_object_images.py:87,99` | **图 → 图** | **3** |
| 6 | `s8_pose.front_pick_model` | `gemini-3.1-pro-preview` | `simfoundry/pipeline/front_canonicalization.py:193` | 多图 → 标签 | ~3（每物体 1）|
| 7 | `s11_sim.vlm_model` | `gemini-3.1-pro-preview` | `stages/11_make_objects_sim_ready.py:276` | 图 → JSON | ~3（每物体 1）|
| — | `s6_upsample.validity_model` | `gemini` | `stages/6_upsample_object_images.py:153` | — | **0**（见下）|

**#1 无需单独设置** —— `frame_selection.py:174-175` 实现了继承：

```python
if out["vlm_model"] is None:
    out["vlm_model"] = OmegaConf.select(cfg, "s3_ground.detection_model")
```

**`s6_upsample.validity_model` 是死配置**：值为 `gemini` 时**硬编码**构造 `gemini-3-pro-preview`，其他值直接 `raise NotImplementedError`；且 `check_valid: false`，**从不实例化**。

**本地模型（不在替换范围）**：`s7_mesh.{shape,texture}_model: hunyuan`。

**`s9_articulate_objects.*`（4 个旋钮）仅在 `--detect-articulation` 时启用**，默认不跑。

### 1.3 两道校验闸门

配置里的模型名要同时通过两处校验：

```python
# 闸门1：stage 级注册表（仅部分旋钮有）
assert_valid_key(key=removal_model_name, valid_keys=REMOVAL_MODELS, ...)
# 闸门2：Gemini 构造函数，所有调用点都过
assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Gemini model")
```

```
DETECTION_MODELS = {gemini-2.5-pro, gemini-2.5-flash, gemini-3-pro-preview,
                    gemini-3.1-pro-preview, gemini-3-flash-preview}     # 5_decompose_scene.py:516
REMOVAL_MODELS   = {gemini, gemini-2.5-flash-image, gemini-3-pro-image, flux}
                                                                       # 5_decompose_scene.py:509
UPSAMPLE_MODELS  = {gemini, gemini-2.5-flash-image, gemini-3-pro-image, gpt, flux}
                                                                       # 6_upsample_object_images.py:40
```

| 旋钮 | 闸门1 | 闸门2 |
|---|---|---|
| #2 #3 `detection_model` | `DETECTION_MODELS` | `VERSIONS` |
| #4 `removal_model` | `REMOVAL_MODELS` | `VERSIONS` |
| #5 `s6_upsample.model` | `UPSAMPLE_MODELS` | `VERSIONS` |
| #1 #6 #7 | **无** | `VERSIONS` |

**这两个闸门是"能不能换"的真正边界。** 中转提供的模型比 `VERSIONS` 多，多出来的必须在 `VERSIONS` 里注册才能用。

---

## 2. 可用模型与"更新"选项

### 2.1 中转实测清单

`GET https://api.ofox.io/gemini/v1beta/models`（需 `x-goog-api-key` 或 `Authorization: Bearer`）返回 **16 个**：

```
gemini-2.5-flash              gemini-2.5-flash-image       gemini-2.5-flash-lite    gemini-2.5-pro
gemini-3-flash-preview        gemini-3-pro-image           gemini-3.1-flash-image   gemini-3.1-flash-lite
gemini-3.1-flash-lite-image   gemini-3.1-pro-preview       gemini-3.5-flash         gemini-3.5-flash-lite
gemini-3.6-flash              gemini-3.7-flash             gemini-3.8-flash         gemini-embedding-2-preview
```

逐模型 `generateContent` 实测（"2+2"）：

| 模型 | 在 `VERSIONS` | HTTP | 结果 |
|---|---|---|---|
| `gemini-2.5-flash` | ✅ | 200 | `text='4'` finish=STOP |
| `gemini-3-flash-preview` | ✅ | 200 | `text='4'` finish=STOP |
| **`gemini-3.1-flash-image`** | ❌ | 200 | `text='4'` finish=STOP |
| `gemini-3.8-flash` | ❌ | 200 | `text='4'` finish=STOP |
| `gemini-2.5-pro`（基线）| ✅ | 200 | `text='4'` finish=STOP |

> **注意**：这些模型默认开启思考。用小 `maxOutputTokens` 测试时，`thoughtsTokenCount`（实测 36~115）会把配额吃光并返回**空文本** —— 容易误判为"模型不可用"。

### 2.2 零代码改动可用的最新组合

| 用途 | 零改动最新 | 说明 |
|---|---|---|
| 文本类（#1 #2 #3 #6 #7）| **`gemini-3-flash-preview`** | 已在 `VERSIONS` 与 `DETECTION_MODELS` |
| 图像类（#4 #5）| **只有 `gemini-2.5-flash-image`** | `VERSIONS` 里**唯一**的 flash 图像模型 |

**换句话说：文本类可以零改动升级到 3 系；图像类想用更新的，绕不开改代码。**

> **2026-09-15 更新**：上表描述的是**改动前**的状态。图像类的两个 3.1 代 id
> （`gemini-3.1-flash-image` / `gemini-3.1-flash-lite-image`）现已按 §9.7 的
> **3 处改动**注册完毕，并已实测可用（无幻影，见 §9.7⑩⑪）。文本类那一行仍然成立。

### 2.3 关于中转上的 `3.5/3.6/3.7/3.8-flash`

实测 `gemini-3.8-flash` 返回正确结果，**但这个命名与官方公开型号序列不符**，第三方中转很可能在做别名映射 —— 请求 `3.8-flash` 时背后实际运行的模型不可知。

**建议**：这类模型即使注册进 `VERSIONS` 也**不要用于生产**，行为不可预期。相对可信的是与官方命名一致的 `gemini-3-flash-preview` / `gemini-3.1-flash-image`。

---

## 3. 替换方案

### 3.1 修改前后模型对比

| 阶段 | 旋钮 | **修改前** | 批次 A 后 | 批次 B 后 |
|---|---|---|---|---|
| 3 | `frame_selection.vlm_model` | `null`→继承 | 继承（自动跟着变）| 同 |
| 3 | `s3_ground.detection_model` | `gemini-3.1-pro-preview` | `gemini-2.5-flash` | 同 |
| 5 | `s5_scene.detection_model` | `gemini-3.1-pro-preview` | `gemini-2.5-flash` | 同 |
| 5 | `s5_scene.removal_model` | `gemini-3-pro-image` | 不变 | **`gemini-2.5-flash-image`** |
| 6 | `s6_upsample.model` | `gemini-3-pro-image` | 不变 | **`gemini-2.5-flash-image`** |
| 8 | `s8_pose.front_pick_model` | `gemini-3.1-pro-preview` | `gemini-2.5-flash` | 同 |
| 11 | `s11_sim.vlm_model` | `gemini-3.1-pro-preview` | `gemini-2.5-flash` | 同 |
| — | `s6_upsample.validity_model` | `gemini`（硬编码 pro）| 不变（不调用）| 同 |
| 7 | `s7_mesh.{shape,texture}_model` | `hunyuan`（本地）| **不动** | **不动** |

**Pro 调用点：8 → 0**（A+B 之后）。

**若采用"更新的"方案**：文本类换 `gemini-3-flash-preview`；图像类换 `gemini-3.1-flash-image`（需代码改动 #2）。

### 3.2 批次 A：文本类（5 个旋钮，4 个 override）

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name put_cup_in_bowl_ab01_det25 \
  --video-fpath <abs>/PutCupInBowl.mp4 \
  -- \
  s3_ground.detection_model=gemini-2.5-flash \
  s5_scene.detection_model=gemini-2.5-flash \
  s8_pose.front_pick_model=gemini-2.5-flash \
  s11_sim.vlm_model=gemini-2.5-flash
```

`#1 frame_selection` 自动继承，无需列出。

### 3.3 批次 B：图像类（2 个旋钮）

```bash
  -- s5_scene.removal_model=gemini-2.5-flash-image \
     s6_upsample.model=gemini-2.5-flash-image
```

### 3.4 为什么分两批

风险性质不同，不是"能不能换"的问题：

| | 批次 A：文本类 | 批次 B：图像类 |
|---|---|---|
| 坏了的症状 | 检测质量下降、框偏移 → **可从 `stage_info.json` + `obj_cat_list` 看出** | **回填出现假象** → 下游多出幻影物体 |
| 风险 | 🟡 中 | 🔴 高 |

**批次 B 的高风险有实证**：默认的 `obj_removal_prompt_order: ["mask","outline","bbox"]` 中，`mask` 提示词只说 "Replace the bright green pixels with empty air"，**不要求重建被遮挡背景**，导致回填产出透明棋盘格/色块，并在每 3 轮的强制重检测中被 VLM 认成新物体，使场景多出一个物体。把 `outline` 提到首位可消除该问题（实测）。

因此批次 B 应**同时固定** `s5_scene.obj_removal_prompt_order='["outline","bbox","mask"]'`。

---

## 4. A/B 测试设计

### 4.1 基线选择

**基线用 `put_cup_in_bowl_outline`（Pro + `outline` 提示词），不是 `put_cup_in_bowl`。**

| 场景 | 提示词 | 模型 | 角色 |
|---|---|---|---|
| `put_cup_in_bowl` | `["mask",...]` | Pro | 旧基线（有幻影）|
| **`put_cup_in_bowl_outline`** | `["outline",...]` | **Pro** | **模型 A/B 的基线** |

在"已修复提示词"的配置上只动模型，变量才是干净的。**两个场景都不删除。**

### 4.2 变体矩阵

命名规则：`<base>_ab<NN>_<slug>`（序号保证可排序，slug 说明变量）。

| 变体 | 场景名 | 文本类 | 图像类 | 目的 |
|---|---|---|---|---|
| 基线 | `put_cup_in_bowl_outline`（已有）| pro-3.1 | pro-image | 参照 |
| A1 | `put_cup_in_bowl_ab01_det25` | 2.5-flash | pro-image | 只换文本类（最低风险）|
| A2 | `put_cup_in_bowl_ab02_det3` | 3-flash-preview | pro-image | 更新的文本模型 |
| B1 | `put_cup_in_bowl_ab03_img25` | pro-3.1 | 2.5-flash-image | 只换图像类（高风险）|
| B2 | `put_cup_in_bowl_ab04_img31` | pro-3.1 | 3.1-flash-image | 依赖代码改动 #2 |
| C | `put_cup_in_bowl_ab05_all25` | 2.5-flash | 2.5-flash-image | 全换 |

**必须固定的四个量**：`obj_removal_prompt_order`、上游产物、`seed`、其余全部配置。

### 4.3 两阶段执行

#### 阶段一：快速迭代（~4 分钟/变体）

**机制**：`--include 5` 时流式窗口条件不成立 ——

```python
# simfoundry/pipeline/orchestrator.py:491
window_ok = (... and spec.stage_id == "5"
            and i + 4 <= len(selected)
            and [selected[i+j].stage_id for j in range(4)] == ["5","6","7","8"])
```

选中集合只有 `5` 时该条件为假 → **stage 5 以单阶段方式运行**，不被流式接管。stage 5 的全部输入来自场景目录（`s1_video` / `s2_da` / `s3_ground` / `s4_frame`）。

```bash
V=Data/put_cup_in_bowl_ab01_det25
mkdir -p $V
for d in s1_video s2_da s3_ground s4_frame; do
  cp -r Data/put_cup_in_bowl_outline/$d $V/
done

bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name put_cup_in_bowl_ab01_det25 \
  --video-fpath <abs>/PutCupInBowl.mp4 \
  --include 5 \
  -- s5_scene.obj_removal_prompt_order='["outline","bbox","mask"]' \
     s3_ground.detection_model=gemini-2.5-flash \
     s5_scene.detection_model=gemini-2.5-flash
```

要顺带验 stage 6 就用 `--include 5,6`（同样不被流式接管）。

#### ⚠️ 两个必须遵守的约束

**① 上游产物必须复用，不能重跑 1b-4。**
`s3_ground.frame_selection` 是**用 VLM 选帧**的（`mode: hybrid`）。重跑可能选出**不同的帧**，A/B 就不是同一输入了。这是最容易踩的坑。

**② 用 `cp` 而不是软链。**
`frame_selection.json` 与 `frame_selection.png` 是**写进 `s3_ground/` 的**，软链有写回并污染基线的风险。磁盘紧张时软链**仅限** `--include 5,6`（此时上游无写入）。

#### 阶段二：端到端确认（~22-41 分钟/变体）

对筛出的胜出者跑**完整 14 阶段**，确认 `s11_sim/objects/` 仍为 3 个物体。

### 4.4 对比指标（全部来自现成产物）

| 维度 | 文件 | 指标 | 判据 |
|---|---|---|---|
| **① 变量唯一性** | `*/stage_info.json` | 与基线做 key 级 diff | **必须只有预期的键不同** —— 这是自证机制 |
| **② 检测质量** | `s5_scene/obj_cat_list/iter_*.json` | 迭代数、`is_valid_removed_obj`、`valid_pixel_proportion`、`removed_obj_phrase` | 迭代数 3；`valid` 全 True；`prop` 接近 1.0 |
| **③ 检测稳定性** | `s5_scene/skipped_iterations/` | 跳过轮数 + `reason` | 与基线一致 |
| **④ 掩码一致性** | `s5_scene/removal_mask/iter_*.png` | 与基线 **IoU** | 骤降说明检测/分割退化 |
| **⑤ 回填质量** | `s5_scene/post_object_removal/iter_*.png` | 背景环亮度误差（见下）| 不应显著回升 |
| **⑥ 端到端** | `s11_sim/objects/` | 物体个数 + 名称集合 | 仍为 3 个 |
| **⑦ 成本** | 需先做代码改动 #1 | 各模型 token 总数 | "省了多少" |

**⑤ 的算法**（实测可用，无需人工看图）：

1. 取 `s5_scene/removal_mask/iter_N.png` 得到掩码 $M$
2. 对 $M$ 做半径 25 的膨胀再减去 $M$，得到"背景环" $R$
3. 以 `s5_scene/source_padded_resized.png` 在 $R$ 上的灰度均值作为基准 $\mu_{bg}$
4. 比较 `post_object_removal/iter_N.png` 在 $M$ 内的均值 $\mu_{fill}$，指标为 $|\mu_{fill} - \mu_{bg}|$

参考量级（Pro + `outline` 基线）：`put_cup_in_bowl` iter_1 的 $|\mu_{fill}-\mu_{bg}| = 0.9$；而旧 `mask` 提示词为 $105.1$。

### 4.5 建议的对比脚本（设计，未实现）

**输入**：一组场景目录
**输出**：一张可读表 + 一份机器可读 JSON

```
对每个变体：
  1. 读全部 stage_info.json，与基线做 key 级 diff → 本变体改动的配置键（验证唯一性）
  2. 读 s5_scene/obj_cat_list/*.json        → 迭代数 / valid / prop / 物体名
  3. 读 s5_scene/removal_mask/*.png vs 基线 → IoU
  4. 读 s5_scene/post_object_removal/*.png  → 背景环亮度误差（§4.4⑤）
  5. 读 s11_sim/objects/                    → 物体集合
  6. （代码改动 #1 后）读 token 统计         → 总数
打平成一张表，按变体名排序输出
```

**建议做**：7 个变体手工比对必然出错，而 §4.4 的指标全部现成，脚本化成本很低。

**不建议**做成"自动跑 + 自动比"一体脚本 —— 跑要花钱，应由人逐个决定。

---

## 5. 值得讨论的代码改动

| # | 改动 | 规模 | 收益 | 建议 |
|---|---|---|---|---|
| **1** | **token/成本统计** | ~20-40 行 | **A/B 的目的是降本，但仓库里完全没有 token 统计** —— `usageMetadata` 从未被读取。`Gemini.__call__` 里 response 带 `promptTokenCount` / `candidatesTokenCount` / `thoughtsTokenCount`，累加后落盘即可 | ⭐ **强烈建议** |
| **2** | ~~注册 `gemini-3.1-flash-image`~~ | **3 行**（`VERSIONS` +1、`REMOVAL_MODELS` +1、`UPSAMPLE_MODELS` +1）| 唯一让图像类用上新一代 flash 的路径 | ✅ **已于 2026-09-15 实施**（§9.7⑧），两个 id 均注册；**已跑 A/B 验证**（§9.7⑩⑪） |
| 3 | 解 `s6_upsample.validity_model` 硬编码 | ~5 行 | ≈0（`check_valid: false`，从不实例化）| 不做 |
| 4 | 解耦 `s3_ground.detection_model` 与 `frame_selection.vlm_model` | ~5 行 | 小（一个 override 已够）| 不做 |

**#1 为何排第一**：跑完一次 A 之后，当前无法回答"这次花了多少、花在哪"。没有这个数据，A/B 只能比质量、比不了钱 —— 而那正是做这件事的理由。

**#2 只有 3 行**，是"想用更新的模型"这个诉求的唯一低成本出口。但换新图像模型后**回填质量必须重新验证**。

> ✅ **已按此要求验证**（2026-09-15）：实际改动是 **3 处共 5 行**，因为要一次注册**两个** id
> （`gemini-3.1-flash-image` 与 `gemini-3.1-flash-lite-image`）。回填质量见 §9.7⑪ ——
> 结论是**弱于 `pro-image`（抹除处偏暗 ~8 级）**，比 2.5 代 flash 图像模型也略差，
> 但**未产生幻影**。所以"必须重新验证"这条判断是对的：只看"有没有幻影"会漏掉这个退化。

---

## 6. 实施顺序建议

| 步骤 | 做什么 | 理由 |
|---|---|---|
| 1 | 先做**代码改动 #1（token 统计）** | 没它 A/B 回答不了"省了多少" |
| 2 | 跑 **A1（文本类 → 2.5-flash）** | 零改动、风险最低；`frame_selection` 本身 fail-soft；`front_pick_model` 的**代码默认值本来就是 `gemini-2.5-flash`** |
| 3 | 对比 A1 vs 基线 | 用 §4.4 的表 |
| 4 | 再考虑 A2（`gemini-3-flash-preview`）| 零改动 |
| 5 | 最后才碰图像类（B1）| 唯一验证过会出幻影的一环 |
| 6 | 若需新一代图像模型 → 代码改动 #2 | 唯一出口 —— ✅ **已执行**（2026-09-15），结果见 §9.7 |

---

## 7. 已知坑清单

1. **`removal_model: gemini` 不是 flash** —— 代码里是别名，仍构造 `gemini-3-pro-image`（`5_decompose_scene.py:876`）。改成 `"gemini"` 等于没改。
2. **中转模型数 > `VERSIONS` 条目数** —— 多出来的模型实测能跑但会被构造函数 `assert` 挡住。
3. **`s6_upsample.validity_model` 换不了** —— 硬编码；但 `check_valid: false`，默认不实例化，不影响。
4. **思考 token** —— flash 模型默认开思考，自定义调用要给足 `maxOutputTokens`；管线内部用 `VERSIONS[model]["max_tokens"]`（`gemini-2.5-flash` = 65535），不受影响。
5. **重跑上游会换帧** —— `frame_selection` 用 VLM 选帧，A/B 必须复用上游产物。
6. **软链上游有污染风险** —— `s3_ground/` 会被写入。
7. **远程调用已产生真实失败** —— 日志中 stage 8 的 `front_pick_model` 曾 `failed after 3 attempts: The read operation timed out`，靠 fail-soft 降级（`front_canonicalization.py:203`）才未中断。换 flash 延迟更低，这类超时会减少。

---

## 8. 附录：本次审计的取证命令

```bash
# 一次跑实际用了什么模型（从产物反推，不依赖日志）
python3 -c "
import json,glob,os
for f in sorted(glob.glob('Data/<scene>/*/stage_info.json')):
    d=json.load(open(f)); s=os.path.basename(os.path.dirname(f))
    m={k:v for k,v in d.items() if 'model' in k}
    if m: print(s, m)
"

# A/B 变量唯一性证明
diff <(python3 -m json.tool Data/<base>/s5_scene/stage_info.json) \
     <(python3 -m json.tool Data/<variant>/s5_scene/stage_info.json)

# 中转可用模型
curl -sS -H "x-goog-api-key: $GEMINI_API_KEY" \
  https://api.ofox.io/gemini/v1beta/models | python3 -m json.tool
```

---

## 9. 执行记录：2026-09-15 的 A/B 运行

### 9.1 决策与范围

按讨论结论确定：

- **不改代码** —— 只用品命令行覆盖（成本/token 统计暂缓，中转站侧可看）。
- **只做 Flash 替换** —— Qwen-Image 本地化、stage 7 换 mesh 后端都推迟。
- **对比脚本不写** —— 结果落盘，人工比对。
- **`Data/` 下已有数据全部保留**，不删除、不覆盖。

### 9.2 变体矩阵（实际执行）

**基线 = `put_cup_in_bowl_outline`**（Pro + `outline` 提示词），**已存在，不重跑**。
所有变体都固定 `s5_scene.obj_removal_prompt_order=["outline","bbox","mask"]`，**只动模型旋钮**。

| 顺序 | 场景名 | 文本类（4 旋钮）| 图像类（2 旋钮）| 隔离什么 |
|---|---|---|---|---|
| — | `put_cup_in_bowl_outline` | pro-3.1 | pro-image | 基线（已有）|
| 1 | `put_cup_in_bowl_ab01_det25` | **`gemini-2.5-flash`** | pro-image | 只换文本类 |
| 2 | `put_cup_in_bowl_ab03_img25` | pro-3.1 | **`gemini-2.5-flash-image`** | 只换图像类 |
| 3 | `put_cup_in_bowl_ab05_all25` | `gemini-2.5-flash` | `gemini-2.5-flash-image` | 全换（目标终态）|
| 4 | `put_cup_in_bowl_ab02_det3` | **`gemini-3-flash-preview`** | pro-image | 更新的文本模型 |

执行顺序按价值排（**中途中断也能拿到最有用的结果**）。

**每个变体的确切命令**（4 个文本旋钮同时覆盖 5 个配置键，因为
`frame_selection.vlm_model` 自动继承 `s3_ground.detection_model`）：

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name <变体名> \
  --video-fpath /root/workspace/SimFoundry/docs/assets/example_videos/PutCupInBowl.mp4 \
  -- \
  's5_scene.obj_removal_prompt_order=["outline","bbox","mask"]' \
  s3_ground.detection_model=<文本模型> \
  s5_scene.detection_model=<文本模型> \
  s8_pose.front_pick_model=<文本模型> \
  s11_sim.vlm_model=<文本模型> \
  s5_scene.removal_model=<图像模型> \
  s6_upsample.model=<图像模型>
```

### 9.3 运行方式

- 执行脚本：`/root/simfoundry_logs/run_ab_flash.sh`（**仓库外**，不含密钥）
- 日志：`/root/simfoundry_logs/run_ab_flash.log`（每变体独立段落，含实际使用的全部覆盖项）
- 监控：`/root/simfoundry_logs/watch_ab_flash.py` → 状态文件
  `/root/simfoundry_logs/ab_flash_status.txt`（每 60 秒刷新）
- 后台方式：`nohup setsid`，**PPID=1**，SSH 断开不影响
- 预估总时长：4 个变体 × ~22-41 分钟 ≈ **1.5-2.7 小时**

### 9.4 怎么看结果（人工比对）

**先验证变量唯一性**（最重要的一步）：

```bash
B=Data/put_cup_in_bowl_outline
V=Data/put_cup_in_bowl_ab01_det25
for st in s3_ground s5_scene s6_upsample s8_pose s11_sim; do
  echo "=== $st ==="
  diff <(python3 -m json.tool $B/$st/stage_info.json) \
       <(python3 -m json.tool $V/$st/stage_info.json)
done
```

**预期**：`s5_scene` 的 `detection_model`/`removal_model`、`s3_ground` 的 `detection_model`、
`s8_pose` 的 `front_pick_model`、`s11_sim` 的 `vlm_model` 出现差异；
**其余键必须完全一致**。若出现预期外的差异，该次对比不成立。

**再按 §4.4 的七个维度逐个比**：

```bash
S=put_cup_in_bowl_ab01_det25
# ② 检测质量
for f in Data/$S/s5_scene/obj_cat_list/iter_*.json; do python3 -m json.tool "$f"; done
# ③ 稳定性
ls Data/$S/s5_scene/skipped_iterations/
# ⑥ 端到端
ls Data/$S/s11_sim/objects/
# ⑤ 回填质量：打开对照图看
#   Data/$S/s5_scene/post_object_removal/iter_1.png
#   Data/put_cup_in_bowl_outline/s5_scene/post_object_removal/iter_1.png
```

**关键判据回顾**：基线（Pro + `outline`）的 `s11_sim/objects/` 是 **3 个物体**、
stage5 **3 轮**且全部 `is_valid_removed_obj: true`；回填背景环亮度误差 **0.9**
（而旧 `mask` 提示词是 **105.1**）。变体不应显著偏离。

### 9.5 ⚠️ 本次踩到的坑：密钥来源失效

**症状**：A/B 启动前 `GEMINI_API_KEY` 无法从 `.bash_history` 取到（此前一直可以）。

**根因**：`.bash_history` 触到了 bash 的 **2000 行上限**，历史轮转把最初那行
`export GEMINI_API_KEY=...` **挤出去了**。文件里只剩下引用它的那些命令
（`KEY=$(grep ...)`、`re.findall(...)` 等），所以匹配到的都是"看起来像 key 的超长串"
而非真值。

**教训**：**"运行时从 shell history 取密钥"是脆弱设计** —— 它只在那一行仍处于
2000 行窗口内时有效。任何依赖它的脚本都会在某天静默失效。

**本次解法**：从原始 VS Code transcript 里**按指纹反查**取回
（`sha256[:8] == e4d7cfdc` 且长度 70），避免误取无关长串。

**推荐的长久修法**（未执行）：把 key 写进 `scripts/installation/api_keys.txt` ——
`simfoundry/models/vlm.py:106` 的 `load_api_keys()` 会**原生读取**该文件并注入
`os.environ`，于是不需要任何 env var、也不依赖 history。
⚠️ 代价是**密钥落盘**；该实例共享/可克隆，`.gitignore` 已覆盖此文件，但落盘本身
仍是权衡（见 `RESUME_INSTALL.md` §12.2 对 token 的同类告诫）。

**当前**：`run_ab_flash.sh` 的 `resolve_key()` 按
**env → `api_keys.txt` → `.bash_history` → transcript（指纹校验）** 四级回退，
所以现在能跑；但若你要长期复用，建议采用上面的长久修法。

### 9.6 结果（2026-09-15，全部 4 个变体 exit=0）

**运行窗口**：11:03:01 → 13:14:21（约 2h11m）

#### ① 变量唯一性 —— 通过 ✅

每个变体的 `stage_info.json` 都显示**只有预期的模型键被改动**，其余保持 Pro 默认：

| 场景 | s3.det | s5.det | s5.removal | s6.model | s8.front | s11.vlm |
|---|---|---|---|---|---|---|
| 基线 | pro-3.1 | pro-3.1 | pro-image | pro-image | pro-3.1 | pro-3.1 |
| `ab01_det25` | **2.5-flash** | **2.5-flash** | pro-image | pro-image | **2.5-flash** | **2.5-flash** |
| `ab03_img25` | pro-3.1 | pro-3.1 | **2.5-flash-image** | **2.5-flash-image** | pro-3.1 | pro-3.1 |
| `ab05_all25` | **2.5-flash** | **2.5-flash** | **2.5-flash-image** | **2.5-flash-image** | **2.5-flash** | **2.5-flash** |
| `ab02_det3` | **3-flash-pv** | **3-flash-pv** | pro-image | pro-image | **3-flash-pv** | **3-flash-pv** |

#### ② stage 5：**4 个变体全部无幻影** ✅

| 变体 | 轮数 | `valid_pixel_proportion` | 幻影 |
|---|---|---|---|
| 基线 | 3 | 1.0000 / 1.0000 / 0.9977 | 无 |
| `ab01_det25` | 3 | 1.0000 / 1.0000 / 0.9989 | 无 |
| `ab03_img25` | 3 | 1.0000 / 1.0000 / 0.9996 | 无 |
| `ab05_all25` | 3 | 1.0000 / 1.0000 / 0.9997 | 无 |
| `ab02_det3` | 3 | 1.0000 / 1.0000 / 0.9983 | 无 |

**这是本次最重要的结果**：即使把图像类换成 `gemini-2.5-flash-image`（我们唯一验证过会出幻影的那一环），
**也没有重新出现幻影**。

#### ③ 端到端 `s11_sim/objects/`

| 变体 | 物体 | 与基线 |
|---|---|---|
| 基线 | `black_pen`, `orange_bowl`, `white_paper_cup` | — |
| `ab01_det25` | 同上 | ✅ 一致 |
| `ab03_img25` | 同上 | ✅ 一致 |
| `ab05_all25` | 同上 | ✅ 一致 |
| `ab02_det3` | `black_and_red_pen`, `orange_bowl`, `white_paper_cup` | ⚠️ 仅**笔的命名**不同 |

> `ab02` 的差异**只是物体名**（同一个笔，`black_pen` vs `black_and_red_pen`）。
> 这个命名抖动与模型无关 —— 两个 Pro 基线自己就不同：
> `put_cup_in_bowl`(Pro) → `black_and_red_pen`，`put_cup_in_bowl_outline`(Pro) → `black_pen`。
> **物体个数与集合完全一致（3 个）**。

#### ④ 掩码一致性与回填质量

| 变体 | iter_0 / iter_1 / iter_2 的 IoU |
|---|---|
| `ab01_det25` | 0.9982 / 0.9545 / 0.9946 |
| `ab03_img25` | 0.9154 / **0.1488** / 0.8039 |
| `ab05_all25` | 0.9858 / **0.7211** / 0.9716 |
| `ab02_det3` | 0.9984 / 0.9526 / 0.9943 |

**文本类几乎无影响（IoU ≥ 0.95）；图像类会让掩码明显改变。**

⚠️ **但 iter_1 的低 IoU 具有误导性** —— 它对应的是那支**细长斜放的笔**：

```
基线 iter_1 mask: y[382,557] x[1044,1168]  面积 3260
ab03 iter_1 mask: y[381,553] x[1055,1178]  面积 3003
```

bbox 几乎重合、面积相近，但 IoU 只有 0.15 —— 因为**细长物体整体平移几个像素就会让 IoU 崩塌**。
所以 **IoU 不适用于细长物体**，应改看 bbox/面积。

用**基线的笔掩码**在两种结果上采样（周围背景环均值 **119.6**）：

| | 原始帧 | iter_0 后 | iter_1 后 |
|---|---|---|---|
| 基线 | 76.0（笔） | 73.6（笔未动） | **118.7**（已填成背景）|
| `ab03` | 76.0（笔） | **95.3**（被扰动） | **122.7**（已填成背景）|

**两个发现**：

1. **iter_1 两者都成功把笔填成了背景**（118.7 / 122.7 vs 背景 119.6）✅
2. ⚠️ **`ab03` 的 iter_0（移除橙碗）连笔的区域也被轻微改动**（73.6 → 95.3）。
   `pro-image` 则几乎不动（73.6）。原因是 `crop_removal_images: true` 会把**带大边距的裁剪块**
   整块贴回，图像模型重新生成该块时会牵连到附近像素。**flash 图像模型对更宽区域的扰动大于 pro。**

#### ⑤ 墙钟时间（噪声较大）

| 变体 | 墙钟 |
|---|---|
| 基线 | 41m 13.7s（该次 stage 8 有 22m59s 异常值）|
| `ab01_det25` | 27m 50.0s |
| `ab03_img25` | **49m 54.7s** |
| `ab05_all25` | **26m 20.4s** |
| `ab02_det3` | 27m 14.2s |

**Flash 并未系统性地更慢** —— `ab05`（两个类都换 flash）反而是最快的，
`ab03`（只换图像类）最慢。差异主要来自 stage 7/8（网格生成与位姿匹配）的运行时长波动。

#### ⑥ 结论

| 结论 | 证据 |
|---|---|
| **四类变体全部可用** | 4/4 无幻影、3 个物体、exit=0 |
| **文本类换 flash 基本无代价** | IoU ≥ 0.95，prop 一致，端到端物体完全一致 |
| **图像类换 flash 可用但可测地改变行为** | 掩码变化更大、对附近区域扰动更宽；但结果仍正确 |

**建议**：文本类（4 个旋钮）可放心采用 `gemini-2.5-flash`；
图像类（2 个旋钮）**需要你在更多场景上人工确认**（现只有 1 个场景、4 次运行）。

### 9.7 第二轮：3.1 代 Flash 图像模型（2026-09-15）

**目标**：把**图像类**两个旋钮（`s5_scene.removal_model` / `s6_upsample.model`）从
`gemini-3-pro-image` 换成 **3.1 代 Flash 图像模型**。**文本类 4 个旋钮保持 Pro 默认** ——
这一轮只隔离图像类，与 §9.6 的 `ab03`/`ab05` 构成同一根轴上的 **2.5 代 vs 3.1 代**对照。

#### ⑧ 前置：3 处代码改动

两个 3.1 代 id 原本既不在 `Gemini.VERSIONS` 里、也不在 stage 级注册表里，
**必须同时补两道闸门**（§1.3），否则分别报 `KeyError` / `AssertionError`：

| # | 文件:行 | 改动 |
|---|---|---|
| 1 | `simfoundry/models/vlm.py:609` / `:613`（`Gemini.VERSIONS`，块起始 `:565`）| `+gemini-3.1-flash-image`、`+gemini-3.1-flash-lite-image`，均 `modalities=["TEXT","IMAGE"]`、`max_tokens=32768` |
| 2 | `scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py:509`（`REMOVAL_MODELS`）| + 上述两个 id |
| 3 | `scripts/pipeline/A_reconstruction/stages/6_upsample_object_images.py:40`（`UPSAMPLE_MODELS`）| + 上述两个 id |

**调用点零改动**：`5_decompose_scene.py:883` 的 `elif "gemini" in removal_model_name:`
是**子串匹配**，会接住任意 `gemini-*` id；stage 6 同理。响应侧 `Gemini.get_result_images`
（`vlm.py:809`）走 `part.inline_data`，与 `gemini-3-pro-image` 完全同形，
所以请求与响应两侧都不需要适配。

**闸门实测**：

```bash
mamba run -n simfoundry python -c "
from simfoundry.models.vlm import Gemini
for m in ['gemini-3.1-flash-image','gemini-3.1-flash-lite-image']:
    print(m, Gemini.VERSIONS[m])"
```

→ 两个 id 均为 `modalities=['TEXT','IMAGE'] max_tokens=32768`，**闸门1 / 闸门2 全 ✅**。

**中转实测**：两个 id 对 `TEXT+IMAGE` 请求都返回 `parts=['IMAGE']`（`out=1120`），
**裸名与 `google/` 前缀写法都可用**。

#### ⑨ 变体矩阵与运行方式

| 顺序 | 场景名 | 文本类 | 图像类 | 隔离什么 |
|---|---|---|---|---|
| — | `put_cup_in_bowl_outline` | pro-3.1 | pro-image | 基线（复用 §9.6）|
| — | `put_cup_in_bowl_ab03_img25` | pro-3.1 | 2.5-flash-image | 2.5 代对照（复用 §9.6）|
| 5 | `put_cup_in_bowl_ab06_img31` | pro-3.1 | **`gemini-3.1-flash-image`** | 3.1 flash |
| 6 | `put_cup_in_bowl_ab07_img31lite` | pro-3.1 | **`gemini-3.1-flash-lite-image`** | 3.1 flash **lite** |

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name put_cup_in_bowl_ab06_img31 \
  --video-fpath /root/workspace/SimFoundry/docs/assets/example_videos/PutCupInBowl.mp4 \
  -- 's5_scene.obj_removal_prompt_order=["outline","bbox","mask"]' \
     s5_scene.removal_model=gemini-3.1-flash-image \
     s6_upsample.model=gemini-3.1-flash-image
```

运行方式同 §9.3：脚本 `/root/simfoundry_logs/run_ab_img31.sh`（**仓库外，不含密钥**）、
日志 `run_ab_img31.log`、监控 `watch_ab_img31.py` → `ab_img31_status.txt`、
`nohup setsid`（**PPID=1**，SSH 断开不影响）。

**运行窗口**：14:57:39 → 15:56:18（约 59 分钟），两个变体 **exit=0**。

#### ⑩ stage 5 与端到端

| 场景 | 图像模型 | stage5 轮数 | `s11_sim/objects/` | 幻影 | 墙钟 |
|---|---|---|---|---|---|
| `put_cup_in_bowl`（**旧 mask 提示词**）| pro-image | **4** | `black_and_red_pen`, **`knife`**, `orange_bowl`, `white_paper_cup` | **`knife`** | 33m57s |
| `put_cup_in_bowl_outline`（基线）| pro-image | 3 | `black_pen`, `orange_bowl`, `white_paper_cup` | 无 | 41m14s |
| `..._ab01_det25` | pro-image | 3 | 同上 | 无 | 27m50s |
| `..._ab02_det3` | pro-image | 3 | `black_and_red_pen`, … | 无 | 27m14s |
| `..._ab03_img25` | 2.5-flash-image | 3 | `black_pen`, `orange_bowl`, `white_paper_cup` | 无 | 49m55s |
| `..._ab05_all25` | 2.5-flash-image | 3 | 同上 | 无 | 26m20s |
| **`..._ab06_img31`** | **3.1-flash-image** | **3** | **`black_pen`, `orange_bowl`, `white_paper_cup`** | **无** | **26m41s** |
| **`..._ab07_img31lite`** | **3.1-flash-lite-image** | **3** | **同上** | **无** | **31m57s** |

两个 3.1 变体的**笔名与基线完全一致**（`black_pen`）—— 比 §9.6 里 `ab02` 的
`black_and_red_pen` 抖动更稳。

#### ⑪ 回填质量：统一探针（本轮核心证据）

沿用 §9.6 的方法：取**基线的笔掩码**
（`put_cup_in_bowl_outline/s5_scene/removal_mask/iter_1.png`，3260 px）作为
**所有变体共用的探针区域**，读其灰度均值。原始帧该区域 **76.0**（笔），
周围背景环 **119.6**。

| 变体 | 图像模型 | iter_0 后<br>（抹碗，笔**应不动**）| iter_1 后<br>（抹笔）| 偏离背景 |
|---|---|---|---|---|
| `put_cup_in_bowl`（旧 mask 提示词）| pro-image | 73.6 | **224.7** | **+105.1** ⚠️ |
| `put_cup_in_bowl_outline`（基线）| pro-image | 73.6 | **118.7** | −0.9 |
| `ab01_det25` | pro-image | 73.8 | **118.9** | −0.7 |
| `ab02_det3` | pro-image | 75.8 | **119.6** | **0.0** |
| `ab03_img25` | 2.5-flash-image | 95.3 | 122.7 | +3.1 |
| `ab05_all25` | 2.5-flash-image | 77.0 | 124.6 | +5.0 |
| **`ab06_img31`** | **3.1-flash-image** | 94.8 | **111.6** | **−8.0** |
| **`ab07_img31lite`** | **3.1-flash-lite-image** | **113.1** | **112.0** | **−7.6** |

**三点读数：**

1. ✅ **`224.7` 是幻影 bug 的数值铁证。** 旧 `mask` 提示词下，笔的位置被回填成一块
   **比周围背景亮 105 级**的白斑 —— 这正是 §3.4 里那句"回填产出透明棋盘格/色块"
   在像素上的样子，也就是 stage 5 在 iter_3"检出" `knife` 的来源。
   它把 §3.4 对"回填差 → 幻影"的**定性**描述变成了**可测量的数字**
   （§9.4 已记下 $105.1$，本轮补上了完整的 8 变体横表）。
2. ✅ **全部 `outline` 优先的变体都落在 111–125 区间**，没有高对比白斑，也就没有幻影（**7/7**）。
3. ⚠️ **两个 3.1 模型在回填忠实度上是这套里最弱的一档**：抹掉笔后比背景**暗 ~8 级（约 6%）**。
   这是渐变式偏暗、不是高对比伪影 —— 所以没触发 VLM 误检，但它**是可见的**。
   另外 `ab07`(lite) 最不守掩码：**抹碗时把不该动的笔区域从 76.0 改到了 113.1**，
   而 `pro-image` 只动到 73.6（§9.6 已指出这是 `crop_removal_images: true` 整块贴回所致）。

**回填忠实度排序**：`ab02` ≈ `ab01` ≈ 基线 > `ab03` > `ab05` > `ab07` ≈ `ab06`

#### ⑫ 结论

| 结论 | 证据 |
|---|---|
| **两个 3.1 代模型都可用** | 2/2 无幻影、3 个物体、笔名与基线一致、exit=0 |
| **`gemini-3.1-flash-image` 是当前最优的省成本选项** | 26m41s vs 基线 41m14s，端到端物体集合与基线一致 |
| **`gemini-3.1-flash-lite-image` 不建议采用** | 比同代 flash **更慢**（31m57s > 26m41s）、回填最不守掩码、无任何收益 |
| **3.1 代在回填忠实度上不优于 2.5 代** | 偏离背景 8.0 / 7.6，vs 2.5-flash-image 的 3.1 / 5.0 |

**建议**：图像类若以**成本**为目标 → `gemini-3.1-flash-image`；
若以**最高保真**为目标 → 仍是 `gemini-3-pro-image`（偏离 0.9）。

#### ⑬ 局限（引用这些数字时必须一并声明）

1. **每个 3.1 变体只跑了 1 次**，单样本。而原始基线的幻影是 **4/4 都出现** ——
   样本量不对等，所以"3.1 无幻影"这个结论**统计上不强**。
2. **`ab03` 的 49m55s 是异常值**，墙钟不要用于精确排名；
   `ab01/ab02/ab05/ab06` 聚集在 **26–28 分钟**才是可信区间。
3. **回填数字全部来自本地自跑变体**，与 `Data/` 里的既有数据**分开记录**，
   未混合成结论。
4. 3.1 flash 那 **~8 级暗斑**是否会在下游（网格贴图 / 渲染）被放大，**未验证**。
