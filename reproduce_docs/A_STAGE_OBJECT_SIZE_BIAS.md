<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A 阶段物体尺寸偏差：深度分辨率导致球状物体被系统性放大

> 用途：**交接文档**。记录「苹果/橙子等球状物体在 A 阶段重建后被放大了 1.5–1.7 倍、
> 而香蕉（长条）基本正确」这一现象的完整排查过程、量化证据、根因定位，以及试过但**无效**的方案。
> 撰写日期：**2026-09-17**。场景：`fruits`（15 帧视频，6 个物体）。
> 证据分级：**[实测]** 表示本次跑出的数字，**[估计]** 表示来自常识别/视频目测、只用于给量级。
> 相关代码：[`8_match_object_poses.py`](../../scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py)、
> [`11_make_objects_sim_ready.py`](../../scripts/pipeline/A_reconstruction/stages/11_make_objects_sim_ready.py)、
> [`scripts/cfg/real2sim_cfg.yaml`](../../scripts/cfg/real2sim_cfg.yaml)。
> 相关诊断脚本：`/root/simfoundry_logs/diag{2,3,4,5,6}.py`、`check_reach.py`、`diff_scenes.py`。

---

## 0. 一句话结论

物体的最终尺寸**完全由 `8_match_object_poses.py:467` 的一个 OBB 对角线之比决定**。
这个对角线来自「掩码区域 + 单目深度」反投影出的点云，而 **DA3 的深度跑在 448×252
（ViT patch 14 → 只有 32×18 个 patch），小物体（苹果掩码仅 36×39 px ≈ 2.8×2.8 patch）
低于模型的分辨率极限**，掩码内深度出现约等于物体尺寸本身的跨度（苹果 146 mm / 物体 75 mm）。
深度噪声把点云撑成一块**斜板**，对角线随之膨胀 1.5–1.8 倍。因为随后的缩放是**各向同性**的，
而球体对角线与三个轴强耦合（$d\sqrt3$），所以**球状物体被整体放大**，长条物体的长轴却基本不受影响。

**掩码本身是准的（误差 3–10%），不是掩码的问题。**

---

## 1. 现象

C 阶段策略评测反复出现同一个模式：**香蕉能抓，苹果和橙子抓不起来**。
用 `misc/metadata.json` 里的 `bbox_size`（这是**测量值**，不是配置值，见 §2）对照真实物体尺寸：

| 物体 | 真实尺寸 | 重建后 `bbox_size` | 倍率 |
|---|---|---|---|
| `red_apple` | ~75 mm **[估计]** | **112.5 × 110.2 × 113.9 mm** | **×1.50** |
| `orange` | ~80 mm **[估计]** | **133.9 × 133.8 × 121.1 mm** | **×1.67** |
| `yellow_banana` | ~200 × 35 × 35 mm **[估计]** | 128.3 × **209.0** × **39.0** mm | 长轴 ×1.05、厚度 ×1.11 |
| `yellow_pear` | ~90 × 75 × 70 mm **[估计]** | 92.9 × 86.3 × 114.8 mm | ≈ ×1.2 |
| `teal_plate` | — | 307 × 227 × 39 mm | — |
| `orange_plate` | — | 255.5 × 209.3 × 47.9 mm | — |

**关键观察：出问题的是「球状」（苹果、橙子），长条的香蕉几乎准确。** 这个不对称是最重要的线索。

> 复现读取：
> ```bash
> for o in red_apple orange yellow_banana yellow_pear teal_plate orange_plate; do
>   find Data/fruits/s13_usd/objects/$o -name metadata.json | head -1 | \
>     xargs -I{} python -c "import json,numpy as np;print('$o',np.round(np.asarray(json.load(open('{}'))['bbox_size'])*1000,1))"
> done
> ```

---

## 2. 尺寸是谁决定的（链路）

```
8_match_object_poses.py:466-467
    source_obb_diag = ‖观测点云.get_oriented_bounding_box().extent‖     ← 掩码+深度反投影
    target_obb_diag = ‖mesh.sample_points().get_oriented_bounding_box().extent‖
    pre_scale_factor = source_obb_diag / target_obb_diag
        ↓  写入 Data/<scene>/s8_pose/info/iter_N.json["pre_scale_factor"]
        ↓  （同文件的 "z_up"["scale"] 恒为 [1,1,1]，不参与尺寸）
        ↓  stage 8 把 pre_scale_factor 直接烘进 canonical mesh
11_make_objects_sim_ready.py:353-376
    pre_scale_factor = obj_info.get("pre_scale_factor", 1.0)
    tf_scale         = obj_info["z_up"]["scale"]          # = 1.0
    rigid_scale      = tf_scale                           # 刚性物体：不再缩放
    tm.apply_scale(rigid_scale)
```

**[实测] 验证这条链**：重投影出的点云 OBB 对角线 ≈ 最终网格对角线，六个物体差值均 ≤1.4%。
所以「物体最终多大」这个问题，**等价于「那个点云的对角线多长」**。

**[实测] 同时确认：`scripts/cfg/real2sim_cfg.yaml` 里没有任何尺度配置键。**
（`s8_pose` / `s11_sim` 段都没有；`s2_fs.scale: 0.5` 与 `s4_frame` 的 `source_voxel_scale_factor`
是别的用途。）**要改尺寸，目前只能改 `pre_scale_factor` 或另开一个键。**

---

## 3. 误差定位：三步分解

点云的尺寸由两件事决定，分开量就能定位：

1. **掩码的角尺寸**（把掩码区域内所有点的深度强制为区域中位深度）：只反映「掩码在图像里占多大」
2. **掩码 + 真实深度**：把深度的起伏也放进来

**[实测]**（PCA 主轴 extent，单位 mm，`/root/simfoundry_logs/diag4.py`）：

| 物体 | 真实对角线 **[估计]** | ① 掩码角尺寸 | ② 掩码+深度 | ②/① | ②/真实 |
|---|---|---|---|---|---|
| `red_apple` | 130 | **126** | **198** | 1.57× | **1.53×** |
| `orange` | 139 | **125** | **225** | 1.80× | **1.62×** |
| `yellow_banana` | 206 | 212 | 256 | 1.21× | 1.24× |
| `yellow_pear` | 136 | 137 | 180 | 1.31× | 1.32× |
| `orange_plate` | — | 292 | 355 | 1.22× | — |
| `teal_plate` | — | 306 | 374 | 1.22× | — |
| **单位** | mm | mm | mm | | |

**两点结论：**

1. **掩码是准的。** 苹果 126 vs 真值 130（0.97×）、橙子 125 vs 139（0.90×）。
   掩码不是问题，**不要往掩码方向查**。
2. **深度是罪魁。** 只看掩码角尺寸时对角线已经很接近真值，一旦把真实深度放进来就膨胀到 1.5–1.8 倍。

**[实测] 深度的离谱程度**：掩码区域内的深度跨度（max − min）

| 物体 | 物体自身尺寸 | **掩码内深度跨度** |
|---|---|---|
| `red_apple` | 75 mm | **146 mm** |
| `orange` | 80 mm | **172 mm** |
| `yellow_banana` | 39 mm 厚 | 166 mm |
| `orange_plate` | — | 207 mm |
| `teal_plate` | — | 220 mm |

**一个 75 mm 的苹果上，深度读数跨了 146 mm —— 噪声幅度约等于物体本身。**

**[实测] 目视证据**：把掩码区域内的深度图放大看（`/root/simfoundry_logs/crop_red_apple_depth.png`），
苹果掩码内的深度是**两段式**的——一半近、一半远，中间一条**硬边**。这不是平滑噪声，
而是掩码**横跨了一条深度断层**。

> 复现：
> ```bash
> cd /root/workspace/SimFoundry
> export OMP_NUM_THREADS=8
> /root/autodl-tmp/simfoundry/conda_envs/simfoundry/bin/python /root/simfoundry_logs/diag4.py   # 三步分解
> /root/autodl-tmp/simfoundry/conda_envs/simfoundry/bin/python /root/simfoundry_logs/diag6.py   # 深度置信度
> ```

---

## 4. 根因：深度分辨率低于物体尺寸

`scripts/cfg/real2sim_cfg.yaml:415`（`s2_da` 段）：

```yaml
s2_da:
  resolution: 448    # Must be multiple of 14 (chunked-VRAM-safe @ 220-frame chunks on 24 GB)
```

**[实测]** `Data/fruits/s2_da/da/exports/npz/results.npz` 里：

```
depth / conf / intrinsics 形状均为 (15, 252, 448)      ← 15 帧、252 高、448 宽
intrinsics[11]: fx=382.73 fy=372.45 cx=224.00 cy=126.00
```

`Data/fruits/s5_scene/original_depth.npy` 与 `results["depth"][11]` **逐元素完全相同**
（最大差 0.00e+00）—— 说明深度**没有被任何后续处理修正过**。

**DA3 是一个 ViT，patch = 14 px。448×252 只剩 `32 × 18 = 576` 个 patch。
苹果的掩码在 448×252 下只有 36×39 px ≈ `2.8 × 2.8` 个 patch。**

也就是说**苹果低于模型的分辨率极限**。模型无法在 3×3 个 patch 内表达一个球面的深度起伏，
预测出的是一块被平滑过、且与掩码**对齐关系任意**的低频面 —— 掩码边界恰好落在
「苹果 patch」与「桌面 patch」之间，于是区域内的深度被劈成两半。

**[实测] 这也解释了为什么香蕉没坏**：香蕉长 200 mm，沿长轴跨约 8 个 patch，
其长度方向的分辨率是够的。

**[实测] 内参无关**：`Data/fruits/s2_fs/` 目录不存在，本场景走的是 DA3 后端
（`s2_da/stage_info.json`: `"backend": "da3"`, `"resolution": 448`）。
掩码是 1408×736，深度是 252×448，stage 8 用
`cv2.resize(mask, (W,H))` + `unpad_image` 在两者之间换算 —— **这不是对齐错误**，
换算链是干净的（1280×720 原始帧，pad 到 1408×736，再缩放）：

```python
# 8_match_object_poses.py:398-406
rgb_resized  = cv2.resize(rgb, (W, H))          # W,H = padded 尺寸 1408x736
mask_resized = cv2.resize(mask, (W, H))
mask_resized_unpadded = unpad_image(mask_resized, delta_w, delta_h)   # -> 252x448
mask_resized_unpadded_eroded = erode_mask(mask_resized_unpadded, kernel_size=3)
pc = compute_point_cloud_from_depth(depth=depth, K=K).reshape(-1, 3)
obj_pc = pc[mask_resized_unpadded_eroded.flatten()]
```

---

## 5. 为什么球状物体受害最重

`pre_scale_factor` 施加的是**各向同性**缩放（`target.scale(pre_scale_factor, center=zeros)`），
而它匹配的量是**对角线**，对角线把三个轴耦合在一起：

- **球状物体**：真值对角线 $= d\sqrt3$，三个轴贡献相等。
  深度噪声注入的第三个轴（实测苹果 51.6 mm、橙子 54.7 mm）让对角线暴涨，
  而这个倍数被**均匀摊到三个轴**上 → 苹果 ×1.53、橙子 ×1.62。
- **长条物体**：对角线已由**长轴**主导（香蕉 ~206 mm），长轴是横向尺寸、测得准；
  噪声主要沿视线方向叠加，对对角线的贡献只是部分叠加 → 放大倍数被网格自身的长宽比吸收。
  缩放后长轴基本不变，只有横截面按网格自身比例放大（绝对量很小）。

**这就是「苹果 ×1.5、橙子 ×1.7、香蕉准」的完整机制。**

---

## 6. 试过但**无效**的方案（负结果，别再走一遍）

| 方案 | 实测结果 | 结论 |
|---|---|---|
| 按中位深度做固定容差滤波（±40 mm） | 苹果 198→**148**、橙子 225→**126**、梨 180→**139**（变好）；但香蕉 256→**168**（反而短 18%）、盘子 355→315 / 374→266（削小） | ✗ **不是通解**。绝对容差对 75 mm 苹果是 53%、对 200 mm 香蕉只有 20%，量纲上就错了 |
| ±25 mm / ±15 mm 收紧容差 | 苹果 →78 / 52，香蕉 →125 / 78 | ✗ 过切 |
| 剔除外层 20% 离群点 | 苹果对角线只从 197 → **174** | ✗ **是系统性膨胀，不是离群点**，剔除无效 |
| 用 DA3 的 `conf`（置信度）过滤 | 物体区域/周围桌面环的置信度之比 = **1.01**（苹果）、0.92（橙子）、0.99（香蕉） | ✗ **区分不开**。虽然苹果区域内「近半 18.26 vs 远半 11.50」有信号，但太弱、需要调参，不适合做通用修法 |
| 去改 `pre_scale_factor` 的定义（换成平面/掩码对角线） | 未执行 | ⚠️ **有副作用**：`pre_scale_factor` 在 `sample_fits` 里同时是**位姿拟合的尺度先验**（`target = target.scale(pre_scale_factor)`），改它会连带改变物体位姿 |

**唯一站得住的方向是提高深度分辨率**（见 §7）。**没有任何尺度配置键**（见 §2）。

---

## 7. 建议的修法（按代价排序）

### 方案 A：提高深度分辨率 —— 纯配置

```yaml
s2_da:
  resolution: 1008    # 或 952；必须是 14 的倍数
```

- **改动**：配置里一个数字，零代码。
- **理由**：直接抬高 ViT 的 patch 网格，让 75 mm 的物体跨足够多的 patch。
  1008 宽时 patch 数从 32 升到 72，苹果从 2.8 个 patch 升到约 6.3 个。
- **代价 [实测]**：`pipeline_timing.log` 里 stage 2（深度）在 448 下**只花 41.5 s**
  （15 帧），提到 1008 预计约 3 min；但 **stage 2 的输出是所有下游的输入**，
  所以 stage 2→14 必须整体重跑：stage 5→8 约 30 min、10→14 约 19 min，
  **A 链路合计约 52 min**（当前全链路 51 min）。
- **风险**：会改变点云 → 改变位姿拟合 → 改变所有物体的位姿与网格。**这是一个「重做 A」的动作，
  不是一次微调。建议先在临时目录只跑 stage 2 验证深度跨度是否回落，再决定是否全跑。**

### 方案 B：后置尺度修正 —— 小改动，不动位姿拟合

新增一个可选键（默认 `null` = 保持现状），在 stage 11 载入 mesh 之后乘一个系数：

```yaml
s11_sim:
  scale_overrides: null        # 例如 {iter_3: 0.657, iter_4: 0.618}
```

- **改动**：新增一个可选配置键 + stage 11 里约 2 行代码。默认行为完全不变。
- **优点**：**只须重跑 stage 11→14（约 20 min）**，不触碰 stage 8 的位姿拟合。
- **系数**（由 §1 的测量直接给出）：`red_apple` **0.657**、`orange` **0.618**、`pear` 0.756、`banana` 0.805。

### 方案 C：不动 pipeline，在场景层改 —— 已在使用

`objects_info.init_info[<obj>].args.scale` 是一个真实可编辑字段，OmniGibson 在载入时施加。
本次已用它把苹果/橙子改到真实尺寸（见 §8），做法是**在 `Data/` 之外**做场景变体，
不修改 pipeline 产物。适合「先让评测跑通」，但不解决根因。

---

## 8. 当前的临时处置（2026-09-17）

为了在不重跑 pipeline 的前提下让评测继续，做了两件事：

1. **`Data/fruits/s14_og_table_robotiq/`** —— 用 `end_effector=robotiq` 重建的桌子场景
   （原因见 `OPENPI05_ADAPTATION.md`：换末端会让夹爪信号反向）。
2. **`/root/autodl-tmp/simfoundry/scene_variants/fruits_table_robotiq_smallfruit/`** ——
   场景变体，只改两个数：

   ```
   iter_3 (red_apple) scale 1.0 -> 0.667   =>  112.5 x 110.2 x 113.9 mm  ->  75.0 x 73.5 x 76.0 mm
   iter_4 (orange)    scale 1.0 -> 0.600   =>  133.9 x 133.8 x 121.1 mm  ->  80.3 x 80.3 x 72.7 mm
   ```

   生成方式（`/root/simfoundry_logs/make_scene_variant.py`，下午 `fruits_apple070` 用的同一个脚本）：
   ```bash
   PY=/root/autodl-tmp/simfoundry/conda_envs/simfoundry/bin/python
   V=/root/autodl-tmp/simfoundry/scene_variants/fruits_table_robotiq_smallfruit/reconstructed_og_scene.json
   $PY /root/simfoundry_logs/make_scene_variant.py \
       --src Data/fruits/s14_og_table_robotiq/reconstructed_og_scene.json \
       --out "$V" --target iter_3 --scale 0.667
   $PY /root/simfoundry_logs/make_scene_variant.py \
       --src "$V" --out "$V" --target iter_4 --scale 0.600
   ```

   > **注意**：`args.scale` 是绕物体原点缩放，所以物体底面会略微抬高，
   > 重置时会有一次很小的下落。下午的 `apple070`（同样做法）连续 3/3 成功，说明这个副作用可接受。

---

## 9. 复现这次排查所需的东西

| 项 | 路径 / 命令 |
|---|---|
| 观测点云（stage 8 产物） | `Data/fruits/s8_pose/pc/iter_N.ply` |
| 尺度元数据 | `Data/fruits/s8_pose/info/iter_N.json` → `pre_scale_factor`、`z_up` |
| 深度 / 内参 / 置信度 | `Data/fruits/s2_da/da/exports/npz/results.npz`（key: `image`/`depth`/`conf`/`extrinsics`/`intrinsics`）|
| 掩码（1408×736） | `Data/fruits/s5_scene/removal_mask/iter_N.png` |
| 深度实际用量（252×448） | `Data/fruits/s5_scene/original_depth.npy` |
| 最终尺寸（测量值） | `Data/fruits/s13_usd/objects/<cat>/<model>/misc/metadata.json` → `bbox_size` |
| 三步分解脚本 | `/root/simfoundry_logs/diag4.py` |
| 深度置信度脚本 | `/root/simfoundry_logs/diag6.py` |
| 容差滤波实验脚本 | `/root/simfoundry_logs/diag5.py` |
| 掩码/深度可视化 | `/root/simfoundry_logs/diag2.py` → `crop_<name>.png`、`crop_<name>_depth.png` |

**环境**：
```bash
cd /root/workspace/SimFoundry
export PATH=/root/miniforge3/bin:$PATH MAMBA_ROOT_PREFIX=/root/miniforge3
export OMP_NUM_THREADS=8          # 否则 open3d 会报 "nthreads must be a positive integer"
unset http_proxy https_proxy
/root/autodl-tmp/simfoundry/conda_envs/simfoundry/bin/python <脚本>
```

---

## 10. 还没弄清楚的 / 建议下一步

1. **确认「提高分辨率真的能修好」**：只跑 stage 2 到 `resolution: 1008`，写进临时目录，
   再量苹果掩码内的深度跨度是否从 146 mm 回落到物体量级（~30 mm）。
   **这是 3 分钟的实验，应该先做，再决定要不要花 52 分钟全跑 A。**
2. **是否存在更好的尺度参考量**：实测中「掩码角尺寸」对四个水果的准确度是
   126/130、125/139、212/206、137/136（误差 −3% ~ −10%），远好于三维对角线。
   但改它会影响 `sample_fits` 的位姿拟合，需要单独评估。
3. **盘子的行为不同**：`orange_plate` / `teal_plate` 的掩码角尺寸只有「真实」的 0.6–0.7 倍
   （若是大而薄的物体，单视角掩码本就会低估对角线）。本节所有「真实值」里盘子的数字是**[估计]**，
   偏低的可能性大，**结论只对水果成立**。
4. **`z_up.scale` 恒为 [1,1,1] 是否是设计如此**：stage 11 的注释说
   「canonical mesh from step 8 already has pre_scale_factor applied」，
   但 `sample_fits(update_scale=True)` 那条分支算出的 `tf_z_up.scale` 最终没被用上 ——
   这条路径值得确认是不是死代码。

---

## 附录：本次踩到的环境坑

- **`pxr` / `open3d` / `matplotlib` 分布在不同的 conda env**：`from pxr import Usd` 只在
  `simfoundry-editor` 里可用；`simfoundry` 里有 `open3d` 和 `numpy` 但**没有可用的 pxr**。
  本次所有点云/深度计算都在 `simfoundry` 里做。
- **`open3d` 的 `get_oriented_bounding_box()` 对平面点云会抛 Qhull QH6154**
  （"Initial simplex is flat"）。量「掩码角尺寸」时必须改用 PCA，或给点云加抖动。
- **`AxisAlignedBoundingBox` 没有 `.extent` 属性**，要用 `get_extent()`。
- **`OMP_NUM_THREADS` 未设或为 0 时 open3d 报错** `nthreads must be a positive integer`。
- **`Data/fruits/s2_fs/` 不存在**：本场景用 DA3，内参要去
  `Data/fruits/s2_da/da/exports/npz/results.npz["intrinsics"]` 拿，
  而不是 `s2_fs/image_<idx>_K.npy`（那是 `use_fs=true` 分支的路径）。
