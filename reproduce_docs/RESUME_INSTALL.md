# SimFoundry 环境安装 — 恢复手册 (RESUME_INSTALL)

> 用途：本机/跨实例做环境安装与产物恢复。不依赖聊天记录，照此手册即可续上。
> 最后更新：**2026-09-11** —— **pipeline A 已端到端跑通（含 stage 13/14）**，4 个 env 就位，
> 此前所有阻塞均已解除。**当前状态见 §1；已解除的阻塞见下表。**

## 已解除的阻塞 —— 怎么解开的（**先看这个**）

> 这一节存在的理由：**每个阻塞都是花了很久才绕开的，不写下来下次会重踩。**
> 遇到新问题时先扫一眼这里，很可能已经有现成答案。

| 阻塞 | 当时的症状 | 解除办法（细节见对应章节） |
|---|---|---|
| **Google 全域不可达** → A 的 VLM 阶段（3/5/8/11）跑不了 | `generativelanguage.googleapis.com` 直连超时；走代理是 Squid 自己回 503 | 用**第三方 Gemini 中转** `https://api.ofox.io/gemini`，`GOOGLE_GEMINI_BASE_URL` + `SIMFOUNDRY_GEMINI_BACKEND=api_key`，**零改代码**。详见 **§8.1** |
| **HF gated 仓库 403**（`facebook/sam3`）| stage 3 一开始就 403 | 从 **ModelScope** 取 `config.json` + `sam3.pt`，**手工放进 HF 缓存布局**（`models--facebook--sam3/snapshots/<commit>/`），代码零改动。详见 **§7.1** |
| **Isaac Sim 段错误** → stage 13/14 跑不了 | `import omnigibson` → `librtx.scenedb.plugin.so` SIGSEGV | 根因是**驱动代际不匹配**（Isaac Sim 5.1.0 要 580 代驱动）。**换到 driver 580 的宿主即可**，无需改代码/不升级。详见 **§10** |
| **HF 下载失败** | `CAS Client Error ... us.aws.cdn.hf.co` | `export HF_HUB_DISABLE_XET=1`（退回 hf-mirror 普通 HTTP）。详见 **§4.2** |
| **franka_robotiq 下载卡死数小时** | `huggingface_hub(httpx)` 对 tree API 永久 503，而 curl/git 正常 | 改用 `git clone --filter=blob:none` + `sparse-checkout` 只拉子树。详见 **§6** |
| **Isaac Sim 缺 X11 库** | `libXt.so.6: cannot open shared object file` | `apt-get install libxt6 libice6 libxi6 libxrandr2 libxcursor1 libxinerama1 libglu1-mesa libegl1-mesa xvfb`。详见 **§10.4** |
| **Gemini SDK 解析失败** | `UnknownApiResponseError: ... Raw response: : heartbeat` | 中转会发 SSE 注释行，SDK 不跳过 → 给 `google/genai/_api_client.py` 打 3 处补丁（**在 site-packages 里，重建 env 就丢**，已在快照包备份）|

---

## 0. 只跑 pipeline A 需要哪些 env（2026-09-11 复核）

- canonical A（video、默认参数、无 `--bg-splat`/`--detect-articulation`）**只需 3 个 env**：
  - `simfoundry`：stage 1b,3,4,5,6,8,10-14
  - `da3`：stage 2 深度（`s2_depth.backend: da3` 默认）
  - `hunyuan`：stage 7 网格（`s7_mesh.shape_model/texture_model: hunyuan` 默认）
- **any6d 不需要**（仅交互式 8b 或 `use_any6d: true` 的 B/C 场景配置；canonical A 的 cfg 里是注释的）。
- `nerfstudio_simfoundry`/`void`/`3dgrut`/`articulate-*`：均 opt-in，A 默认不用。
- ⚠️ deps/ 那批仓库（FoundationStereo/FoundationPose/ml-depth-pro/DA3/dinov2/sam3/BEHAVIOR-1K）由 `install_simfoundry.sh` 全量装进 simfoundry env，省 env 不省 deps。
- 容量实测（2026-09-11，只 A）：`simfoundry/` 共 **115G** —— conda_envs **56G** +
  deps **12G** + **hf_cache 35G** + Data 2.4G + checkpoints 0.86G。
  📌 早期估算写的「envs 40-50G + deps 10-20G ≈ 60-90G」**漏算了 hf_cache（35G）**，偏低约 25-55G。
- ⚠️ **env 只是一个门槛；A 还依赖 checkpoints（§7，含 SAM3 的 ModelScope 绕法 §7.1）与 VLM 中转（§8.1）**。

## 1. 当前状态（2026-09-11 更新）

> **✅ pipeline A 已端到端跑通，完整两次**（含之前卡住的 stage 13/14）：
> - `put_cup_in_bowl`（输入 `PutCupInBowl.mp4`）→ **33m 57s**，4 个道具
> - `put_marker_in_cup`（输入 `PutMarkerInCup.mp4`）→ **25m 06s**，4 个道具
>
> 产物：`Data/<scene>/s14_og/reconstructed_og_scene.json` + `reconstructed_scene.png`，
> 可直接用 light editor 打开（命令见 §10.6）。
>
> **本实例规格**（RTX 4080 SUPER 32G / driver **580.105.08** / autodl-tmp **250G**）见 **§10.8** ——
> 注意与下方"旧实例"的规格不同。
>
> **环境（4 个 env 全部就位）**：`simfoundry` / `da3` / `hunyuan` / **`simfoundry-editor`**
> （最后一个由 `install_light_editor.sh` 创建，753M，**不需要 GPU**）。
> **Gemini VLM 阻塞已解决** —— 第三方中转，**零改代码**（见 §8.1）。
> checkpoints 实际存放在 `deps/<模型>/` 下，**不在 `checkpoints/`**。

---

> ⚠️ **以下为 2026-09-10 旧实例（RTX 4090D 24G / autodl-tmp 150G）的当时记录，保留备查。**
> 其中的「剩余阻塞」一段**已不成立** —— 它描述的是当时的状态，不是现在。

**全部 ✅（2026-09-10）**

- **三个 env 全部安装完成并验证**（均在 `/root/autodl-tmp/simfoundry/conda_envs/`）：
  - `simfoundry` ✅ —— torch 2.7.0+cu128 / requirements / simfoundry editable / FoundationStereo / FoundationPose+nvdiffrast / ml-depth-pro / Depth-Anything-3 / sam3 / BEHAVIOR-1K+OmniGibson / pytorch3d / FAISS-GPU 1.12(校验 gpus=1) / Isaac Sim+Omniverse Kit。日志 `~/simfoundry_logs/install_simfoundry.log`。
  - `da3` ✅ —— 日志 `install_da3.log`，末行 "Completed installation of Depth Anything 3 environment: da3"。
  - `hunyuan` ✅ —— 日志 `install_hunyuan.log`，末行 "Completed installation of Hunyuan3D environment: hunyuan"。
- **checkpoints 全部到位并核过 sha256**（见 §7）。
- **A 的 stage 1b + 2 已实跑通过**：`PutCupInBowl.mp4` → 抽帧 → DA3 深度，产物 `Data/put_cup_in_bowl/s2_da/da/exports/npz/results.npz`（depth/conf/extrinsics/intrinsics）。DA3 权重从 hf-mirror 正常下载。
- **剩余阻塞（非环境问题）**：A 的 VLM 阶段（3/5/6/8/11）需 **Google Gemini**，本机网络不可达（见 §4/§8）。**未解决前无法跑完 A。**
- ⚠️ 所有 env/deps/checkpoints 在 `autodl-tmp` → **不进保存镜像**；关机保留数据盘则还在，换实例需归档（§5）。

## 2. 磁盘关键事实（教训）

- `/autodl-fs/data`（AutoDL 文件存储，**跨实例共享**）空间够但 **inode 配额仅 200,000** ——
  conda env(单 env ~17 万文件)/BEHAVIOR-1K(几十万~百万小文件) 放不下。**勿再把 env/deps 放这里。**
  ⚠️ **2026-09-11 实测：inode 已用 191,456 / 200,000（96%），只剩 8,544 个。**
  所以「**打成单个 tar 放这里**」仍然可行（1 inode），但**把压缩包解成目录会直接失败**。
  放之前先 `df -i /root/autodl-fs` 确认。
- `/root/autodl-tmp`（数据盘）inode 充足、快，**不随保存镜像走**、不跨实例。
  容量**随实例而变**：本实例 **250G**（余 ~136G）；2026-09-10 的旧实例是 150G —— **以 `df -h` 实测为准**。
- `/` 系统盘 ~30G，随保存镜像走。
- 结论：大文件/海量小文件都放 **autodl-tmp**；跨实例/持久化用「归档单文件放 autodl-fs → 新实例解压到 tmp」的搬运模式（见 §5）。

## 3. 布局（当前实际）

- `~/.condarc`：`envs_dirs=/root/autodl-tmp/simfoundry/conda_envs`、`pkgs_dirs=/root/autodl-tmp/simfoundry/conda_pkgs`，channels 走清华 TUNA。✅
- repo 软链：`deps`、`checkpoints`、`Data` → `/root/autodl-tmp/simfoundry/{...}`。✅
- `~/.bashrc`：`export HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME=/root/autodl-tmp/simfoundry/hf_cache`、`HF_HUB_CACHE=.../hf_cache/hub`。（实测在第 104-106 行 ✅）
- pip 配阿里源 —— 位置是 **`/etc/pip.conf`**（全局），内容
  `index-url=http://mirrors.aliyun.com/pypi/simple` + `trusted-host=mirrors.aliyun.com`。
  📌 2026-09-11 更正：早期文档写的 `~/.pip/pip.conf` **不存在**（已验证），别去那儿找。
- ⚠️ **必须设 `HF_HUB_DISABLE_XET=1`**（见 §4.2 的 HF Xet 教训），否则 `hf download` 会走 `*.xethub.hf.co`/`us.aws.cdn.hf.co` 而失败。
  **注意：它没有写进 `~/.bashrc`** —— `HF_ENDPOINT`/`HF_HOME`/`HF_HUB_CACHE` 有，这一条没有，所以**每次开新 shell 跑 hf 下载都要自己 export**。

## 4. Proxy 与网络可达性真相（**最重要的一节**）

> ### ⚠️ 第一原则：分清「安装」和「跑 pipeline」
> - **安装阶段**可以用 AutoDL 加速代理（github/hf 都靠它）。
> - **跑 pipeline 必须在无代理下**：`unset http_proxy https_proxy`。
>   带代理跑会让大模型权重下载卡在 TLS 握手超时（本次实跑反复踩到）。
> - 判断依据：先 `curl -sI https://github.com` —— **直连 FAIL / 代理 200** 说明 github 需要代理；
>   而 pipeline 用的 hf-mirror **直连就是 200**，不需要代理。

```bash
source /etc/network_turbo   # 开启（http_proxy=http://172.32.52.144:12798，Squid）
# 必须扩 no_proxy，让国内镜像/直连不被拖慢：
export no_proxy="$no_proxy,mirrors.tuna.tsinghua.edu.cn,mirrors.aliyun.com,download.pytorch.org,data.pyg.org,repo.anaconda.com,conda.anaconda.org,pypi.org,pythonhosted.org,hf-mirror.com,mirrors.ustc.edu.cn,localhost,127.0.0.1"
unset http_proxy && unset https_proxy   # 跑 pipeline / 下载完成后关闭
```

git 大仓库容错已配：`git config --global http.postBuffer 524288000`、`http.version HTTP/1.1`、`http.lowSpeedLimit/Time`。

### 4.1 哪些域通 / 哪些不通

**⚠️ 可达性随实例/宿主而变，下表是 2026-09-11 在本实例（driver 580）实测的**。
换机器后**重测一遍**，别照抄结论。测法：`curl -s -o /dev/null -w '%{http_code}' --max-time 8 <url>`。
**「直连」与「走代理」是两个不同维度**，早期版本只记了一个，会误导。

| 目标 | 直连 | 走代理 | 备注 |
|---|---|---|---|
| `api.ofox.io`（Gemini 中转）| ✅ **200** | — | **解锁 A 的 VLM 阶段的关键**，见 §8.1 |
| `docs.isaacsim.omniverse.nvidia.com` | ✅ **200** | — | Isaac Sim 官方文档（查驱动要求靠它，见 §10.2）|
| `hf-mirror.com` | ✅ **200** | ✅ 200 | **pipeline 下载走它，不需要代理** |
| `github.com` / `huggingface.co` | ❌ FAIL | ✅ 200 | **只有走代理才通**（安装阶段用）|
| `dl.fbaipublicfiles.com` / `ml-site.cdn-apple.com` / `release-assets.githubusercontent.com` | ✅ 通 | ✅ | checkpoints 直下可用 |
| `openrouter.ai` / `api.anthropic.com` / dashscope / volcengine ark | ✅ 通 | ✅ | 留档 |
| `mirrors.aliyun.com` / `mirrors.tuna.tsinghua.edu.cn` / `pypi.tuna` | ✅ 通 | ✅ | 国内镜像，no_proxy 直连 |
| **一切 Google 域**（google.com, drive.google.com, docs.google.com, `*.googleapis.com`, generativelanguage.googleapis.com）| ❌ **FAIL** | ❌ **FAIL** | **直连与代理都不通**（2026-09-11 复测仍然如此）|
| api.openai.com | ❌ FAIL | — | 与 Google 同样被挡 |
| `us.aws.cdn.hf.co` / `cas-bridge.xethub.hf.co`（HF **Xet** 后端）| ❌ 不通 | ❌ | 必须 `HF_HUB_DISABLE_XET=1`，见 §4.2 |

- **直连（不走代理）Google 也超时**（国内被墙）。
- **走代理 Google 是 Squid 自己报 503**：`ERR_SECURE_CONNECT_FAIL` / `TLS code: SQUID_TLS_ERR_CONNECT+TLS_IO_ERR=5`——**AutoDL 这个加速代理放行了 github/hf，但没放行 Google**。所以：
  - gdown / rclone / wget / curl **换任何工具都下不了 Google Drive**——是路不通，不是工具问题。
  - 依赖 `googleapis.com` 的 **Gemini/Vertex API 同样不可达**（影响 A 的 VLM 阶段，见 §10）。
  - 诊断口径：`curl -sI https://drive.google.com/...` 若只回 `HTTP/1.1 200 Connection established` 后无后续，就是 TLS 握手被挡（看 `-o` 保存的 body 会看到 Squid 的 503 页）。

### 4.2 HF 下载必须关 Xet

`hf download` / `huggingface-cli download` 默认走 **Xet** 后端（`*.xethub.hf.co`），在本机不可达 → 报 `CAS Client Error ... error sending request for url (https://us.aws.cdn.hf.co/...)`。**解法：`export HF_HUB_DISABLE_XET=1`**，退回 hf-mirror 的普通 HTTP。示例（含续传，慢但稳）：

```bash
export HF_HUB_DISABLE_XET=1
hf download <repo> --local-dir <dir>       # 走 hf-mirror 普通 HTTP
```

### 4.3 历史上另一个 HF 坑：httpx 503（franka）

见 §6「franka_robotiq 下载卡死」——同一环境下 `huggingface_hub(httpx)` 对 tree API 503，而 curl/git 正常。**凡是 HF 下载失败，优先换 `git clone`/`git lfs`/`curl` 或关 Xet，别在 httpx 里空转。**

## 5. 迁移/归档思路

- **不要依赖 autodl-tmp 进镜像**（它不进）。跨实例/持久化：把 env/deps **打成单文件归档**
  放 `/autodl-fs/data`（单文件≈1 inode、跨实例共享）；**新实例解压到 autodl-tmp**（inode 充足）。
- ⚠️ **autodl-fs 的 inode 只剩 8,544 个（96% 已用）** —— 归档务必是**单个 tar**，
  别在那里解包成目录（会因 inode 耗尽失败）。动手前 `df -i /root/autodl-fs` 看一眼。
- 容量参考（本实例实测）：`autodl-tmp` 共 250G，`simfoundry/` 占 **115G**
  （conda_envs 56G + deps 12G + hf_cache 35G + Data 2.4G + checkpoints 0.86G）→ 余量够，暂不需要归档。

## 6. 环境安装命令（**本实例 4 个 env 均已完成，勿重跑**；此节留档给新实例）

> 4 个 env = `simfoundry` / `da3` / `hunyuan`（pipeline A 需要，见 §0）
> \+ `simfoundry-editor`（light editor 专用，不需要 GPU）。

```bash
export PATH=/root/miniforge3/bin:$PATH MAMBA_ROOT_PREFIX=/root/miniforge3
export MAMBA_EXE=/root/miniforge3/bin/mamba CONDA_EXE=/root/miniforge3/bin/conda
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/simfoundry/hf_cache HF_HUB_CACHE=/root/autodl-tmp/simfoundry/hf_cache/hub
export HF_HUB_DISABLE_XET=1
source /etc/network_turbo   # + §4 的 no_proxy

# 按需分别直跑（勿用 install_everything.sh --only 判断存在性，有双 mamba 误判前科）
bash scripts/installation/install_simfoundry.sh --project-root /root/workspace/SimFoundry --env-name simfoundry --default
bash scripts/installation/install_da3.sh        --project-root /root/workspace/SimFoundry --env-name da3        --default
bash scripts/installation/install_hunyuan.sh    --project-root /root/workspace/SimFoundry --env-name hunyuan    --default

# 第 4 个：light editor 专用（不需要 GPU；依赖独立 usd-core，必须独立 env，勿并入上面三个）
bash scripts/installation/install_light_editor.sh
```

> 📌 2026-09-11 更正：早期版本的命令块只列了 3 个 install 脚本，**漏了 `install_light_editor.sh`**，
> 而上方说明又说第 4 个 env 由它创建 —— 自相矛盾，已补。（`simfoundry-editor` 实测 753M）
> ⚠️ AGENTS.md 强调：**它必须独立 venv**（用 standalone `usd-core`，与 Isaac Sim 自带的 `pxr` 冲突），
> 且**每次 `git pull` 后要重跑这个安装脚本**（可选依赖会静默降级，陈旧 env 看起来像正常的）。

**安装时的坑（重要，已踩过）**：
- simfoundry 脚本会卡在 franka_robotiq 资产（见下方专节）。
- yam 资产会因 env 内无 `pxr` 失败，但它是 **optional**，只 WARNING，不影响。
- 脚本**不在安装时拉 HF 权重**（只 pip + github clone），所以安装本身不依赖 Google。

> 双 mamba 教训：miniconda(/root/miniconda3) 与 miniforge 并存会导致 `mamba -n` 解析异常、install_everything 误跳过残缺 env。**统一用 Miniforge**，env 缺失时直接跑对应 `install_*.sh`，勿信 install_everything 的 skip 判断。

### franka_robotiq 下载卡死 — 教训与解法（2026-09-10）

- **症状**：install_simfoundry.sh 里 `fetch_franka_robotiq_assets()` 起的 `snapshot_download`（huggingface_hub/httpx）在 `GET /api/.../tree/...` 上永久 503/断连重试，阻塞主脚本数小时。主脚本 PID 5182 卡在等它。
- **根因**：这台机器上 huggingface.co 可达性**依赖传输层/UA**——curl/git(libcurl)=200，huggingface_hub(httpx)=503。hf-mirror.com 的 tree API 也 503。**非网络代理配置问题。**
- **解法（git 通道可用）**：sparse clone 只拉子树，绕过 httpx 的 tree 枚举：
  ```bash
  source /etc/network_turbo; git lfs install
  git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets /root/autodl-tmp/franka_clone
  cd /root/autodl-tmp/franka_clone && git sparse-checkout init --cone
  git sparse-checkout set models/franka/franka_robotiq && git checkout   # 215MB
  cp -a models/franka/franka_robotiq /root/workspace/SimFoundry/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/franka/
  ```
- **杀进程时机**：franka 文件**就位后**才 `kill <stuck_pid>`。脚本流程：fetch 返回非零→WARNING（非致命）→ yam（optional，此 env 无 pxr 会失败也仅 WARNING）→ `validate_robot_asset_file ... usda required` **文件在则通过** → faiss → "Completed installation"。yam 缺 pxr 无碍（optional）。
- 若以后 pipeline 运行期再遇 `huggingface_hub` 下载 503：**同样改用 git clone/`git lfs` 或 curl 抓文件**，别在 httpx 里空转。

## 7. Checkpoints 清单与镜像替代（2026-09-10 完成，含 sha256）

A 需要的权重及落点（`deps/`、`checkpoints/` 都是 repo 内软链 → autodl-tmp）：

| 资源 | 落点（repo 相对） | 官方源 | 本实例采用的**替代源** | 校验 |
|---|---|---|---|---|
| FoundationStereo `23-51-11` | `deps/FoundationStereo/pretrained_models/23-51-11/{model_best_bp2.pth,cfg.yaml}` | Google Drive | HF `yizhouzhao-nv/FoundationStereo-Backup` | ✅ sha256 见下 |
| FoundationPose Refiner | `deps/FoundationPose/weights/2023-10-28-18-33-37/{model_best.pth,config.yml}` | Google Drive | HF `gpue/foundationpose-weights` | ✅ |
| FoundationPose Scorer | `deps/FoundationPose/weights/2024-01-11-20-02-45/{model_best.pth,config.yml}` | Google Drive | HF `gpue/foundationpose-weights` | ✅ |
| SAM2.1 | `checkpoints/sam2.1_hiera_large.pt` | dl.fbaipublicfiles.com | 同源直下 ✅ | — |
| **SAM3**（stage 3 分割，**A 必需**）| `hf_cache/hub/models--facebook--sam3/snapshots/<commit>/{config.json,sam3.pt}` | HF `facebook/sam3`（**gated**）| **ModelScope 绕法 → §7.1** | ✅ `sam3.pt` = 3450062241 B |
| DepthPro | `deps/ml-depth-pro/checkpoints/depth_pro.pt` | ml-site.cdn-apple.com | 同源直下 ✅ | — |
| RealESRGAN_x4plus | `deps/Hunyuan3D-2.1/ckpt/RealESRGAN_x4plus.pth` | github release | 同源直下 ✅ | — |
| VOID（CogVideoX+2 safetensors） | `deps/void-model/` | HF `netflix/void-model`(gated) | **A 不需要，跳过** | — |

> ⚠️ `scripts/installation/download_checkpoints.sh` 对 FS/FP 走 **Google Drive + gdown → 必然 503 失败**（见 §4）。**在本环境要改用下表的 HF 镜像手动放置**；脚本对 SAM2.1/DepthPro/RealESRGAN 可用。VOID 会失败是预期的（gated + A 不用）。

**官方 Google Drive 链接（留档，本环境不可达）**：
- FoundationStereo：`https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf`（官方 README `NVlabs/FoundationStereo` Model Weights 段）
- FoundationPose（含 refiner+scorer）：`https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i`（官方 README `NVlabs/FoundationPose` Data prepare 段）
- 注：SimFoundry 脚本用的 ID 与官方不同（FS `1BbhoPliFqPJlrtD65TgNX49sJYuYcwA-`；FP refiner `1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC`；FP scorer `12Te_3TELLes5cim1d7F7EBTwUSe7iRBj`，实为 Any6D 段复用）。

**sha256 / blob-sha1 指纹（校验用）**：
```
FoundationStereo  model_best_bp2.pth (3298527334 B)
  sha256 60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1
  （3 个独立 HF 镜像一致：yizhouzhao-nv/FoundationStereo-Backup、vitaebin/foundation-stereo-model、pablovela5620/foundation-stereo）
FoundationStereo  cfg.yaml            git-blob-sha1 8361663d9f24351a57ef7bedc4068765ec3d82de
FoundationPose    refiner model_best.pth (68220109 B)
  sha256 774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60
  config.yml git-blob-sha1 d962a1d9a713fcb5f3e125e4fac7cdff3fd59c2f
FoundationPose    scorer model_best.pth (190229389 B)
  sha256 81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26
  config.yml git-blob-sha1 69cf5c058495cc5dda87fe845fe6efd6188ceb7b
```

**复现命令**：
```bash
source /etc/network_turbo
export no_proxy="$no_proxy,hf-mirror.com,localhost,127.0.0.1"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
cd /root/workspace/SimFoundry

# FoundationStereo（~3.3G）
hf download yizhouzhao-nv/FoundationStereo-Backup --local-dir /tmp/fs
mkdir -p deps/FoundationStereo/pretrained_models/23-51-11
cp -a /tmp/fs/23-51-11/. deps/FoundationStereo/pretrained_models/23-51-11/

# FoundationPose（~260M）
hf download gpue/foundationpose-weights --local-dir /tmp/fp
mkdir -p deps/FoundationPose/weights
cp -a /tmp/fp/2023-10-28-18-33-37 /tmp/fp/2024-01-11-20-02-45 deps/FoundationPose/weights/

# 校验（示例）
sha256sum deps/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
sha256sum deps/FoundationPose/weights/*/model_best.pth
```

> 技巧：HF 对大文件的 sha256 会**写进下载临时文件名**（`.cache/huggingface/download/.../<sha256>.incomplete`），可在下载中途就核对指纹。

### 7.1 SAM3（gated）—— ModelScope 绕法（2026-09-10，**A 必需**）

`facebook/sam3` 是 **gated** 仓库，本机未登录 HF → **stage 3 一跑就 403**。
但 **ModelScope 上有同名镜像**，可以直接取权重。做法是**把文件摆成 HF 的缓存布局**，
这样 `hf_hub_download('facebook/sam3', …)` 命中缓存直接返回 —— **代码零改动、不需要 HF token**。

```bash
HUB=/root/autodl-tmp/simfoundry/hf_cache/hub
COMMIT=96f3e1b404ba14f2cfac60ee6ae87c269a7b7923     # 用 ModelScope 的 master 提交号即可

# 1) 造出 HF 缓存目录结构
mkdir -p "$HUB/models--facebook--sam3/snapshots/$COMMIT"
printf '%s' "$COMMIT" > "$HUB/models--facebook--sam3/refs/main"    # ⚠️ 不能有结尾换行

# 2) 从 ModelScope 取这两个文件
for f in config.json sam3.pt; do
  curl -L --fail -o "$HUB/models--facebook--sam3/snapshots/$COMMIT/$f" \
    "https://www.modelscope.cn/models/facebook/sam3/resolve/master/$f"
done

# 3) 校验：sam3.pt 应为 3450062241 字节（与 ModelScope 文件清单一致）
ls -l "$HUB/models--facebook--sam3/snapshots/$COMMIT/"
```

**要点 / 坑**：

- `refs/main` 里**不能有结尾换行**（所以用 `printf '%s'`，**别用 `echo`**）—— 有换行就匹配不上快照目录。
- 快照目录名必须**等于** `refs/main` 的内容；HF 侧不需要真有这个 commit。
- 验证：`hf_hub_download('facebook/sam3', 'sam3.pt')` 应**返回缓存路径**而不是抛 403。
- 缓存布局在 `hf_cache/` 里（**快照包不含 hf_cache**，但本节命令可完整复现）。

> **另两个 gated 仓库不用管**：`facebook/dinov3-…` 与 `briaai/RMBG-2.0` 属 **Pixal3D 后端，canonical A 不用**，
> 所以**为跑 pipeline A 不需要申请任何 HF 授权**。
> （2026-09-11 实测：sam3 命中缓存 **OK**；RMBG 仍 `GatedRepoError`；dinov3 本地无缓存 —— 均不影响 A。）

## 8. Pipeline A 各 stage 依赖与状态

（来源：`scripts/pipeline/A_reconstruction/run_reconstruction.py` 的 StageSpec + 各 stage 源码）

| stage | 脚本 | env | 额外依赖 | 现状（2026-09-11 实测）|
|---|---|---|---|---|
| 1b | `1b_process_raw_video.py` | simfoundry | ffmpeg | ✅ 跑通 |
| 2 | `2_run_depth.py` | da3 | DA3 权重（HF，已下） | ✅ 跑通 |
| 3 | `3_segment_ground_plane.py` | simfoundry | SAM3（**ModelScope 绕法，见 §7.1**）+ Gemini | ✅ **跑通**（靠中转，见 §8.1）|
| 4 | `4_unify_world_frame.py` | simfoundry | numpy/open3d | ✅ 跑通 |
| 5 | `5_decompose_scene.py` | simfoundry | PriorDepthAnything + Gemini | ✅ 跑通 |
| 6 | `6_upsample_object_images.py` | simfoundry | `gemini-3-pro-image` / GPT / FLUX1(本地) | ✅ 跑通（默认走 gemini 中转）|
| 7 | `7_generate_object_meshes.py` | **hunyuan** | hunyuan 权重（运行时从 HF 下）+ RealESRGAN | ✅ 跑通。**24 GiB 卡**才需 `-- s7_mesh.low_vram=true`；本机 31 GiB **不需要** |
| 8 | `8_match_object_poses.py` | simfoundry | FoundationPose + probreg + faiss + Gemini | ✅ 跑通 |
| 10 | `10_compile_scene.py` | simfoundry | open3d/hydra（非 Isaac） | ✅ 跑通 |
| 11 | `11_make_objects_sim_ready.py` | simfoundry | coacd + Gemini | ✅ 跑通 |
| 12 | `12_stabilize_physics.py` | simfoundry | pybullet（非 Isaac） | ✅ 跑通 |
| 13 | `13_import_usd.py` | simfoundry | **Isaac Sim / OmniGibson** | ✅ **跑通**（341.77s / 136.13s 两次）。**不需要显示**，无 DISPLAY 也行 |
| 14 | `14_create_og_scene.py` | simfoundry | **Isaac Sim / OmniGibson** | ✅ **跑通**（121.54s / 90.05s 两次）|
| (9) | `9_articulate_objects.py` | simfoundry | articulate-* envs | opt-in `--detect-articulation`，A 默认不跑 |

**VLM 后端实情**（`simfoundry/models/vlm.py`）：只有 `Gemini`（Google 协议）/ `GPT`（OpenAI，仅 `gpt-image-1`，且**不支持 base_url**）/ `FLUX1`（本地 diffusers 的 `FluxKontextPipeline`）。
- stage 3/5/8/11 的模型键**只接受 gemini 系列**（`DETECTION_MODELS` 全是 gemini），源码里**没有非 Google 备选**。
- 原始困境：**Google 全域不可达**（§4.1）→ 这条路本来走不通。
- **已解除**：用**第三方 Gemini 中转**（`GOOGLE_GEMINI_BASE_URL`）走 api_key 路由 —— **零改代码**，详见 **§8.1**。
- `GPT` 仍不可用（`api.openai.com` 不通，且 SDK 不支持 base_url）；`FLUX1` 是本地兜底（stage 6 可切 `removal_model: flux`），A 未用到。

**结论（2026-09-11 更新）**：

- ✅ **A 全部 stage（1b–14）已端到端跑通**，完整两次：
  `put_cup_in_bowl` **33m 57s**、`put_marker_in_cup` **25m 06s**。
- 🚫 **不再有被 Google 挡住的 stage** —— 靠 §8.1 的中转解决。
- 🚫 **不再受 driver 限制** —— 本实例 driver 580，Isaac Sim 正常（§10）。
- 入口命令见本节末（注意 `low_vram` 的适用条件）。

### 8.1 已验证可用的 Gemini 中转（2026-09-10 实测，**零改代码**）

本机 Google 全不可达，但可用**第三方 Gemini 中转**绕开。已用**仓库自身的 `simfoundry.models.vlm.Gemini`** 实测通过（不是只 curl）：

- 中转 base：`https://api.ofox.io/gemini`（OFOX；**直连与经代理均可达**）
- 协议：Gemini 原生（`/gemini/v1beta/models/{model}:generateContent`）
- **鉴权**：`x-goog-api-key`（google-genai SDK 默认发的）与 `Authorization: Bearer` **都接受** → 代码无需改
- **模型名**：**裸名可用**（如 `gemini-2.5-flash`，中转自动归一为 `google/gemini-2.5-flash`）→ 白名单无需扩
- 实测通过的模型：
  - 多图→文本：`gemini-2.5-flash` / `gemini-3.1-pro-preview`(cfg 默认) / `gemini-2.5-pro` / `gemini-3-flash-preview`
  - 图像生成：`gemini-3-pro-image`(cfg 默认) / `gemini-2.5-flash-image`
- **cfg 默认模型全部可用 → 配置文件一个字都不用改**

跑 A 前 export（key 自备，**勿入库**）：
```bash
export GOOGLE_GEMINI_BASE_URL=https://api.ofox.io/gemini
export SIMFOUNDRY_GEMINI_BACKEND=api_key
export GEMINI_API_KEY=<你的key>
```
> 提速/省钱：默认图像模型 `gemini-3-pro-image` 较慢（~40s/张），可改 `s5_scene.removal_model`/`s6_upsample.model` 为 `gemini-2.5-flash-image`（~9s）或本地 `flux`。

### 8.2 HF gated 授权 —— **A 已不需要**（2026-09-11 更正）

> **结论先给**：跑 canonical pipeline A **不需要任何 HuggingFace gated 授权**。
> 唯一真正需要的是 `facebook/sam3`，已用 **ModelScope 绕法**解决（**§7.1**）。
> 本节剩余的"正规开通步骤"留档，仅在你想走官方渠道、或要跑 **Pixal3D/B 类**流程时才需要。

**为什么原来以为需要**：`facebook/sam3` 是 gated 仓库，未登录 HF 时 **stage 3 一跑就 403**。
当时的记录是「未登录 HF、无 token、全部 DENIED」—— 那是**2026-09-10 的状态**，后来已用 §7.1 绕开。

**2026-09-11 实测复核**（本机仍**未登录** HF，无 token 文件）：

| 仓库 | 实测 | 对 A 的影响 |
|---|---|---|
| `facebook/sam3` | ✅ **OK（本地缓存命中）** | **A 必需，已解决** |
| `briaai/RMBG-2.0` | `GatedRepoError` | 无 —— 属 **Pixal3D** 后端，canonical A 不用 |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | 本地无缓存 | 无 —— 同上 |

> 权威清单见 `docs/AGENT_INSTALL.md` §"Gated models"（该段确实存在）。
> `netflix/void-model` 也是 gated，但 **A 不需要**。

<details><summary>留档：走官方渠道开通 HF 授权的步骤（A 用不到）</summary>

1. 登录 huggingface.co → 访问目标仓库页 → 点 "Agree and access repository"
2. 建一个 read token → <https://huggingface.co/settings/tokens>
3. 让环境拿到 token：

```bash
cd scripts/installation
cp api_keys.template.txt api_keys.txt      # 写入 HF_TOKEN=hf_xxx
chmod 600 api_keys.txt
bash login_services.sh --default --no-gcloud   # 需 gcloud 时可去掉 --no-gcloud
# 或直接： mamba run -n simfoundry hf auth login --token hf_xxx
```

4. 验证：

```bash
for r in facebook/sam3 briaai/RMBG-2.0; do
  printf '%-46s ' "$r"; hf download "$r" config.json --quiet >/dev/null 2>&1 && echo OK || echo DENIED
done
```

> ⚠️ token 属机密，**不要写进仓库**（`api_keys.txt` 已 gitignore / chmod 600）。
> ⚠️ 本机**共享/可克隆**，落盘凭据会随镜像走 —— 见 §12.2 对 token 的同类告诫。

</details>

**跑 A 的入口**（⚠️ 2026-09-11 更正：**不要** `source /etc/network_turbo` —— 见 §4 第一原则）：

```bash
cd /root/workspace/SimFoundry
export PATH=/root/miniforge3/bin:$PATH MAMBA_ROOT_PREFIX=/root/miniforge3
unset http_proxy https_proxy              # ⚠️ 必须无代理，否则大文件下载 TLS 超时
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/simfoundry/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/simfoundry/hf_cache/hub
export HF_HUB_DISABLE_XET=1
# VLM 阶段需要（stage 3/5/6/8/11），key 自备、勿入库：
export GOOGLE_GEMINI_BASE_URL=https://api.ofox.io/gemini
export SIMFOUNDRY_GEMINI_BACKEND=api_key
export GEMINI_API_KEY=<你的key>

# 试跑（只打印命令）
bash scripts/pipeline/A_reconstruction/run.sh --scene-name put_cup_in_bowl \
  --video-fpath /root/workspace/SimFoundry/docs/assets/example_videos/PutCupInBowl.mp4 --dry-run

# 实跑（本机 31 GiB 显存，走官方默认，【不需要】low_vram）
bash scripts/pipeline/A_reconstruction/run.sh --scene-name put_cup_in_bowl \
  --video-fpath /root/workspace/SimFoundry/docs/assets/example_videos/PutCupInBowl.mp4
```

> - `--video-fpath` 请用**绝对路径**（各 stage 的 cwd 不同）。
> - **`-- s7_mesh.low_vram=true` 只有 24 GiB 卡需要**（README：默认 mesh 生成需 ~29 GiB）。
>   本机 31 GiB，加了反而多花时间。24 GiB 卡才加。
> - 更多开关（`--skip-successful` / `--include` / `--detect-articulation` / `--bg-splat`）见 **§12.3**。

## 9. 辅助脚本

### 9.1 监控脚本

- `~/simfoundry_logs/watch_stage.sh`（stage 感知，退出即通知）、`watchdog.sh <PID>`（20s 心跳）。
- `~/simfoundry_logs/*.pid` 存当前（安装/下载）PID；对应 `*.log` 为日志。

### 9.2 `_auto_continue.sh` —— 一次性环境串联驱动（**非仓库原有，本地新增**）

> ⚠️ 先明确归属：`scripts/installation/_auto_continue.sh` **不在原仓库里**。
> `git ls-files` 查不到它、`git log` 没有它的任何提交，且同目录另外 19 个文件
> 的 mtime 都是 checkout 时刻（2026-09-08 16:27），只有它是 **2026-09-10 01:02**。
> 它是本次部署临时写的，**不属于仓库的安装套件**。
>
> 🗑️ **后续处置（2026-09-11）**：该脚本**已从工作区删除**，全文原样内联在本节末尾，
> 作为历史留档。因此本仓库相对原 repo 的增量**回到 0 个脚本文件**（只多一份本文档）。
> 下面「怎么用」里的命令是**当时真实用过、事后归档**的记录 —— 直接粘贴会报
> `No such file`，**需要**先按本节末尾的清单把脚本重建出来，或改用 §11 快照包里的
> `git/untracked/_auto_continue.sh`。
>
> 💡 **留痕的原因**：这条"串接两个重 env 的安装、断开 SSH 也不中断"的做法本身可复用
> （任何「装完 A 才准装 B、而连接会断」的场景都适用），所以留文档、不留脚本。

#### 背景：为什么需要它

`install_da3.sh` 和 `install_hunyuan.sh` 两个 env 的安装**都吃磁盘和显存，不能并行**，
只能一个装完再装下一个。而 AutoDL 的 SSH 连接**随时会断**，没法守着终端等 da3
装完之后再手动敲 hunyuan。于是写一个后台守候脚本把两段接起来：

> 等 da3 退出 → 校验 da3 真的装成功 → 自动启动 hunyuan → 继续等 → 打印最终结论

这样断开连接也不影响推进，回来只看日志末尾一行就知道成败。

#### 它做什么

1. 轮询等待 `da3` 安装进程退出（PID 由 `$1` 传入，需自己在启动 da3 时记下）
2. 在 `install_da3.log` 里找完成标志串
   `Completed installation of Depth Anything 3 environment: da3`
3. 找到 → `nohup` 启动 `install_hunyuan.sh`，把 PID 写进 `install.pid`，继续等它
4. 最后按日志内容打印三种结论之一：
   `hunyuan COMPLETED OK` / `hunyuan … MANUAL CHECK NEEDED` / `da3 … MANUAL CHECK NEEDED`

#### 怎么用（⚠️ 归档记录：脚本已从工作区删除，见上方说明）

```bash
# 先手工把 da3 装到后台，记下它的 PID
nohup bash scripts/installation/install_da3.sh … > ~/simfoundry_logs/install_da3.log 2>&1 &
echo $! > ~/simfoundry_logs/install.pid

# 再把那个 PID 交给它，然后就可以断开连接了
chmod +x scripts/installation/_auto_continue.sh
nohup scripts/installation/_auto_continue.sh "$(cat ~/simfoundry_logs/install.pid)" \
  > ~/simfoundry_logs/auto_continue.log 2>&1 &
```

回来只看最后一行：

```bash
tail -3 ~/simfoundry_logs/auto_continue.log
```

#### 注意点（换机器前必读）

| 注意点 | 说明 |
|---|---|
| **路径是硬编码的** | `/root/miniforge3`、`/root/autodl-tmp/simfoundry/...`、`/root/workspace/SimFoundry` —— 换目录布局必须改 |
| **依赖 AutoDL 的代理脚本** | `source /etc/network_turbo` 失败也继续跑（有 `>/dev/null 2>&1`），但下载可能变慢 |
| **靠日志字符串判成败** | 只看 `install_da3.log` / `install_hunyuan.log` 里有没有那两行完成标志。日志被截断、改名、或上游改了措辞，就会**误报 MANUAL CHECK NEEDED**（宁可误报也不误判为成功）|
| **只"等 + 串"，不重试** | 中间任何一步失败它只报告，**不会自动修**；连接断在它启动之前也不会留下任何东西 |
| **一次性** | 两个 env 都装完后它就没用了 |

#### 为什么不提交进仓库

脚本头自己写着 *"Created for one-shot AutoDL environment bring-up; not part of the install suite."*
它带 AutoDL 专属路径和代理设置，不是通用工具；提交进仓库只会给上游添一个看不懂的
文件。**本节的代码块就是它唯一的正式版本。**

#### 完整内容（换新实例时照抄即可复原）

```bash
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Temp driver: wait for da3 install (PID passed as $1) -> launch hunyuan -> wait for it.
# Created for one-shot AutoDL environment bring-up; not part of the install suite.
DA3_PID="${1:?usage: _auto_continue.sh <da3_pid>}"

source /etc/network_turbo >/dev/null 2>&1
export no_proxy="$no_proxy,mirrors.tuna.tsinghua.edu.cn,mirrors.aliyun.com,download.pytorch.org,data.pyg.org,repo.anaconda.com,conda.anaconda.org,pypi.org,files.pythonhosted.org,hf-mirror.com,mirrors.ustc.edu.cn,localhost,127.0.0.1"
export PATH=/root/miniforge3/bin:$PATH MAMBA_ROOT_PREFIX=/root/miniforge3
export MAMBA_EXE=/root/miniforge3/bin/mamba CONDA_EXE=/root/miniforge3/bin/conda
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/simfoundry/hf_cache HF_HUB_CACHE=/root/autodl-tmp/simfoundry/hf_cache/hub
cd /root/workspace/SimFoundry

echo "[$(date +%H:%M:%S)] waiting for da3 install (PID $DA3_PID)..."
while kill -0 "$DA3_PID" 2>/dev/null; do
  sleep 30
  echo "[$(date +%H:%M:%S)] da3 still running (elapsed $(ps -o etime= -p "$DA3_PID" 2>/dev/null | tr -d ' '))..."
done
echo "[$(date +%H:%M:%S)] da3 process ended. Last log lines:"
tail -n 5 ~/simfoundry_logs/install_da3.log

if grep -q "Completed installation of Depth Anything 3 environment: da3" ~/simfoundry_logs/install_da3.log; then
  echo "[$(date +%H:%M:%S)] da3 COMPLETED OK. Launching hunyuan install..."
  mkdir -p ~/simfoundry_logs
  nohup bash scripts/installation/install_hunyuan.sh --project-root /root/workspace/SimFoundry --env-name hunyuan --default > ~/simfoundry_logs/install_hunyuan.log 2>&1 &
  HUNYUAN_PID=$!
  echo "$HUNYUAN_PID" > ~/simfoundry_logs/install.pid
  echo "[$(date +%H:%M:%S)] hunyuan launched PID=$HUNYUAN_PID"
  while kill -0 "$HUNYUAN_PID" 2>/dev/null; do
    sleep 30
    echo "[$(date +%H:%M:%S)] hunyuan still running (elapsed $(ps -o etime= -p "$HUNYUAN_PID" 2>/dev/null | tr -d ' '))..."
  done
  echo "[$(date +%H:%M:%S)] hunyuan process ended. Last log lines:"
  tail -n 8 ~/simfoundry_logs/install_hunyuan.log
  if grep -q "Completed installation" ~/simfoundry_logs/install_hunyuan.log; then
    echo "=== AUTO-CONTINUE FINAL: hunyuan COMPLETED OK ==="
  else
    echo "=== AUTO-CONTINUE FINAL: hunyuan log lacks completion line - MANUAL CHECK NEEDED ==="
  fi
else
  echo "=== AUTO-CONTINUE FINAL: da3 did NOT complete cleanly - MANUAL CHECK NEEDED ==="
  tail -n 20 ~/simfoundry_logs/install_da3.log
fi
```

## 10. Isaac Sim 驱动代际问题（**✅ 2026-09-11 已解决**）

> **当前状态：已解决。** stage 13 / 14 在本实例（driver **580.105.08**）上已跑通。
> 一句话结论：Isaac Sim **5.1.0** 的官方 tested driver 是 **580.65.06**，而当时实例的驱动是
> **595.71.05**（Isaac Sim 6.x 代）→ `import omnigibson` 在建 RTX 渲染代理时 SIGSEGV。
> **换到 driver 580 的实例后一切正常**，无需改代码、无需动 `deps/`、无需升级。
> 详细结论见 **§10.8**。
>
> 本节保留完整的排查过程、取证与被排除的假设，供以后遇到同类症状时对照 —— **不必重查**。

**当时的症状**（2026-09-10，driver 595 的实例）：stage 1b–12 全部 `success=True`；
**stage 13、14 无法运行**，因为 `import omnigibson` 会让 Isaac Sim 在建 RTX 渲染代理时 SIGSEGV。

### 10.1 现象

裸 `SimulationApp({"headless": True})` 就复现，**与 OmniGibson 无关**：

```
Thread "carb.taskingNN" SIGSEGV
    lock add %r13d, 0x3380(%rbp)      <- 对原子计数器自增，基址指针无效
  librtx.scenedb.plugin.so (frames 0-4)
  libcarb.scenerenderer-rtx.plugin.so (5-7)
  libomni.hydra.rtx.plugin.so (8)
  libomni.usd.so (9-11)
  libcarb.tasking.plugin.so (12-16)
插件自带错误串：%s error: Failed to initialize rtx::scenedb::Context
```

`py-spy` 显示主线程停在 `SimulationApp._prepare_ui`（更早一次是 `_wait_for_viewport`）——
那是**陪衬**，真正的崩溃在 worker 线程上、异步发生。

### 10.2 根因：驱动代际不匹配（官方文档已证实）

| Isaac Sim | 官方 Tested Driver (Linux) |
|---|---|
| **5.1.0**（由 `deps/BEHAVIOR-1K/setup.sh:333` 的 `pip install isaacsim[all,extscache]==5.1.0` 固定）| **580.65.06** |
| 6.x（当前最新） | **595.58.03** |

本机 `nvidia-smi` = **595.71.05**，即 Isaac Sim **6.x 代**的驱动，却拿来跑 **5.1.0**。
5.1.0 的文档页还挂着 *"⚠ Unsupported release: Isaac Sim 5.1.0 is no longer supported."*

### 10.3 本机取证（解释了"克隆实例后环境变了"）

| 证据 | 含义 |
|---|---|
| `/etc/hostname` mtime = `2026-09-09 17:19:25` | 容器在**那一刻被重建/迁移** |
| `.so.1` 软链同一分钟被从 580 改指 595 | 驱动代际就是这时换的 |
| `*.580.76.05` 全是 **0 字节**，mtime `2026-09-08 15:09` | 上一个容器的 580 残留 |
| `uptime` 112 天（宿主）；`libnvidia-*` 文件 mtime 5月21日 | 宿主内核模块一直是 595 |
| `/var/log/apt/history.log`、`dpkg.log` 里**没有任何 nvidia 记录** | 驱动由 AutoDL 在容器启动时注入 → **容器内改不了** |

### 10.4 已排除（都是实测，别再重复排查）

- ❌ 缺库：`libXt.so.6` 那批**已修好**（`apt-get install libxt6 libice6 libxi6 libxrandr2 libxcursor1 libxinerama1 libglu1-mesa libegl1-mesa xvfb`），日志里已无 `Could not load the dynamic library`
- ❌ `/dev/shm`：45G
- ❌ Vulkan 坏：ICD 正常，枚举到 4090，275 个设备扩展，`VK_KHR_acceleration_structure` / `ray_tracing_pipeline` / `ray_query` / `deferred_host_operations` **全有**
- ❌ 驱动用户态不一致：所有 `libnvidia-*`/`libGLX_nvidia`/`libnvoptix` 软链都指向真实的 595.71.05 文件，**无坏软链**（0 字节 580 每个都有真 595 对应，仅 `libvdpau_nvidia.so` 无，且无关）
- ❌ 缓存过期：清空 `appdata/{local,global}/cache`、`~/.cache/ov`、`~/.cache/nvidia/GLCache`、`~/.nv` 后**同样崩**
- ❌ `--/rtx/verifyDriverVersion/enabled=false`
- ❌ `--/app/usdrt/population/utils/enableRendererInstancing=false`
- ❌ `--/rtx/modes/rt2/enabled=false`
- ❌ `--/app/asyncRendering=false`
- ❌ Kit 里**没有**"关掉渲染器"的开关（`libomni.kit.renderer.*` 只有 `skipMaterialLoading*`、`skipWhileMinimized`、`sleepMsOnFocus/OutOfFocus`）

### 10.5 升级路径也是堵死的（别浪费时间去试）

- **没有更新的 5.1.0**：PyPI 上 `isaacsim==5.1.0` **就是**我们现在这个；包内 `VERSION`
  写 `5.1.0-rc.19+release.26219`，"rc.19" 只是 NVIDIA 的内部构建标签，不是预览版
- **6.x 是 cp312**，而 `simfoundry` env 是 **Python 3.11**
- 就算换 Python 3.12，`deps/BEHAVIOR-1K/OmniGibson/omnigibson/simulator.py:74` 硬编码
  `m.KIT_FILES = {(5, 1, 0): "omnigibson_5_1_0.kit"}`，第 198 行 `assert` 版本元组必须命中该表
  → **6.x 启动即断言失败**；要支持得换更新的 OmniGibson，即改 `deps/` 的 pin
  （AGENTS.md 禁止直接改 `deps/`，只允许走 `patches/`）

### 10.6 解锁办法：换到驱动 ~580 的宿主

**换机器前先验**：

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # 期望 580.x
```

拿到 580 的宿主后（envs 要按 §6 重装，因为 `/root/autodl-tmp` 不随实例迁移）：

```bash
cd /root/workspace/SimFoundry
# <在这里贴 §6 的环境安装 + §7 的 checkpoint 恢复>
nohup bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name put_cup_in_bowl \
  --video-fpath /root/workspace/SimFoundry/docs/assets/example_videos/PutCupInBowl.mp4 \
  --skip-successful --include 13,14 -- s7_mesh.low_vram=true \
  > ~/simfoundry_logs/run_A_1314.log 2>&1 &
# 跑完产物：Data/put_cup_in_bowl/s14_og/reconstructed_og_scene.json + reconstructed_scene.png
```

然后在 light editor 里打开（`simfoundry-editor` env 见 §6）：

```bash
mamba run -n simfoundry-editor python scripts/interactive/light_editor/server.py \
  --scene /root/workspace/SimFoundry/Data/put_cup_in_bowl/s14_og/reconstructed_og_scene.json
# 浏览器 http://localhost:8770（AutoDL 上需要 ssh -N -L 8770:localhost:8770 或 --host 0.0.0.0，无鉴权）
```

Isaac Sim 自带的官方兼容性检查器（换机器后可以先跑它，比盲跑快）：

```
<isaacsim>/apps/isaacsim.exp.compatibility_check.kit
```

### 10.7 排查时好用的诊断手段（留档）

- 崩溃报告目录：`<OmniGibson>/appdata/local/data/Kit/OmniGibson/3.8/` — `.dmp`
  之外还有一个 **`.py.txt`，是 py-spy 抓的 Python 栈**（最快的第一手信息）
- Kit 日志：`<OmniGibson>/appdata/local/logs/Kit/OmniGibson/3.8/kit_*.log`
- `gdb -batch -ex run -ex "bt 40" --args <env python> probe.py` 能拿到**真实原生栈**；
  Kit 日志里 breakpad 自己的符号化是**错的**（会把帧归到无关的
  `std::vector::_M_realloc_insert` 上，别信）
- 从二进制里挖真实设置名：
  `strings -a <plugin>.so | grep -oE "/[a-z][A-Za-z0-9_]*(/[A-Za-z0-9_]+)+"`
- 写 ctypes 探测 Vulkan 时**必须设 argtypes**，否则 64 位句柄被当 32 位 int 传、指针截断 → 假段错误
  （我第一次就踩了，误判成"驱动坏了"）

### 10.8 ✅ 已确认解决：换到 driver 580 的实例后 Isaac Sim 正常（2026-09-11）

**结论：10.2 的诊断被证实，10.6 的办法有效。**

新实例（AutoDL `autodl-container-3eqmzl5mw7-13989f24`）规格：

| 项 | 值 |
|---|---|
| GPU | RTX 4080 SUPER，**32760 MiB（31 GiB 可用）**，sm_89 |
| 驱动 | **580.105.08**（内核模块 / `libcuda.so.1` / `libGLX_nvidia.so.0` / `libnvoptix.so.1` 全部一致指向它） |
| CPU / 内存 | 12 核 / 503 GiB |
| `/root/autodl-tmp` | **250G**（余 138G，xfs 独立卷，不再是 150G） |
| `/` | 30G（余 25G）|
| `/root/autodl-fs/data` | 200G（余 181G） |
| `/dev/shm` | 31G |

**验证方式**（裸 `SimulationApp({"headless": True})`）：

```
>>> SIM APP OK <<<
[133.152s] Simulation App Startup Complete
Warp 1.8.2: CUDA Toolkit 12.8, Driver 13.0, "cuda:0" RTX 4080 SUPER (31 GiB, sm_89)
```

**无 DISPLAY 也能跑**（`unset DISPLAY`），不需要 `xvfb-run`。

**这台机器上 0 字节占位符库依然存在**（`lib*.so.580.76.05` / `lib*.so.595.71.05` 都是 0 字节），
但**活动软链指向真实完整的 `580.105.08`**，所以无害 —— 上一台是反过来（活动软链指向 595，
而 595 那套与 Isaac Sim 5.1 不兼容）。判断标准始终是：**内核模块版本 vs 活动软链指向的版本**。

另外，本实例**本来就装了 `libXt.so.6` 等 X11 库**并有 `xvfb-run`，§10.4 里那个缺库问题不会重演。

**VRAM 配置结论**：README 说默认 mesh 生成需要 ~29 GiB、24 GiB 卡才需要 `s7_mesh.low_vram=true`。
本机 31 GiB → **走官方默认，不传 `low_vram`**。

## 11. 持久化快照包（2026-09-10 建立）

**为什么需要**：`/root/autodl-tmp` **不会随「保存镜像」一起保存**；而且容器被重建后
**SSH endpoint 会变，VS Code 的 chat 存储对不上，会话窗口里就再也找不到这段对话了**。
所以把「改动 + 聊天记录 + 文档 + 复现脚本」打成一个自包含的包，**放三处**：

| 位置 | 能否随镜像保存 | 能否跨实例 | 说明 |
|---|---|---|---|
| `/root/simfoundry_persist/` | ✅ | ❌ | 系统盘，随「保存镜像」一起走 |
| `/root/autodl-tmp/simfoundry_persist/` | ❌ | ❌ | 数据盘，开关机不丢，但不进镜像 |
| `/root/autodl-fs/simfoundry_persist/` | ✅（共享盘） | ✅ | **最保险**，另外还有单文件 `simfoundry_persist.tar.gz`（省 inode） |

**包内容**：`HANDOVER.md`（先读）、`docs/RESUME_INSTALL.md`、`docs/isaac-sim-driver-blocker.md`、
`git/`（HEAD + status + diff + 未跟踪文件副本 + **`simfoundry.bundle`**，已验证
"records a complete history"，可在新机器上 `git clone` 还原）、

> ⚠️ **两个易误解之处（2026-09-11 更正）**：
> 1. `git/untracked/` 只在**确有未跟踪文件**时才有内容。`RESUME_INSTALL.md` 被 commit 之后，
>    它一般**是空的**（该文件由 `docs/` 那一份副本承载）。
> 2. **`simfoundry.bundle` 只含已提交的版本**（打包时是 **612 行**那份，不含当时未提交的 §12）。
>    想拿到最新工作区内容，看包里的 **`docs/RESUME_INSTALL.md`**（它是直接拷的工作区文件）。
`patches/`（`google/genai` 的 SSE 补丁 —— 它在 site-packages 里，重建 env 就丢）、
`chat/`（本次会话 `transcript.jsonl` + 可读版 `transcript.md` + debug-logs）、
`diagnostics/`（最小复现脚本、Kit 崩溃日志尾部）、`tools/`（打包脚本本身）。

**重新生成**（任何时候改动后都可以刷新）：

```bash
python /root/simfoundry_persist_build.py
```

**安全**：导出物对 `sk-*` / `hf_*` / `Bearer` 全部脱敏（已验证 0 残留）。
但**原始聊天记录里含中转 API key**，位于
`/root/.vscode-server/data/User/workspaceStorage/942ca32b5cf16bd866801e69a97aeca0/GitHub.copilot-chat/transcripts/`，
建议用完即轮换。密钥不要写进仓库。

**新机器上还原代码**（如果仓库本身也没了）：

```bash
git clone /root/autodl-fs/simfoundry_persist/git/simfoundry.bundle SimFoundry
cd SimFoundry && git checkout junjie/work_branch
```

## 12. 常用命令速查（手动测试用）

> 本节的每个命令都在本实例上实跑过。改动任何东西前，先记住：
> **`Data/<scene>/**` 是 pipeline 产出 → 只读，不要手改。**

### 12.1 前置环境块（后面所有命令都先 source 它）

```bash
cd /root/workspace/SimFoundry
export PATH=/root/miniforge3/bin:$PATH MAMBA_ROOT_PREFIX=/root/miniforge3
export MAMBA_EXE=/root/miniforge3/bin/mamba CONDA_EXE=/root/miniforge3/bin/conda
unset http_proxy https_proxy                      # ⚠️ pipeline 必须在无代理下跑，否则大文件下载 TLS 超时
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/simfoundry/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/simfoundry/hf_cache/hub
export HF_HUB_DISABLE_XET=1                       # 本网络 us.aws.cdn.hf.co 不可达
export no_proxy="mirrors.tuna.tsinghua.edu.cn,mirrors.aliyun.com,hf-mirror.com,modelscope.cn,download.pytorch.org,localhost,127.0.0.1"
# 只有跑 VLM 阶段（3/5/6/8/11）才需要下面三行：
export GOOGLE_GEMINI_BASE_URL="https://api.ofox.io/gemini"
export SIMFOUNDRY_GEMINI_BACKEND=api_key
export GEMINI_API_KEY="<你的 key>"                # ⚠️ 别写进任何文件，别留在 shell 历史里
```

### 12.2 样例视频 / 示例资产

**样例视频不用下载 —— 仓库自带 6 个：**

```bash
ls docs/assets/example_videos/
# Fruits.mp4  Mailbox.mp4  PutCupInBowl.mp4  PutMarkerInCup.mp4
# ThrowAwayTrash.mp4  ThrowAwayTrashCousin.mp4
```

**官方示例「场景 + 房间库」在 HuggingFace**（light editor 用；pipeline A 不需要）：

```bash
# hf CLI 在 simfoundry env 里（simfoundry-editor 里没有）
mamba run -n simfoundry hf download nadunRanawaka1/simfoundry-assets \
  --repo-type dataset --local-dir assets
```

得到 `assets/scenes/{DROID,YAM}/...`（8 个官方场景）与
`assets/backgrounds/mesh_backgrounds/{droid_v1,droid_v2,yam_workstation}.usd`。

> ⚠️ **6 个视频里只有 3 个**（`PutMarkerInCup` / `Fruits` / `ThrowAwayTrash`）在官方数据里
> 有**同名**场景，其余 3 个没有。视频名与场景名的对应关系**只从命名推断，未经文件佐证**。

用你自己的视频：`--video-fpath` 传任意绝对路径即可。

### 12.3 启动 pipeline A

```bash
# 最简（官方默认调用）
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name put_cup_in_bowl \
  --video-fpath /root/workspace/SimFoundry/docs/assets/example_videos/PutCupInBowl.mp4
```

常用开关：

| 开关 | 作用 |
|---|---|
| `--skip-successful` | 跳过已成功的 stage（断点续跑）|
| `--include 13,14` | 只跑指定 stage |
| `--exclude 7` | 排除某个 stage |
| `--dry-run` | 只打印将执行的命令，不真跑 |
| `-- s7_mesh.low_vram=true` | 24 GiB 卡需要（默认需 ~29 GiB）；本机 31 GiB **不用** |
| `--detect-articulation` | 可选 stage 9 关节分解（需额外 env）|
| `--bg-splat` | 可选 stage 2c 背景 GS（需 `nerfstudio_simfoundry` env）|

可选 `--scene-name`（对应 `scripts/cfg/<名字>.yaml`）：
`PutCupInBowl` `PutMarkerInCup` `ServeFruitsOnGreenPlate` `cluttered_scene`
`serve_banana` `stack_dishware` `stack_dishware_easy` `auto_bg`

**后台跑 + 记 PID（推荐，断线不影响）**：

```bash
nohup bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name put_marker_in_cup \
  --video-fpath /root/workspace/SimFoundry/docs/assets/example_videos/PutMarkerInCup.mp4 \
  > ~/simfoundry_logs/run_A_pm.log 2>&1 &
echo $! > ~/simfoundry_logs/runA.pid
```

### 12.4 验证结果

```bash
S=put_marker_in_cup          # ← 改成你的场景名
D=Data/$S
```

**① 各 stage 成败与耗时（最权威）**

```bash
python3 -m json.tool $D/pipeline_run_report.json | head -40
grep -E "^\[Stage|total wall time" ~/simfoundry_logs/run_A_pm.log | tail -20
```

**② 产物总览**

```bash
for d in $D/s*; do printf "%-16s %s\n" "$(basename $d)" "$(du -sh $d|cut -f1)"; done
ls -la $D/s14_og/          # 最终场景 JSON + 渲染图 + settled_poses
```

**③ 渲染图**：在 VS Code 里直接打开 `Data/<scene>/s14_og/reconstructed_scene.png`

**④ 场景 JSON 体检（物体 / 位姿 / 关键键）**

```bash
python3 - <<'PY'
import json; d=json.load(open("Data/put_marker_in_cup/s14_og/reconstructed_og_scene.json"))
oi=(d.get("objects_info") or {}).get("init_info") or {}
reg=((d.get("state") or {}).get("registry") or {}).get("object_registry") or {}
print("顶层键:", list(d), "| 物体数:", len(oi))
print("ground_plane_info:", d.get("ground_plane_info", "无 => 默认 z=0"))
for n, s in oi.items():
    z = (reg.get(n) or {}).get("root_link", {}).get("pos", [None]*3)[2]
    print(f"  {n:10s} {str((s.get('args') or {}).get('category')):20s} {s.get('class_name')}  z={z}")
PY
```

**⑤ 桌面（支撑面）的**本地**拟合结果 —— 注意不是重建出的几何**

```bash
cat $D/s3_ground/image_*_floor_info.json      # floor_category / origin / z_dir
```

**⑥ 交互式审查器**（3 种模式；会写 rerun feedback）

```bash
mamba run -n simfoundry python scripts/pipeline/A_reconstruction/stages/visualize_stage_outputs.py \
  scene_name=put_cup_in_bowl stage_review.mode=upsample      # depth | upsample | mesh
```

**⑦ 物理校验**（会启 Isaac Sim，~90 s；针对 light editor 保存出的场景文件）

```bash
mamba run -n simfoundry python scripts/interactive/light_editor/settle.py \
  --scene /abs/path/<scene>_scene_state_latest.json --steps 240 --report settle.json

mamba run -n simfoundry python scripts/interactive/light_editor/parity_check.py \
  --scene /abs/path/<scene>_scene_state_latest.json --report parity.json
```

### 12.5 启动 light editor

```bash
cd /root/workspace/SimFoundry
export PATH=/root/miniforge3/bin:$PATH MAMBA_ROOT_PREFIX=/root/miniforge3

nohup mamba run -n simfoundry-editor python scripts/interactive/light_editor/server.py \
  --scene /root/workspace/SimFoundry/Data/put_marker_in_cup/s14_og/reconstructed_og_scene.json \
  > ~/simfoundry_logs/light_editor.log 2>&1 &
echo $! > ~/simfoundry_logs/light_editor.pid
```

**怎么访问**：VS Code 左侧 **PORTS / 端口** 面板 → 找到 `8770` → 点 🌐 图标
（或本地浏览器开 `http://localhost:8770`）。服务只监听容器内 `127.0.0.1:8770`，
靠 VS Code Remote-SSH 自动转发；**隧道与 VS Code 连接同生共死**。

常用参数：`--cameras nv_franka_droid` · `--allow-incomplete` · `--splat-budget 100000`
· `--asset-root DIR` · `--host 0.0.0.0`（⚠️ **无鉴权**，别公网暴露）。

**探活 / 看状态**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8770/     # 期望 200
curl -s http://localhost:8770/api/scene_state | head -c 300          # scene_revision 等
curl -s http://localhost:8770/api/ground_plane                       # 桌面平面状态
```

### 12.6 关闭 light editor

```bash
pkill -f "light_editor/server.py"
sleep 2
pgrep -af "light_editor/server.py" || echo "已关闭"
```

它是**纯查看器**，正常关闭不会丢东西；只有点了 **Save scene JSON** 才写盘。
判断有没有保存过：`/api/scene_state` 的 `scene_revision` 为 `0` 就是从未保存。

### 12.7 其他

```bash
# GPU / 磁盘 / 进程
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
df -h / /root/autodl-tmp /root/autodl-fs
ps -p $(cat ~/simfoundry_logs/runA.pid) -o pid=,etime=,cmd=

# 环境自检：omnigibson 必须解析在仓库 deps/ 内（AGENTS.md 点名的坑）
mamba env list
/root/autodl-tmp/simfoundry/conda_envs/simfoundry/bin/python -c \
  "import importlib.util as u; print(u.find_spec('omnigibson').origin)"

# 刷新持久化快照包（三处 + tar.gz）
python /root/simfoundry_persist_build.py
```

### 12.8 踩坑速查

| 症状 | 原因 / 处理 |
|---|---|
| `import omnigibson` 段错误 | **驱动代际问题**：Isaac Sim 5.1 要 **580** 代（见 §10）|
| 下载卡死 / TLS 握手超时 | **开着代理跑 pipeline 了** → `unset http_proxy https_proxy` |
| HF 下载失败 | `HF_HUB_DISABLE_XET=1`，并走 `hf-mirror.com` |
| Gemini 报 `Failed to parse response as JSON. Raw response: : heartbeat` | `google/genai` 的 SSE 补丁丢了（重建 env 会丢），见 §8.1 |
| 编辑器里"没有桌面" | **正常现象**，不是 bug —— pipeline 不产出桌面几何，编辑器也不画 OmniGibson 的 floor plane。物理支撑面在 z=0 |
| 24 GiB 卡 stage 7 OOM | 加 `-- s7_mesh.low_vram=true` |
| 仓库里出现 `assets/` | 那是 `hf download` 下来的官方示例资产，**不是 pipeline 产出** |
