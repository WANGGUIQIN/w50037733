# RoboBrain-3DGS 训练指南

## 1. 项目概述

基于 RoboBrain2.5-8B-NV (Qwen3-VL) 的 3D Gaussian Splatting 增强模型，用于机械臂操控任务。

```
RGBD 输入 -> DepthToGaussian -> GS Encoder (PointNet++) -> 3D Tokens
                                                             |
                                    [3D Tokens] + [Text Tokens] -> Qwen3-VL LLM -> Affordance/Constraint
```

**训练模式：**
- **LoRA 微调**：冻结 LLM，注入低秩适配器，0.42% 参数可训练（36.8M），单卡可跑
- **全量微调**：解冻 LLM 全部或最后 N 层，93% 参数可训练（8.2B），需要 4 卡 + ZeRO-3

---

## 2. 硬件需求

| 模式 | 最低 GPU | 推荐 GPU | 显存/卡 | DeepSpeed |
|------|---------|---------|---------|-----------|
| LoRA | 1x | 4x RTX 5090/A100 | >=24GB | ZeRO-2 |
| Full (last 8 layers) | 2x | 4x RTX 5090/A100 | >=32GB | ZeRO-2/3 |
| Full (all layers) | 4x | 4x A100 80GB | >=32GB | ZeRO-3 + CPU offload |

---

## 3. 环境配置

### 3.1 创建 Conda 环境

```bash
conda create -n robobrain python=3.11 -y
conda activate robobrain

# PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 核心依赖
pip install -r requirements.txt

# FFmpeg (用于 DROID 视频解码)
conda install -c conda-forge ffmpeg -y
```

### 3.2 requirements.txt 内容

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
einops>=0.7.0
scipy>=1.11.0
transformers>=4.45.0
accelerate>=0.25.0
peft>=0.10.0
deepspeed>=0.14.0
pillow>=10.0.0
pyyaml>=6.0
huggingface-hub>=0.20.0
```

### 3.3 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import transformers, peft, deepspeed; print('All imports OK')"
```

---

## 4. 模型下载

### 4.1 基础模型：RoboBrain2.5-8B-NV

```bash
# 方式一：HuggingFace CLI
huggingface-cli login  # 输入你的 HF token
huggingface-cli download BAAI/RoboBrain2.5-8B-NV --local-dir ./models/RoboBrain2.5-8B-NV

# 方式二：Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/RoboBrain2.5-8B-NV', local_dir='./models/RoboBrain2.5-8B-NV')
"
```

**模型信息：**
- 地址：https://huggingface.co/BAAI/RoboBrain2.5-8B-NV
- 大小：约 17GB (safetensors 格式)
- 架构：Qwen3-VL (8B参数)

下载后修改 `config/train_lora.yaml` 和 `config/train_full.yaml` 中的 `model.base_model` 路径。

---

## 5. 数据集下载

### 5.1 RLBench (仿真数据，含原生深度)

**HuggingFace 地址：** https://huggingface.co/datasets/hqfang/rlbench-18-tasks

```bash
mkdir -p data/rlbench

# 下载单个任务 (验证用)
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    'hqfang/rlbench-18-tasks',
    'val/close_jar.zip',
    repo_type='dataset',
    local_dir='data/rlbench'
)
print(f'Downloaded to: {path}')
"
cd data/rlbench && unzip val/close_jar.zip -d rlbench_sample/ && cd ../..

# 下载全部 18 个任务 (完整训练)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'hqfang/rlbench-18-tasks',
    repo_type='dataset',
    local_dir='data/rlbench'
)
"
# 逐个解压
cd data/rlbench
for f in train/*.zip val/*.zip; do
    unzip -q "$f" -d rlbench_full/
done
cd ../..
```

**数据格式：**
- 每个 episode：5 个摄像头 x (RGB + Depth + Mask) x N 帧
- 深度编码：RGB PNG，解码公式 `depth_m = 0.01 + 4.99 * (R + G*256 + B*65536) / (256^3 - 1)`
- 图像大小：128x128（加载时自动 resize 到 256x256）

### 5.2 DROID-100 (真实数据，RGB only，需生成伪深度)

**HuggingFace 地址：** https://huggingface.co/datasets/lerobot/droid_100

```bash
mkdir -p data/droid_sample

# 下载视频并提取帧
python -c "
from huggingface_hub import hf_hub_download
import subprocess, os

# 下载一个外部摄像头视频
video_path = hf_hub_download(
    'lerobot/droid_100',
    'videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4',
    repo_type='dataset'
)
print(f'Video: {video_path}')

# 提取帧 (每秒 1 帧，最多 50 帧)
os.makedirs('data/droid_sample', exist_ok=True)
subprocess.run([
    'ffmpeg', '-i', video_path,
    '-vf', 'fps=1', '-frames:v', '50',
    'data/droid_sample/frame_%04d.png', '-y'
], capture_output=True)
print(f'Extracted frames to data/droid_sample/')
"
```

**完整 DROID 数据集（大规模训练）：**
- 地址：https://droid-dataset.github.io/
- 76k 条示范轨迹，564 个场景
- 下载需要额外存储空间（约 1.5TB）

### 5.3 其他推荐数据集

| 数据集 | 类型 | 地址 | 用途 |
|--------|------|------|------|
| **Bridge V2** | 真实 RGB | https://huggingface.co/datasets/lerobot/bridge_v2 | 桌面操控 |
| **Open X-Embodiment** | 混合 | https://robotics-transformer-x.github.io/ | 多机器人泛化 |
| **RoboVQA** | 真实 RGB+问答 | https://huggingface.co/datasets/BAAI/RoboVQA | 视觉问答 |
| **Depth Anything V2** | 深度估计模型 | https://huggingface.co/depth-anything/Depth-Anything-V2-Large | 给 RGB 数据生成伪深度 |

### 5.4 为 RGB-only 数据生成伪深度 (Depth Anything V2)

```bash
pip install depth-anything-v2
```

```python
# generate_pseudo_depth.py
from depth_anything_v2.dpt import DepthAnythingV2
import torch, os, glob
from PIL import Image
import numpy as np

model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256,512,1024,1024])
model.load_state_dict(torch.load('path/to/depth_anything_v2_vitl.pth'))
model = model.cuda()

rgb_dir = 'data/droid_sample'
depth_dir = os.path.join(rgb_dir, 'depth')
os.makedirs(depth_dir, exist_ok=True)

for img_path in sorted(glob.glob(os.path.join(rgb_dir, '*.png'))):
    depth = model.infer_image(img_path)  # returns [H, W] float32
    out_path = os.path.join(depth_dir, os.path.basename(img_path).replace('.png', '.npy'))
    np.save(out_path, depth.astype(np.float32))
    print(f'  {img_path} -> {out_path}')
```

> 注：如果不安装 Depth Anything V2，代码会自动生成基于亮度的启发式伪深度（仅限验证，不建议正式训练使用）。

---

## 6. 训练运行

### 6.1 数据目录结构

确保数据按以下结构组织：

```
data/
  rlbench_sample/              # 或 rlbench_full/
    close_jar/
      all_variations/episodes/episode0/
        front_rgb/0.png, 1.png, ...
        front_depth/0.png, 1.png, ...
  droid_sample/
    frame_0001.png, frame_0002.png, ...
    depth/
      frame_0001.npy, frame_0002.npy, ...
```

修改 `config/train_lora.yaml` 或 `config/train_full.yaml` 中的数据路径：

```yaml
data:
  rlbench_root: "data/rlbench_sample"   # 你的 RLBench 数据路径
  droid_root: "data/droid_sample"        # 你的 DROID 数据路径
```

### 6.2 验证 Pipeline (Dry Run)

在正式训练前，务必先验证流水线：

```bash
# 验证 LoRA 模式
python train.py --config config/train_lora.yaml --dry_run

# 验证全量微调模式
python train.py --config config/train_full.yaml --dry_run
```

预期输出：
```
[LORA fine-tuning]
  Trainable: 36.8M / 8803.9M (0.42%)
  [Dry run] 1 step OK. lm_loss=3.xxxx
```

### 6.3 LoRA 微调 (推荐起步)

```bash
# 单卡训练
python train.py --config config/train_lora.yaml

# 多卡训练 (4 GPU + DeepSpeed ZeRO-2)
deepspeed --num_gpus=4 train.py \
    --config config/train_lora.yaml \
    --deepspeed config/deepspeed_zero2.json
```

**配置要点（`config/train_lora.yaml`）：**
```yaml
finetune_mode: "lora"
lora:
  r: 16              # LoRA 秩，越大能力越强但显存越大
  lora_alpha: 32      # 一般设为 2*r
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
training:
  learning_rate: 2.0e-4    # LoRA adapter 学习率
  lr_3d_branch: 1.0e-3     # 3D 分支学习率 (新参数，需要更大 LR)
  per_device_batch_size: 2
  gradient_accumulation_steps: 4
```

### 6.4 全量微调

```bash
# 全量微调必须用多卡 + ZeRO-3
deepspeed --num_gpus=4 train.py \
    --config config/train_full.yaml \
    --deepspeed config/deepspeed_zero3.json
```

**配置要点（`config/train_full.yaml`）：**
```yaml
finetune_mode: "full"
full_finetune:
  unfreeze_llm_layers: -1     # -1 = 全部解冻, 8 = 只解冻最后 8 层
  gradient_checkpointing: true # 省显存，代价约 30% 速度
training:
  learning_rate: 5.0e-5    # 更低的 LR (预训练权重)
  lr_3d_branch: 1.0e-3     # 3D 分支同样高 LR
  per_device_batch_size: 1  # 全量需要更小 batch
  gradient_accumulation_steps: 8
```

**部分层解冻（折中方案）：**
```yaml
full_finetune:
  unfreeze_llm_layers: 8      # 只解冻最后 8 层 (36 层中)
  # 效果：31.97% 参数可训练 (2,809.7M)，显存需求比全量少约 60%
```

### 6.5 CLI 参数一览

```bash
python train.py \
    --config config/train_lora.yaml \              # 配置文件路径
    --finetune_mode lora \                         # 覆盖配置中的微调模式 (lora/full)
    --deepspeed config/deepspeed_zero2.json \      # DeepSpeed 配置
    --resume outputs/lora/checkpoint-500 \         # 断点续训
    --dry_run                                      # 只跑 1 步验证
```

### 6.6 断点续训

```bash
# 从 checkpoint 恢复
python train.py \
    --config config/train_lora.yaml \
    --resume outputs/lora/checkpoint-500
```

Checkpoint 内容：
```
checkpoint-500/
  3d_branch.pt              # DepthToGaussian + GS Encoder + Projector 权重
  lora_adapter/             # (LoRA 模式) PEFT adapter 权重 (~60MB)
    adapter_config.json
    adapter_model.safetensors
  vlm_trainable.pt          # (全量模式) LLM 可训练参数
  training_state.pt         # optimizer + scheduler 状态 + step
```

---

## 7. 训练参数说明

### 7.1 损失函数

```
L_total = L_lm + 0.1 * L_render

L_lm:     标准自回归语言建模损失 (next token prediction)
L_render: 3DGS 渲染重建损失 = 0.8*L1(RGB) + 0.2*(1-SSIM) + 0.5*L1(Depth) + 0.01*Opacity
```

### 7.2 学习率策略

| 参数组 | LoRA 模式 | 全量模式 | 说明 |
|--------|-----------|---------|------|
| 3D 分支 | 1e-3 | 1e-3 | 新参数，需要大 LR 快速收敛 |
| LoRA adapter | 2e-4 | -- | 低秩适配器 |
| LLM 全量 | -- | 5e-5 | 预训练权重，需要小 LR 防止遗忘 |

调度器：Cosine Decay with Linear Warmup

### 7.3 显存估算 (8B 模型, bfloat16)

| 组件 | LoRA (ZeRO-2) | Full (ZeRO-3, 4卡) |
|------|--------------|---------------------|
| 模型权重 | ~16GB (分布在多卡) | ~4GB/卡 (ZeRO-3 分片) |
| 优化器 | ~0.3GB (只优化 LoRA+3D) | ~16GB/卡 (CPU offload) |
| 梯度 | ~0.3GB | ~4GB/卡 |
| 激活值 | ~2-4GB | ~2-4GB (grad ckpt) |
| **总计/卡** | **~20GB** | **~28GB** |

---

## 8. 项目文件说明

```
robobrain_3dgs/
  train.py                         # 主训练脚本 (LoRA + Full 统一接口)
  requirements.txt                 # Python 依赖
  TRAINING_GUIDE.md                # 本文档

  config/
    train_lora.yaml                # LoRA 微调配置
    train_full.yaml                # 全量微调配置
    train_config.yaml              # 默认配置
    deepspeed_zero2.json           # DeepSpeed ZeRO-2 (LoRA)
    deepspeed_zero3.json           # DeepSpeed ZeRO-3 (Full)

  models/
    robobrain_vlm.py               # 核心：Qwen3-VL + 3DGS 集成
    depth_to_gaussian.py           # RGBD -> 3D Gaussian 参数
    gs_encoder.py                  # PointNet++ 编码器
    gs_renderer.py                 # 可微渲染 + 辅助损失
    fusion.py                      # 2D/3D 双流融合
    robobrain_3dgs.py              # 独立验证用模型
    visual_encoder_2d.py           # 2D 视觉编码器 (占位)

  data/
    rlbench_loader.py              # RLBench 数据加载器
    droid_loader.py                # DROID 数据加载器
    synthetic.py                   # 合成数据生成

  utils/
    camera.py                      # 深度反投影 + 点云归一化

  validate_single.py               # 单样本 Pipeline 验证
  validate_vlm.py                  # Tiny VLM 集成验证
  validate_vlm_8b.py               # 8B 模型完整验证
```

---

## 9. 完整训练流程 (Quick Start)

```bash
# 1. 环境配置
conda create -n robobrain python=3.11 -y && conda activate robobrain
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y

# 2. 下载基础模型
huggingface-cli download BAAI/RoboBrain2.5-8B-NV --local-dir ./models/RoboBrain2.5-8B-NV

# 3. 下载数据集 (最小验证集)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('hqfang/rlbench-18-tasks', 'val/close_jar.zip', repo_type='dataset', local_dir='data/rlbench')
"
cd data/rlbench && unzip val/close_jar.zip -d ../rlbench_sample/ && cd ../..

# 4. 修改配置中的路径
#    编辑 config/train_lora.yaml，设置正确的 model.base_model 和 data 路径

# 5. 验证 Pipeline
python train.py --config config/train_lora.yaml --dry_run

# 6. 开始训练
# 单卡 LoRA
python train.py --config config/train_lora.yaml

# 多卡 LoRA
deepspeed --num_gpus=4 train.py --config config/train_lora.yaml --deepspeed config/deepspeed_zero2.json

# 多卡全量
deepspeed --num_gpus=4 train.py --config config/train_full.yaml --deepspeed config/deepspeed_zero3.json
```
