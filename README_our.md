# README_our

本文件为项目的“整套操作流程”说明，基于仓库现有脚本与说明文档整理，覆盖：环境准备 -> 训练 Step-Aware Preference Model（可选） -> SPO 训练 -> 推理。

## 项目结构速览
- `spo_training_and_inference/`: SPO 训练与推理主代码。
- `step_aware_preference_model/`: Step-Aware Preference Model 训练代码。
- `assets/`: 论文示例图。

## 整体流程概览
1. 环境准备（推荐 Docker，也可用 conda）。
2. 可选：训练 Step-Aware Preference Model（如果不训练，可直接下载官方权重）。
3. SPO 训练（需要 Step-Aware Preference Model 权重）。
4. 推理（使用 Hugging Face 上的 checkpoint 或本地训练的模型）。

下面按步骤给出具体操作。

---

## 1. 环境准备

### 方案 A：Docker（推荐）
在宿主机具备 NVIDIA 驱动与 GPU 的前提下：
```bash
sudo docker pull rockeycoss/spo:v1
sudo docker run --gpus all -it --ipc=host rockeycoss/spo:v1 /bin/bash
```

### 方案 B：Conda（仅 SPO 目录提供）
```bash
cd spo_training_and_inference
conda env create -f environment.yaml --name spo
conda activate spo
```

### 统一准备
克隆仓库并进入目标目录：
```bash
git clone https://github.com/RockeyCoss/SPO
cd ./SPO
```

登录 wandb（训练需要）：
```bash
wandb login {Your wandb key}
```

可选：设置 Hugging Face 缓存目录：
```bash
export HUGGING_FACE_CACHE_DIR=/path/to/your/cache/dir
```

---

## 2. 训练 Step-Aware Preference Model（可选）
> 如果只想复现 SPO 训练或推理，可直接下载官方提供的 step-aware 权重，跳到第 3 步。

进入目录并安装依赖：
```bash
cd step_aware_preference_model
pip uninstall peft -y
pip install -r requirements.txt
```

下载 Pick-a-Pic 数据集：
```python
from datasets import load_dataset
dataset = load_dataset("yuvalkirstain/pickapic_v1", num_proc=64)
```
更改本地数据集路径：
改step_aware_preference_model/datasetss/pick_a_pic_spm_dataset.py
的
from_disk: bool = True
dataset_name改成本地路径即可

启动训练（默认 4 张 A100 80GB）：
```bash
bash run_commands/train_spm_sd15.sh
# 或
bash run_commands/train_spm_sdxl.sh
```




训练完成后的权重：
- SD v1.5：`work_dirs/sdv15_spm/final_ckpt.bin`
- SDXL：`work_dirs/sdxl_spm/final_ckpt.bin`

这些权重用于后续 SPO 训练。

---

## 3. SPO 训练
进入目录：
```bash
cd spo_training_and_inference
```

### 3.1 下载官方 Step-Aware 权重（若不自行训练）
```bash
sudo apt update
sudo apt install wget

mkdir model_ckpts
cd model_ckpts

wget https://huggingface.co/SPO-Diffusion-Models/Step-Aware_Preference_Models/resolve/main/sd-v1-5_step-aware_preference_model.bin
wget https://huggingface.co/SPO-Diffusion-Models/Step-Aware_Preference_Models/resolve/main/sdxl_step-aware_preference_model.bin

cd ..
```

### 3.2 配置训练参数
默认配置文件位于：
- SD v1.5：`configs/spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10.py`
- SDXL：`configs/spo_sdxl_4k-prompts_num-sam-2_3-is_10ep_bs2_gradacc2.py`

若使用自训权重，请将上述配置中的 `config.preference_model_func_cfg.ckpt_path` 改成你的 `final_ckpt.bin` 路径。

### 3.3 训练命令
```bash
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_spo.py \
  --config configs/spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10.py

# SDXL
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_spo_sdxl.py \
  --config configs/spo_sdxl_4k-prompts_num-sam-2_3-is_10ep_bs2_gradacc2.py
```

---

## 4. 推理（Inference）
在 `spo_training_and_inference/` 目录下执行：

SD v1.5：
```bash
python inference_scripts/inference_spo_sd-v1-5.py \
  --ckpt_id SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep \
  --prompt "an image of a beautiful lake"
```

SDXL：
```bash
python inference_scripts/inference_spo_sdxl.py \
  --ckpt_id SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep \
  --prompt "a child and a penguin sitting in front of the moon"
```

说明：
- 默认使用 HF 的 checkpoint（`--ckpt_id` 可改为本地路径或 HF repo）。
- SDXL 推理脚本会额外加载 `madebyollin/sdxl-vae-fp16-fix` VAE。
- `--cfg_scale`、`--seed`、`--output_filename` 均可自定义。

---

## 5. 常用检查点
Hugging Face 上可用的官方权重：
- `SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep`
- `SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA`
- `SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep`
- `SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA`

---

## 6. 最小可跑路径（仅推理）
1. 进入 `spo_training_and_inference/`。
2. 保证环境已包含 diffusers/torch 等依赖（Docker 推荐）。
3. 直接运行 inference 脚本（第 4 步）。

如需完整训练，依次执行第 2 -> 3 步。
