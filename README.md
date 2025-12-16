# BEVFusion Fine-Tuning on nuScenes (CMPE 249 Project)

This repository contains our course project code for fine-tuning BEVFusion on the nuScenes dataset (mini or full) under different training configurations:

- Baseline pre-trained BEVFusion evaluation

- Full-model fine-tuning

- Data Augmentation 

- Fusion-only fine-tuning (frozen encoders)

- Two-stage LiDAR-first -> Camera+LiDAR fusion

The code is based on the MIT Han Lab BEVFusion implementation and adds a few light modifications and scripts to make it easier to run experiments on an HPC cluster or a single workstation.

## Repository Layout

Key directories and files:

- configs/ - model & dataset configs (from BEVFusion, with our minor tweaks)

- mmdet3d/ - BEVFusion’s fork of MMDetection3D (includes spconv ops)

- tools/

  - train.py - main training entry point

  - test.py - evaluation script

  - create_data.py - nuScenes preprocessing / info file generation

- data/ - expected location for nuScenes data (not tracked by git)

- pretrained/ - pre-trained model weights (Swin, BEVFusion, etc.)

- runs/ - model checkpoints & internal run directories (created at runtime)

- setup.py - builds mmdet3d and custom CUDA / spconv ops

- environment.yaml - conda environment spec

## Requirements

### Software

- Python 3.8
- CUDA 11.x (we used 11.3) and a compatible NVIDIA driver
- Conda (Anaconda / Miniconda)
- Git

### Python / PyTorch Stack

The exact versions you use may differ, but our working setup used:

- pytorch==1.10.2 (CUDA 11.3 build)

- torchvision==0.11.3

- mmcv==1.4.0

- mmdet and mmdet3d provided via this repo & python setup.py develop

- numpy==1.23.x

- torchpack, numba, pyyaml, yapf, etc.

## Environment Setup

### Clone the Repo

```bash
git clone <this-repo-url>
cd bevfusion-finetune
```

### Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate bevfusion
```

Or if you want to install packages yourself:

```bash
conda create -n bevfusion python=3.8 -y
conda activate bevfusion
```

Install PyTorch (example for CUDA 11.3):

```bash
conda install pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.3 \
  -c pytorch -c nvidia -y
```

Then install basic Python dependencies (you can extend this list):

```bash
pip install torchpack numba pyyaml yapf numpy==1.23.5
# mmcv etc. can be pulled in via setup.py
```

To avoid interference from ~/.local packages, you can set:

```bash
export PYTHONNOUSERSITE=1
```

### Build mmdet3d and Custom Ops

From the repo root:

```bash
# Make sure no conflicting mmdet3d is installed
pip uninstall -y mmdet3d || true

# Make sure now previously built files exist
python setup.py clean
rm -rf build mmdet3d.egg-info

python setup.py develop
```

This will compile and install the forked mmdet3d (including spconv CUDA ops) into the bevfusion environment.

## Dataset Setup

### Download nuScenes

Follow the official nuScenes instructions to download the dataset (mini or full) and place it under `data/`:

```bash
<REPO_ROOT>/
  data/
    nuscenes/
      samples/
      sweeps/
      maps/
      v1.0-mini/ or v1.0-trainval/
``` 
### Create nuScenes Info Files

From the repo root:

```bash
conda activate bevfusion

python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini   # or v1.0-trainval for full
```

This generates the `nuscenes_infos_*.pkl` files that the dataset class uses.

## Pretrained Models

### Download Pretrained Weights

BEVFusion uses:

- Swin-T backbone pretrained on nuImages (for camera branch)

- BEVFusion detection checkpoint on nuScenes (camera+LiDAR)

Typical filenames (place them under pretrained/):

- pretrained/swint-nuimages-pretrained.pth

- pretrained/bevfusion-det.pth

You can obtain them by running:

```bash
wget -O bevfusion-det.pth https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v 
wget -O swint-nuimages-pretrained.pth https://www.dropbox.com/scl/fi/f3e67wgn2omoftah4ceri/swint-nuimages-pretrained.pth?rlkey=k9kafympye80b3b1quutti4yq
```

## Running Experiments

### General Command Structure

```bash
# Base configs
BASE_CONFIG=configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
LIDAR_CONFIG=configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml

# Logging root directory
LOGROOT=./logs
mkdir -p "$LOGROOT"

export PYTHONNOUSERSITE=1
```

For logging every run, we use this consistent pattern:

```bash
EXPNAME=<experiment_name>
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOGDIR=$LOGROOT/${EXPNAME}_${RUN_ID}
mkdir -p "$LOGDIR"

# Example: training
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py ... \
  2>&1 | tee "$LOGDIR/train.out"

# Example: evaluation
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py ... \
  2>&1 | tee "$LOGDIR/test.out"
```

All console output goes to train.out / test.out for later analysis.

### Baseline Evaluation

Evaluate the official BEVFusion detector on nuScenes:

```bash
EXPNAME=baseline_eval
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOGDIR=$LOGROOT/${EXPNAME}_${RUN_ID}
mkdir -p "$LOGDIR"

CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
  $BASE_CONFIG \
  pretrained/bevfusion-det.pth \
  --eval bbox \
  --launcher none \
  2>&1 | tee "$LOGDIR/test.out"
```

--launcher none ensures single-GPU eval without distributed initialization.
At the end of test.out, you’ll see metrics like mAP and NDS on nuScenes.

### Full-Model Fine-Tuning

Fine-tune the full BEVFusion model (camera + LiDAR encoders + fusion + head):

```bash
EXPNAME=finetune_full
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOGDIR=$LOGROOT/${EXPNAME}_${RUN_ID}
mkdir -p "$LOGDIR"

SAMPLES_PER_GPU=2    # adjust based on GPU memory
WORKERS_PER_GPU=4

CUDA_VISIBLE_DEVICES=0 \
python tools/train.py $BASE_CONFIG \
  --data.samples_per_gpu=$SAMPLES_PER_GPU \
  --data.workers_per_gpu=$WORKERS_PER_GPU \
  --optimizer.lr=1e-4 \
  --runner.max_epochs=2 \
  --model.encoders.camera.backbone.init_cfg.checkpoint=pretrained/swint-nuimages-pretrained.pth \
  --load_from=pretrained/bevfusion-det.pth \
  2>&1 | tee "$LOGDIR/train.out"
```

- --runner.max_epochs=2 is a short schedule for experimentation. You can increase this if you have more compute time.
- --load_from initializes from the official BEVFusion checkpoint.

To evaluate the fine-tuned checkpoint (e.g., epoch_2.pth):

```bash
EXPNAME=finetune_full_eval
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOGDIR=$LOGROOT/${EXPNAME}_${RUN_ID}
mkdir -p "$LOGDIR"

CKPT=<path_to_epoch_2_or_latest.pth>

CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
  $BASE_CONFIG \
  $CKPT \
  --eval bbox \
  --launcher none \
  2>&1 | tee "$LOGDIR/test.out"
```
### Fusion-Only Fine-Tuning (Frozen Encoders)

Here we freeze camera & LiDAR encoders and only train BEV fusion layers + detection head:

```bash
EXPNAME=finetune_fusion_only
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOGDIR=$LOGROOT/${EXPNAME}_${RUN_ID}
mkdir -p "$LOGDIR"

SAMPLES_PER_GPU=2
WORKERS_PER_GPU=4

CUDA_VISIBLE_DEVICES=0 \
python tools/train.py $BASE_CONFIG \
  --data.samples_per_gpu=$SAMPLES_PER_GPU \
  --data.workers_per_gpu=$WORKERS_PER_GPU \
  --freeze_encoders=True \
  --optimizer.lr=1e-4 \
  --runner.max_epochs=2 \
  --model.encoders.camera.backbone.init_cfg.checkpoint=pretrained/swint-nuimages-pretrained.pth \
  --load_from=pretrained/bevfusion-det.pth \
  2>&1 | tee "$LOGDIR/train.out"
```

Evaluate the resulting checkpoint with the same tools/test.py pattern as above.

### Two-Stage LiDAR-First → Fusion

Stage 1: LiDAR-Only Training

```bash
EXPNAME=twostage_lidar_only
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOGDIR=$LOGROOT/${EXPNAME}_${RUN_ID}
mkdir -p "$LOGDIR"

LIDAR_SAMPLES_PER_GPU=4
WORKERS_PER_GPU=4

CUDA_VISIBLE_DEVICES=0 \
python tools/train.py $LIDAR_CONFIG \
  --data.samples_per_gpu=$LIDAR_SAMPLES_PER_GPU \
  --data.workers_per_gpu=$WORKERS_PER_GPU \
  --optimizer.lr=1e-4 \
  --runner.max_epochs=2 \
  2>&1 | tee "$LOGDIR/train.out"
```

After this finishes, find the checkpoint (e.g. runs/run-XXXX/epoch_2.pth) and set:

```bash
LIDAR_CKPT=<path_to_lidar_epoch_2.pth>
```

Stage 2: Camera+LiDAR Fusion from LiDAR Checkpoint

```bash
EXPNAME=twostage_fusion_from_lidar
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOGDIR=$LOGROOT/${EXPNAME}_${RUN_ID}
mkdir -p "$LOGDIR"

SAMPLES_PER_GPU=2
WORKERS_PER_GPU=4

CUDA_VISIBLE_DEVICES=0 \
python tools/train.py $BASE_CONFIG \
  --data.samples_per_gpu=$SAMPLES_PER_GPU \
  --data.workers_per_gpu=$WORKERS_PER_GPU \
  --optimizer.lr=1e-4 \
  --runner.max_epochs=2 \
  --model.encoders.camera.backbone.init_cfg.checkpoint=pretrained/swint-nuimages-pretrained.pth \
  --load_from=$LIDAR_CKPT \
  2>&1 | tee "$LOGDIR/train.out"
```

Then evaluate that checkpoint with tools/test.py as usual.

## Reproducibility Notes

To reproduce our experiments:

1. Set up environment and build with python setup.py develop.

2. Download nuScenes and generate info files via tools/create_data.py.

3. Download pretrained weights into pretrained/.

4. Run the desired training commands in (short schedules: 2 epochs).

5. Run evaluation on the official BEVFusion checkpoint (baseline), and each fine-tuned checkpoint (full, fusion-only, two-stage).

6. Inspect logs/<experiment>_<timestamp>/train.out and test.out for:

   - Loss values and GPU memory (memory: field).

   - Final evaluation metrics (bbox / nuScenes metrics) for comparison.

## Acknowledgements

This project is based on the [BEVFusion](https://github.com/mit-han-lab/bevfusion) implementation by the MIT Han Lab.




