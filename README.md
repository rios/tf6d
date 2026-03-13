# TF6D — Template-based 6D Object Pose Estimation

TF6D estimates the 6D pose (3D rotation + 3D translation) of objects in RGB images using Vision Transformer (ViT) feature matching. Given a set of pre-rendered object templates and a segmentation mask, the pipeline matches the test image crop against templates using DINOv2 descriptors, establishes 2D-3D correspondences, and solves for the pose via PnP.

This project evaluates on the [BOP Benchmark](https://bop.felk.cvut.cz/) datasets: LMO, YCB-Video, T-LESS, iToDD, HomebrewedDB, IC-BIN, and TU Dresden Light.

> **Note:** This repo only handles 6D pose estimation. You need segmentation masks as input, which can be obtained from supervised methods or zero-shot approaches like [CNOS](https://github.com/nv-nguyen/cnos).

## Pipeline Overview

```
Input Image + Mask
        |
        v
  Crop & Preprocess
        |
        v
  Extract ViT Descriptors (DINOv2)
        |                          Pre-computed Template Descriptors
        v                                    |
  Template Matching (Cosine Similarity) <----+
        |
        v
  Find 2D-3D Correspondences (Feature Matching + KMeans)
        |
        v
  Solve PnP --> 6D Pose (R, t)
```

## Project Structure

```
tf6d/
├── src/                    # Core pose estimation modules
│   ├── pose_extractor.py   # PoseViTExtractor: ViT-based feature extraction & matching
│   ├── extractor.py        # Base ViT feature extractor
│   ├── correspondences.py  # Feature matching & correspondence finding (FAISS, KMeans)
│   └── bop_pose_error.py   # BOP evaluation metrics (VSD, MSSD, MSPD, ADD-S)
├── models/                 # Vision model integrations
│   ├── dino.py, ibot.py    # Self-supervised ViT backbones
│   ├── clip.py, siglip.py  # CLIP-family models
│   ├── croco/              # CroCo stereo model (CC BY-NC-SA 4.0, see THIRD_PARTY.md)
│   └── dust3r/             # DUSt3R 3D reconstruction (CC BY-NC-SA 4.0, see THIRD_PARTY.md)
├── pose_utils/             # Utility functions
│   ├── utils.py            # Template matching, PnP solving, pose refinement
│   ├── img_utils.py        # Image preprocessing, masking, cropping
│   └── vis_utils.py        # Visualization helpers
├── rendering/              # 3D rendering for template generation
├── database/               # Configs and ground truth data
│   ├── zs6d_configs/       # Dataset-specific configuration files
│   └── gts/                # Ground truth pose annotations
├── zs6d.py                 # Main ZS6D class: high-level pose estimation API
├── evaluate_bop.py         # BOP benchmark evaluation script
├── prepare_templates_and_gt.py  # Template and ground truth preparation
└── extract_gt_poses*.py    # Ground truth extraction utilities
```

## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA 11.8+ support
- conda (recommended for PyTorch and FAISS installation)

### Setup

```bash
# Create conda environment
conda create --name tf6d python=3.9
conda activate tf6d

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cudatoolkit-dev

# Install Python dependencies
pip install -r requirements.txt

# Install FAISS (GPU version, via conda)
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Build CroCo CUDA extension (optional, for RoPE2D positional embeddings)
cd models/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../..
```

### Data Setup

Download BOP datasets from [bop.felk.cvut.cz](https://bop.felk.cvut.cz/datasets/) and place them under `./data/bop/`:

```
data/bop/
├── lmo/
├── ycbv/
├── tless/
├── itodd/
├── hb/
├── icbin/
└── tudl/
```

### External Dependencies (Optional)

Some evaluation scripts require additional repositories:

- **MASt3R** — For stereo-based correspondence finding (`eval_mast3r*.py` scripts).
  Clone [MASt3R](https://github.com/naver/mast3r) alongside this repo or set `PYTHONPATH`.
- **Zero123** — For novel view synthesis features.
  Set the `ZERO123_PATH` environment variable to point to your Zero123 installation.

## Usage

### 1. Prepare Templates

Generate template descriptors and ground truth from 3D models:

```bash
python prepare_templates_and_gt.py \
    --config_file ./database/zs6d_configs/template_gt_preparation_configs/cfg_template_gt_generation_lmo.json
```

### 2. Run Pose Estimation

```python
from zs6d import ZS6D

# Initialize with template descriptors
estimator = ZS6D(
    templates_gt_path="./database/gts/template_gts/lmo_template_panda48_gt_dinov2.json",
    norm_factors_path="./database/templates/lmo/models_panda48/norm_factor.json",
    model_type="dino_vits8",
    stride=4
)

# Estimate pose for a detected object
R_est, t_est = estimator.get_pose(
    img=image,           # PIL Image
    obj_id="1",          # Object ID string
    mask=mask,           # Binary segmentation mask (numpy array)
    cam_K=camera_matrix  # 3x3 intrinsic matrix (numpy array)
)
```

### 3. Evaluate on BOP Benchmark

```bash
python evaluate_bop.py \
    --config_file ./database/zs6d_configs/bop_eval_configs/cfg_lmo_inference_bop_zero.json
```

## Configuration

All dataset-specific settings are in JSON config files under `database/zs6d_configs/`.

**Template generation configs** (`template_gt_preparation_configs/`):
- Paths to 3D models, template poses, and output directories
- Camera intrinsics and template resolution

**Evaluation configs** (`bop_eval_configs/`):
- Dataset path, ground truth file, template descriptors
- Number of matched templates, image resolution

Update the `dataset_path` field in these configs to point to your local BOP data directory.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

**Important:** The `models/croco/` and `models/dust3r/` directories contain code from Naver Corporation licensed under **CC BY-NC-SA 4.0** (non-commercial use only). See [THIRD_PARTY.md](THIRD_PARTY.md) for full attribution and license details for all third-party components.
