# Third-Party Code and Licenses

This project incorporates code from the following third-party sources.
Users of this project must comply with the license terms of each component.

## Vendored / Copied Code

### CroCo (Cross-view Completion)

- **Location:** `models/croco/`
- **Source:** https://github.com/naver/croco
- **License:** CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)
- **Copyright:** Copyright (C) 2022-present Naver Corporation
- **Note:** **Non-commercial use only.** This component and any derivatives are restricted to non-commercial purposes.
- **Papers:**
  - CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-view Completion (NeurIPS 2022)
  - CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow (ICCV 2023)

### DUSt3R

- **Location:** `models/dust3r/`
- **Source:** https://github.com/naver/dust3r
- **License:** CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)
- **Copyright:** Copyright (C) 2024-present Naver Corporation
- **Note:** **Non-commercial use only.** This component and any derivatives are restricted to non-commercial purposes.

### SSD-6D (Rendering Utilities)

- **Location:** `rendering/model.py`, `rendering/renderer.py`, `rendering/renderer_xyz.py`, `rendering/utils.py`
- **Source:** https://github.com/wadimkehl/ssd-6d
- **License:** MIT License
- **Note:** Code copied and modified for 3D model rendering and depth map generation.

### iBOT (Image BERT Pre-Training with Online Tokenizer)

- **Location:** `models/ibot_transformers.py`
- **Source:** https://github.com/bytedance/ibot
- **License:** Apache License 2.0
- **Copyright:** Copyright ByteDance, Inc. and its affiliates
- **Note:** Vision Transformer implementation adapted from iBOT.

### DeiT (Data-efficient Image Transformers)

- **Location:** `models/deit_utils.py`
- **Source:** https://github.com/facebookresearch/deit
- **License:** Apache License 2.0
- **Copyright:** Copyright (c) Meta Platforms, Inc. and affiliates
- **Note:** Utility functions from DeiT and PyTorch Image Models (timm).

### MAE (Masked Autoencoders)

- **Referenced in:** `models/croco/models/pos_embed.py`
- **Source:** https://github.com/facebookresearch/mae
- **License:** CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)
- **Copyright:** Copyright (c) Meta Platforms, Inc. and affiliates
- **Note:** Positional embedding utilities.

### UniMatch

- **Referenced in:** `models/croco/stereoflow/augmentor.py`, `models/croco/stereoflow/datasets_flow.py`
- **Source:** https://github.com/autonomousvision/unimatch
- **License:** MIT License
- **Note:** Data augmentation and optical flow dataset utilities.

## Runtime Dependencies (not vendored)

The following packages are required at runtime and installed separately.
See README.md for installation instructions.

| Package | License | Usage |
|---------|---------|-------|
| PyTorch | BSD-3-Clause | Core ML framework |
| torchvision | BSD-3-Clause | Vision utilities |
| timm (PyTorch Image Models) | Apache 2.0 | ViT architectures |
| Hugging Face Transformers | Apache 2.0 | DeiT, iBOT model loading |
| OpenCV (cv2) | Apache 2.0 | Image processing |
| FAISS | MIT | GPU-accelerated similarity search |
| Segment Anything (SAM) | Apache 2.0 | Segmentation masks |
| Diffusers | Apache 2.0 | Stable Diffusion features |
| scikit-learn | BSD-3-Clause | KMeans clustering |
| trimesh | MIT | 3D mesh handling |
| kornia | Apache 2.0 | Differentiable image processing |
| poselib | BSD-3-Clause | Pose estimation utilities |

## License Compatibility Notice

The main project code is licensed under Apache License 2.0.

**Important:** The `models/croco/` and `models/dust3r/` directories contain code
licensed under CC BY-NC-SA 4.0, which restricts use to **non-commercial purposes only**.
If you use these components, the CC BY-NC-SA 4.0 terms apply to your use of those
components and any derivatives thereof. The remaining project code under Apache 2.0
is not affected by this restriction when used independently of the CC BY-NC-SA 4.0
components.
