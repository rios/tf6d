Note that this repo only deals with 6D pose estimation, you need segmentation masks as input. These can be obtained with supervised trained methods or zero-shot methods. For zero-shot we refer to [cnos](https://github.com/nv-nguyen/cnos).

## Installation:
To setup the environment to run the code locally follow these steps:

```
conda create --name pos3r python=3.9
conda activate pos3r
conda install -c conda-forge cudatoolkit-dev
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Pip install packages
pip install tqdm==4.65.0
pip install timm==0.9.16
pip install matplotlib==3.8.3
pip install scikit-learn==1.4.1.post1
pip install opencv-python==4.10.0.82
pip install git+https://github.com/lucasb-eyer/pydensecrf.git@dd070546eda51e21ab772ee6f14807c7f5b1548b
pip install transforms3d==0.4.1
pip install pillow==9.4.0
pip install plyfile==1.0.3
pip install trimesh==4.1.4
pip install imageio==2.34.0
pip install pypng==0.20220715.0
pip install vispy==0.12.2
pip install pyopengl==3.1.1a1
pip install pyglet==2.0.10
pip install numba==0.59.0
pip install jupyter==1.0.0
pip install future==0.18.0
pip install hydra-core>=1.1
pip install loguru>=0.3
pip install numpy>=1.17
pip install omegaconf>=2.1
pip install scikit-learn>=1.0  # Keeping the latest scikit-learn
pip install jupyterlab==3.5.1
pip install albumentations==1.3.1
pip install open_clip_torch==2.23.0
pip install einops==0.7.0
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install transformers==4.29.2
pip install diffusers==0.16.1
pip install accelerate==0.20.1
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==1.4.2
pip install torchmetrics==0.6.0
pip install kornia==0.6
pip install mat73==0.62
pip install pre-commit
pip install hydra-submitit-launcher
pip install pandas
pip install seaborn
pip install loguru
pip install protobuf==3.20.3    # weird dependency with datasets and google's api
pip install poselib
# Conda install for faiss-gpu
conda install -c conda-forge nb_conda_kernels=2.3.1
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```