# HS-PAC
## Preparation
Configure the conf file properly

## Dataset
dataset_freihand: FreiHand loader ported from mobrecon

## Files
train.py is the training script without acceleration
trainer.py is a refactored trainer with improved extensibility
src/utils/data_keys.py manages all data loading and access keys in one place to avoid confusion

## New Features
Progress bar now adapts to window size changes, preventing screen flooding

## Environment Setup
First install PyTorch. CUDAGraph acceleration requires pytorch>=2.0
```bash
conda create -n yourname python=3.8
conda activate yourname
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

Reference versions:
torch                     2.4.1
torchvideotransforms      0.1.2 
torchvision               0.19.1
```bash
pip install git+https://github.com/hassony2/torch_videovision
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```

```bash
pip install attrs brotlipy certifi chumpy cycler fonttools fvcore h5py imageio iniconfig iopath Jinja2 joblib kiwisolver MarkupSafe matplotlib mkl-fft mkl-service networkx numpy opencv-python openmesh packaging pandas Pillow pluggy portalocker protobuf pycocotools PyOpenGL pyparsing pytest python-dateutil pytz PyWavelets PyYAML pyzmq scikit-image scikit-learn scipy tabulate tensorboardX termcolor threadpoolctl tifffile tomli tqdm transforms3d trimesh vctoolkit vctools yacs open3d
pip install torch-cluster torch-geometric torch-scatter torch-sparse torch-spline-conv # These install very slowly as they need compilation
# If installation fails, try the following with specified torch version
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

#ln -s data/model/MANO_RIGHT.pkl template/MANO_RIGHT.pkl
```

Install [MPI-IS Mesh](https://github.com/MPI-IS/mesh) from source
Have AI generate Linux-specific commands for installation, as the configuration steps in the official repository are somewhat confusing. You can first try the commands below to see if they work successfully
```bash
conda install -c conda-forge boost pyopengl
git clone https://github.com/MPI-IS/mesh.git # Don't forget to change directory
cd mesh
```

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev
```
The following commands need to be copied and executed at once
```bash
export BOOST_INCLUDE_DIRS=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
pip install --verbose --no-deps --no-cache-dir .
```
Verify successful installation
```bash
python -c "import psbody.mesh; print('Installation successful!')"
```

## Recommended: Create a dedicated environment for viewing tensorboard files
```bash
conda create -n tb python=3.8
conda activate tb  # Replace 'tb' with your actual environment name
pip install tensorboard
tensorboard --version
CUDA_VISIBLE_DEVICES="" tensorboard --logdir=/
```

Environment configuration has been verified


# For additional setup, refer to:
https://github.com/lixiny/manotorch
pip install git+https://github.com/mattloper/chumpy
pip install pyrender