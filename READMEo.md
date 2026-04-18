# HS-PAC
## 准备
conf改好
运行：
```bash
cd sh
bash train.sh
```
多卡运行：
```bash
cd sh
bash trainDDP.sh
```
⚠️ 注意：目前DDP版本无法使用CUDAGraph加速，由于cuda在反向传播过程中的信息通信，无法正确使用CUDAGraph。如果需要使用CUDAGraph加速
目前会自动取消CUDAGraph加速。后续如果有需要可以考虑剔除DDP版本中的反向传播通信，TODO: 


## 数据集
dataset_freihand:mobrecon搬运的freihand加载器

## 文件
train.py是无加速版本的训练脚本
trainer.py 重构过的trainer，增加了可扩展性
src/utils/data_keys.py 以后加载和取用数据的键统一在这里面管理，避免混乱

## 新增
支持进度条适应窗口变化，进度条不会刷屏了

## 环境配置
先安装pytorch CUDAGraph加速要求pytorch>=2.0
```bash
conda create -n yourname python=3.8
conda activate yourname
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

参考版本
torch                     2.4.1
torchvideotransforms      0.1.2 
torchvision               0.19.1
```bash
pip install git+https://github.com/hassony2/torch_videovision
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```

```bash
pip install attrs brotlipy certifi chumpy cycler fonttools fvcore h5py imageio iniconfig iopath Jinja2 joblib kiwisolver MarkupSafe matplotlib mkl-fft mkl-service networkx numpy opencv-python openmesh packaging pandas Pillow pluggy portalocker protobuf pycocotools PyOpenGL pyparsing pytest python-dateutil pytz PyWavelets PyYAML pyzmq scikit-image scikit-learn scipy tabulate tensorboardX termcolor threadpoolctl tifffile tomli tqdm transforms3d trimesh vctoolkit vctools yacs open3d
pip install torch-cluster torch-geometric torch-scatter torch-sparse torch-spline-conv #这几个安的贼慢，因为需要编译
#如果失败了，就按照下面的来，指定torch的版本
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

#ln -s data/model/MANO_RIGHT.pkl template/MANO_RIGHT.pkl
```

Install [MPI-IS Mesh](https://github.com/MPI-IS/mesh) from the source
具体怎么安让AI生成针对linux的指令，官方仓库里的配置步骤有点迷，可以先试试下面的命令是否成功
```bash
conda install -c conda-forge boost pyopengl
git clone https://github.com/MPI-IS/mesh.git #别忘记切换文件夹
cd mesh
```

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev
```
以下命令需要一次性复制进来执行
```bash
export BOOST_INCLUDE_DIRS=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
pip install --verbose --no-deps --no-cache-dir .
```
验证安装成功
```bash
python -c "import psbody.mesh; print('Installation successful!')"
```

## 建议新建一个专门的环境查看tb文件
```bash
conda create -n tb python=3.8
conda activate tb  # 将 'tb_viewer' 替换为你实际的环境名称
pip install tensorboard
tensorboard --version
CUDA_VISIBLE_DEVICES="" tensorboard --logdir=/
```

环境配置已经过验证


新增，HO3D or inference如需配置，参照：
https://github.com/lixiny/manotorch
pip install git+https://github.com/mattloper/chumpy
pip install pyrender