# 计图点云库

## 已经实现的模型

| Model       | Classification | Segmentation |
| ----------- | -------------- | ------------ |
| PointNet    | √              | √            |
| PointNet ++ | √              | √            |
| PointCNN    | √              | √            |
| DGCNN       | √              | √            |
| PointConv   | √              | √            |
| KPConv      | √              |              |

## 使用方法 

### 安装依赖

```bash
sudo apt install python3.7-dev libomp-dev
sudo python3.7 -m pip install git+https://github.com/Jittor/jittor.git
python3.7 -m pip install sklearn lmdb msgpack_numpy
```
### 安装点云库

```bash
git clone https://github.com/Jittor/PointCloudLib.git # 将库下载的本地
# 您需要将 ModelNet40 和 ShapeNet 数据集下载到 data_util/data/ 里面
ModelNet40 数据集链接 ： https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip 
ShapeNet 数据集链接 ： https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip 

sh run_cls.sh # 点云分类的训练和测试（以PointNet为例） 
sh run_seg.sh # 点云分割的训练和测试（以PointNet为例）

# 对于kpconv，需要额外执行脚本再开始训练
cd cpp_wrappers
bash compile_wrappers.sh
cd ..
python train_cls.py --model kpconv # kpconv训练
python train_cls.py --model kpconv --eval # 修改train_cls.py中的chkp_path指定模型进行测试
```

## 所依赖的库 

```bash
Python 3.7
Jittor 
Numpy
sklearn
lmdb
msgpack_numpy
...
```



## 实验结果

### 分类训练效果测试

| Model       | Input             | overall accuracy |
| ----------- | ----------------- | ---------------- |
| PointNet    | 1024 xyz          | 87.2             |
| PointNet ++ | 4096 xyz + normal | 92.3             |
| PointCNN    | 1024 xyz          | 92.6             |
| DGCNN       | 1024 xyz          | 92.9             |
| PointConv   | 1024 xyz + normal | 92.4             |
| KPConv      | xyz + neighbors + pools + lengths + features | 92.5             |

### 分类训练时间测试

| Model       | Speed up ratio (Compare with Pytorch) |
| :---------- | ------------------------------------- |
| PointNet    | 1.22                                  |
| PointNet ++ | 2.72                                  |
| PointCNN    | 2.41                                  |
| DGCNN       | 1.22                                  |
| PointConv   |                                       |
| KPConv      |                                       |

### 分割训练效果测试

| Model       | Input                         | pIoU |
| ----------- | ----------------------------- | ---- |
| PointNet    | 2048 xyz + cls label          | 83.5 |
| PointNet ++ | 2048 xyz + cls label + normal | 85.0 |
| PointCNN    | 2048 xyz + normal             | 86.0 |
| DGCNN       | 2048 xyz + cls label          | 85.1 |
| PointConv   | 2048 xyz                      | 85.4 |

### 分割训练时间测试

| Model       | Speed up ratio (Compare with Pytorch) |
| ----------- | ------------------------------------- |
| PointNet    | 1.06                                  |
| PointNet ++ | 1.85                                  |
| PointCNN    | None (No pytorch implementation)      |
| DGCNN       | 1.05                                  |
| PointConv   | None (No pytorch implementation)      |

## 目录结构

```
.
├── data_utils                   # 数据相关工具
│   ├── data                     # 数据存放路径
│   ├── modelnet40_loader.py
│   └── shapenet_loader.py
├── misc
│   ├── layers.py
│   ├── ops.py
│   ├── pointconv_utils.py
│   └── utils.py
├── networks
│   ├── cls
│   │   ├── dgcnn.py
│   │   ├── pointcnn.py
│   │   ├── pointconv.py
│   │   ├── pointnet2.py
│   │   └── pointnet.py
│   └── seg
│       ├── dgcnn_partseg.py
│       ├── pointcnn_partseg.py
│       ├── pointconv_partseg.py
│       ├── pointnet2_partseg.py
│       └── pointnet_partseg.py

├── README.md
├── run_cls.sh
├── run_partseg.sh
├── train_cls.py
└── train_partseg.py
```

非常欢迎您使用计图的点云库进行相关的研究，如在使用中有问题，欢迎提交 issues。
## Reference code :
https://github.com/AnTao97/dgcnn.pytorch
