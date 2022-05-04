
# Tutorial

本文档是本项目的基本教程，虽然不会面面俱到，但会确保尽量覆盖基本内容。

## MMDetection 简介

简单来说，`MMDetection` 需要关注的有：配置文件、自定义模块注册等。

### 配置文件

所有配置文件都由另一个更为基础的模块 `mmcv` 处理，与 `MMDetection` 解耦。配置文件存储于 `configs` 下，按照类型可以划分为：`_base_`，`common` 以及其他配置。其文件结构如下：
```
PATH2REPO
|_ configs
    |_ _base_
        |_ datasets
            |_ coco_detection.py
            ......
        |_ models
            |_ retinanet.py
            ......
        |_ schedules
            |_ schedule_1x.py
            |_ schedule_2x.py
            ......
        default_runtime.py
    |_ common
        |_ mstrain_xxxxxx.py
        ......
    |_ retinanet
        |_ retinanet_xxxxx.py
        ......
```

其中，`_base_` 被称为原始配置，所有配置文件都应该继承于这些配置，即：配置文件要包含 **数据集**，**模型定义**，**训练策略**，**runtime**：
- `datasets` 定义了使用的数据集以及各个阶段（训练、推理）的数据处理流程，数据集存储位置等信息。
- `models` 定义了模型的基本结构以及训练、推理方式，在 `MMDetection` 中，一个模型通常包含：
    - `backbone`：例如，resnet
    - `neck`：例如：FPN
    - `bbox_head`：包括锚框生成，训练样本采样方法，损失函数等；
    - `train_cfg`
    - `test_cfg`
- `schedules` 定义了诸如 优化器、学习率调节策略 等。
- `default_runtime.py`：提前定义了一些用户基本不必关心的配置，**注意：** 用户自定义的模块需要在这个文件导入：
```python
# Custom imports
custom_imports = dict(
    imports=['counter.models.fpn_dcd', 'counter.data.datasets'],
    allow_failed_imports=False)
```
**注意：** 只能导入一次，否则前面的会被覆盖，详见：[Issues](https://github.com/open-mmlab/mmdetection/issues/7883)

### 自定义模块

可以自定义的模块主要涉及：
- `mmdet.datasets.builder`: 
    - `DATASETS`: 自定义数据集，一般需要继承 `mmdet.datasets.custom.CustomDataset` 或者它的子类（例如：`CocoDataset`），需要复写的方法详见源码。评价指标被集成在了数据集中（`CustomDataset.evaluate`），数据评价时，会调用该方法。
    - `PIPLINES`: 自定义数据处理方式。
- `mmdet.models.builder` 有以下几个注册器，但它们本质上都是同一个 `object`，按需使用即可：
    - `MODELS`：例如，`RetinaNet`
    - `BACKBONES`：例如，`ResNet`
    - `NECKS`：例如，`FPN`
    - `ROI_EXTRACTORS`
    - `SHARED_HEADS`
    - `HEADS`：例如，`ATSSHead`
    - `LOSSES`：例如，`FocalLoss`
    - `DETECTORS`

示例 [counter.data.datasets.py](./counter/data/datasets.py)：
```python
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class PedestrianDataset(CocoDataset):
    pass
```
又如 [counter.models.fpn_dcd.py](./counter/models/fpn_dcd.py)
```python
from mmdet.models.builder import NECKS
from mmcv.runner import BaseModule

@NECKS.register_module()
class FPN_dcd(BaseModule):
    pass
```
如果想要改损失函数，则需要 
```python
from mmdet.models.builder import LOSSES # 导入注册器

@LOSSES.register_module() # 使用注册器注册自定义损失函数
class CustomLoss(nn.Module):
    pass # 自定义损失函数的方法和属性
```
**注意：** 完成之后，切勿忘记在 [configs/_base_/default_runtime.py](./configs/_base_/default_runtime.py) 中的 `custom_imports` 语句添加自定义的模块：
```python
# Custom imports
custom_imports = dict(
    imports=['counter.models.fpn_dcd', 'counter.data.datasets'],
    allow_failed_imports=False)
```
