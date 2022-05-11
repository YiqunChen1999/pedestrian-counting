
# Pedestrian Counting

本文档记录了如何如何将FPN中所用到的动态卷积模块应用到atss-head中

**注意** 以下所有以 `[全大写英文]` 出现的均为需要按照自己的机器情况修改的地方，例如：`[PROJECT_ROOT]` 表示本项目文件夹在自己机器的位置，实际中可能为：`~/projects/pedestrian-counting/`。

**注意：** 所有的实验结果均已迁移到 [results.md](./results.md) 文档中。

## 更新日志

- 2022-05-01：将更新后的动态卷积模块应用至atss head中。

## 1. 代码说明

### 1.1. 代码存放位置

```bash
/counter/models/atss_head_dcd.py
```
该位置为添加了动态卷积模块的head代码

```bash
/configs/atss/atss_r50_fpn_1x_coco_head_dcd.py
```
该位置为使用的配置文件，在head部分略有不同，其余部分与原有使用的配置文件相同

### 1.2. 代码说明

- 对atss_head_dcd.py的说明
此代码使用动态卷积模块与ptorch自带的GN、ReLU模块对mmcv中的Conv模块进行替换
分别将cls分支与reg中的全部卷积层替换为动态卷积

- 对配置文件的说明
配置文件与mmdet所提供的初始配置文件完全相同
如果需要修改动态卷积中的K与t参数，直接在atss_head_dcd.py中的类初始化中修改即可

### 1.3. 模型训练
模型训练的过程与陈逸群所写的Readme基本完全相同

多卡：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=19197 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco_head_dcd.py 4 --cfg-options "data.samples_per_gpu=2 optimizer.lr=0.0025"
```
单卡：
```bash
python tools/train.py configs/atss/atss_r50_fpn_1x_coco_head_dcd.py --cfg-options "optimizer.lr=0.00125"
```
-注意：
这里使用多卡和单卡训练时batch size都调整为2，若为4则会溢出，学习率也相应地进行了调整
其中多卡的训练率经过实验验证，0.0025情况下效果最好

-参数设置
动态卷积模块中K=4， t=30，需要调参优化

## 2. 实验结果

### 2.1. TJU-DHD-Pedestrian-Traffic 

#### 2.1.1. configs/atss/atss_r50_fpn_1x_coco_head_dcd.py

| Miss Rate                                  | baseline  | head_dcd_0.0025|head_dcd_0.005|head_dcd_0.00125
|--------------------------------------------|-----------|--------------|
| Average Miss Rate  (MR) @ Reasonable       |  25.01%   |    <b>24.95%    |26.73%|25.80%|
| Average Miss Rate  (MR) @ ReasonableSmall  |  35.51%   |    <b>35.26%    |<b>35.44%|36.24%|
| Average Miss Rate  (MR) @ ReasonableHeavy  |  61.97%   |    <b>61.60%    |64.31%|62.24%|
| Average Miss Rate  (MR) @ All              |  41.30%   |    <b>41.08%    |42.56%|42.08%|

