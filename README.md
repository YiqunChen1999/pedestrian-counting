
# Pedestrian Counting DyConv

本文档介绍如何将我完成的动态卷积部分添加到 @陈逸群 完成的目标检测baseline代码中。

**注意** 以下所有以 `[全大写英文]` 出现的均为需要按照自己的机器情况修改的地方，例如：`[PROJECT_ROOT]` 表示本项目文件夹在自己机器的位置，实际中可能为：`~/projects/pedestrian-counting/`。

**注意：** 基础流程请参考 @陈逸群 编写的文档，本文档主要目的在于介绍新增的动态卷积应该如何添加及相关注意事项。

## 更新日志

- 2022-04-30：发布 动态卷积代码 与使用说明。

## 1. 代码说明

### 1.1. 动态卷积代码存放位置
此处代码存放位置均为项目中的文件夹位置

```bash
/counter/models/fpn_dcd.py
```
该位置存放动态卷积源码

```bash
/configs/atss/atss_r50_dcd_fpn_1x_coco.py
```
该位置存放使用动态卷积FPN的配置文件


### 1.2. 代码说明

- 对fpn_dcd.py的说明：
该代码使用动态卷积实现了FPN，此处的FPN是写死的，也就是不可配置的，抽取把backbone的第1,2,3个特征层，即x[1],x[2],x[3],输出5个特征图。分别使用卷积层toplayer,smooth1,smooth2,smooth3,latlayer1,latlayer2,extra1,extra2实现，FPN的结构可以参考博客[MMDet逐行代码解读之ResNet50+FPN](https://blog.csdn.net/wulele2/article/details/122703149)。
代码中的类conv_dy就是动态卷积层的构建，如果想要用nn.Conv2d构建,可以直接替换conv_dy从而构建普通卷积层。

目前为止我试验了两种方式，一种是只对smooth1,smooth2,smooth3三层使用动态卷积，实验结果见后。
另一种是对所有用到的卷积层都使用动态卷积，但可能是学习率或者其他什么参数的问题，导致收敛不是很明显，如果想要出较好的实验结果，可能后期要麻烦调参的同学。

- 对atss_r50_dcd_fpn_1x_coco.py的说明
这里主要是将动态卷积的FPN整合到整个目标检测网络中，只要修改neck中的type='FPN_dcd'即可。在该配置文件的开头需要进行custom_import以将自己编写的动态卷积FPN导入。

- 注意
目前我只完成了固定形式的动态卷积，也就是无法进行配置，如需配置，需要手动修改fpn_dcd.py文件，后期如有时间会对代码进行修改以方便配置（小白只会手动改代码去配置啊sigh）。

### 1.3 模型运行
和baseline的运行方式类似，只需要改一下配置文件的地址即可
多卡：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_dcd_fpn_1x_coco.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```
单卡：
```bash
python tools/train.py configs/atss/atss_r50_fpn_1x_coco.py --cfg-options "optimizer.lr=0.00125"
```
- 特别提醒！

使用单卡训练一定要调整好学习率，我在使用单卡训练时因为学习率的问题，总是计算溢出，上述命令的lr=0.00125只是推算出的学习率，如果不合适可能还需要调整，应该是向小的方向调整。学习率调整可以适当参考博客[mmdetecion-学习率调整-线性缩放原则](https://blog.csdn.net/qq_20793791/article/details/108399919)。

- 如果真的碰到什么问题没办法解决，可以联系作者，或者直接找 @陈逸群 也十分OK，毕竟我都是跟他学的orz。

## 2. 实验结果

### 2.1. TJU-DHD-Pedestrian-Traffic 

#### 2.1.1. configs/atss/atss_r50_dcd_fpn_1x_coco.py

| Miss Rate                                  | baseline  | fpn_dcd<br>(only smooth)      |
|--------------------------------------------|-----------|--------------|
| Average Miss Rate  (MR) @ Reasonable       |  25.01%   |    25.07%    |
| Average Miss Rate  (MR) @ ReasonableSmall  |  35.51%   |    <b>34.38%    |
| Average Miss Rate  (MR) @ ReasonableHeavy  |  61.97%   |    <b>61.20%    |
| Average Miss Rate  (MR) @ All              |  41.30%   |    41.48%    |

结论：动态卷积还是有一定的效果的。
（Miss Rate都是越低越好）





