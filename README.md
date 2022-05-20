
# Pedestrian Counting DyConv

本文档介绍如何将我完成的动态卷积部分添加到 @陈逸群 完成的目标检测baseline代码中。

**注意** 以下所有以 `[全大写英文]` 出现的均为需要按照自己的机器情况修改的地方，例如：`[PROJECT_ROOT]` 表示本项目文件夹在自己机器的位置，实际中可能为：`~/projects/pedestrian-counting/`。

**注意：** 基础流程请参考 @陈逸群 编写的文档，本文档主要目的在于介绍新增的动态卷积应该如何添加及相关注意事项。

## 更新日志
- 2022-05-20: 更新了矩阵分解实现动态通道融合方法的代码
- 2022-05-11：更新了实验结果，添加对照组
- 2022-05-10：更新了FPN_dcd模块，将动态卷积的超参K与t改为可配置
- 2022-05-10：更新了实验结果与FPN_dcd模块的代码及对应的cfg文件
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
另一种是对所有用到的卷积层都使用动态卷积，实验结果同样见后。

- 对atss_r50_dcd_fpn_1x_coco.py的说明
这里主要是将动态卷积的FPN整合到整个目标检测网络中，只要修改neck中的type='FPN_dcd'即可。在该配置文件的开头需要进行custom_import以将自己编写的动态卷积FPN导入。此处可以参考我写好的cfg文件。

-对atss_r50_de_dcd_fpn_1x_coco.py的说明
使用该配置文件即可运行采用动态通道融合方式的模型，动态通道融合的实现代码在/counter/models/de_fpn_dcd.py中

- 注意
已修改
（目前我只完成了固定形式的动态卷积，也就是无法进行配置，如需配置，需要手动修改fpn_dcd.py文件，后期如有时间会对代码进行修改以方便配置（小白只会手动改代码去配置啊sigh）。）

### 1.3 模型运行
#### 1.3.1 模型训练
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
#### 1.3.2 动态卷积参数及调整建议
动态卷积部分可以配置的参数有两个，一个是代表每层卷积中的动态卷积核的数量K，另外一个是求动态卷积权重的softmax函数的温度t。
此处调整参数需在对应的cfg文件中调整。
理论上K值越大，最后效果就越好，但是因为只改变了FPN，而没有改backbone，所以可能提升效果不明显，推荐的K值：2,4,8
t可以用两种选择：
1.恒定的t，这里推荐t：1,10,30
2.变换的t，即，t的值随着训练在不断变化，根据原论文的情况，可以是前n次迭代，t是30，n次迭代之后，t变为1。这些变化方式可以自己安排。

## 2. 实验结果

### 2.1. TJU-DHD-Pedestrian-Traffic 

#### 2.1.1. configs/atss/atss_r50_dcd_fpn_1x_coco.py

| Miss Rate                                  | baseline  | fpn_dcd<br>(only smooth)      | fpn_dcd<br>(all conv)|dcd_mstrain_x3|mstrain_x3|
|--------------------------------------------|-----------|--------------|---------------------------------------|--------------|----------|
| Average Miss Rate  (MR) @ Reasonable       |  25.01%   |    25.07%    |<b>24.65%|22.17%|23.03%|
| Average Miss Rate  (MR) @ ReasonableSmall  |  35.51%   |    <b>34.38%    |<b>34.28%|29.90%|30.34%|
| Average Miss Rate  (MR) @ ReasonableHeavy  |  61.97%   |    <b>61.20%    |<b>61.00%|59.18%|59.58%|
| Average Miss Rate  (MR) @ All              |  41.30%   |    41.48%    |41.02%|38.20%|38.79%|
 

all conv    : 表示替换fpn中的所有卷积层
 
only smooth ：表示只替换fpn中的smooth卷积层
 
结论：动态卷积还是有一定的效果的。
（Miss Rate都是越低越好）





