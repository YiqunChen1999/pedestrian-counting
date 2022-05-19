# 可变形卷积Deformable Conv

## 1.代码修改

在ATSS Head内将部分原始卷积替换为可变形卷积，优化卷积结构

由于deformable conv需要学习参数较多，因此仅在最后一层添加deformable conv，具体为self.atss_cls，模块名为ATSSHead_deform，需在对应__init__.py里添加本类

由于占用内存较大，samples_per_gpu需改为1（GPU内存大的可忽略本条）

## 2.实验结果

#### 2.1.1. configs/atss/atss_r50_fpn_1x_coco.py

| Miss Rate                                  | cls | cls+reg      | cls+reg+centerness|
|--------------------------------------------|-----------|--------------|---------------------------------------|
| Average Miss Rate  (MR) @ Reasonable       |  /   |    44.44%    |58.80%|
| Average Miss Rate  (MR) @ ReasonableSmall  |  /   |    59.84%    |69.62%|
| Average Miss Rate  (MR) @ ReasonableHeavy  |  /   |    71.32%    |76.83%|
| Average Miss Rate  (MR) @ All              |  /   |    57.66%    |68.17%|


由于可变形卷积对每一个3x3卷积核都要额外学习8个偏置参数，所以只加在最后几层，对应self.cls、self.reg、self.centerness

由于GPU内存空间有限及训练epoch数限制，可变形卷积没有起到优化实验结果的作用是正常的。。。
