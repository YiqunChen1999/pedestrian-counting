# 可变形卷积Deformable Conv

在ATSS Head内将部分原始卷积替换为可变形卷积，优化卷积结构

由于deformable conv需要学习参数较多，因此仅在最后一层添加deformable conv，具体为self.atss_cls，模块名为ATSSHead_deform，需在对应__init__.py里添加本类

由于占用内存较大，samples_per_gpu需改为1（GPU内存大的可忽略本条）
