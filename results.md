
# Results

本文档记录了各个实验的实验结果，其中，AP & AR 仅作为参考。

**注意：** 所有 `MR` 指标都是越小越好。

## 0. 全部结果速览

| 缩写   | Target Attribute      | Height Range  | Visibility Range  |
|--------|-----------------------|--------------|-------------------|
| MR@All | All	                 | [20,+∞)	    | [0.2,+∞)          |
| MR@R   | Reasonable (R)	     | [50,+∞)	    | [0.65,+∞)         |
| MR@L   | Large	             | [100,+∞)	    | [0.65,+∞)         |
| MR@M   | Medium	             | [75,100)	    | [0.65,+∞)         |
| MR@S   | Small	             | [50,75)	    | [0.65,+∞)         |
| MR@B   | Bare	                 | [50,+∞)	    | [0.9, 1]          |
| MR@P   | Partial	             | [50,+∞)	    | [0.65,0.9)        |
| MR@HO  | Heavy (HO)	         | [50,+∞)	    | [0,0.65)          |
| MR@RO  | Reasonable Heavy (RO) | [50,+∞)	    | [0.2,0.65)        |

### TJU-DHD-Ped-traffic

| Model                             | lr      | batch size | policy     | MR@R   | MR@S   | MR@RO  | MR@All |
|-----------------------------------|---------|------------|------------|--------|--------|--------|--------|
| ATSS-Res50                        | 0.005   | 16         | 1x         | 25.01% | 35.51% | 61.97% | 41.30% |
| ATSS-Res50                        | 0.005   | 16         | mstrain-3x | 23.03% | 30.34% | 59.58% | 38.79% |
| ATSS-Res50-DyConvFPN (SmoothOnly) | 0.005   | 16         | 1x         | 25.07% | 34.38% | 61.20% | 41.48% |
| ATSS-Res50-DyConvFPN (All)        | 0.005   | 16         | 1x         | 24.65% | 34.28% | 61.00% | 41.02% |
| ATSS-Res50-DyConvFPN (All)        | 0.005   | 16         | mstrain-3x | 22.17% | 29.90% | 59.18% | 38.20% |
| ATSS-Res50-DyConvHead             | 0.0025  | 8          | 1x         | 24.95% | 35.26% | 61.60% | 41.08% |
| ATSS-Res50-DyConvHead             | 0.005   | 8          | 1x         | 26.73% | 35.44% | 64.31% | 42.56% |
| ATSS-Res50-DyConvHead             | 0.00125 | 8          | 1x         | 25.80% | 36.24% | 62.24% | 42.08% |
| ATSS-Res101                       | 0.005   | 16         | 1x         | 24.61% | 34.19% | 64.36% | 41.47% |
| ATSS-Res101                       | 0.005   | 16         | mstrain-3x | 23.06% | 30.92% | 61.12% | 39.14% |
| ATSS-Res50-DeformConv             | 0.005   | 16         | 1x         | 61.76% | 57.87% | 84.90% | 71.40% |
| ATSS-Res50-DeformConv             | 0.005   | 16         | 2x         | 59.50% | 54.53% | 85.77% | 69.47% |
| ATSS-Res50-DeformConv             | 0.005   | 16         | 3x         | 34.72% | 42.44% | 68.63% | 50.17% |
| ATSS-Res50-DeformConv             | 0.005   | 16         | mstrain-3x | 27.03% | 35.06% | 64.15% | 43.07% |

**注意：** 从实验日志来看，似乎 ATSS-Res101 存在过拟合的情况，其在验证集上的 MR 略呈 U 形。

### TJU-DHD-Ped-campus

| Model       | lr    | batch size | policy     | MR@R   | MR@S   | MR@RO  | MR@All |
|-------------|-------|------------|------------|--------|--------|--------|--------|
| ATSS-Res50  | 0.005 | 16         | 1x         | 32.45% | 67.55% | 68.79% | 41.96% |
| ATSS-Res50  | 0.005 | 16         | mstrain-3x | 29.43% | 59.96% | 65.14% | 39.00% |
| ATSS-Res101 | 0.005 | 16         | 1x         |        |        |        |        |
| ATSS-Res101 | 0.005 | 16         | mstrain-3x |        |        |        |        |

## 1. TJU-DHD-Pedestrian-Traffic 

### 1.1. configs/atss/atss_r50_fpn_1x_coco.py

训练命令：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```

| Miss Rate                                  | Value  |
|--------------------------------------------|--------|
| Average Miss Rate  (MR) @ Reasonable       | 25.01% |
| Average Miss Rate  (MR) @ ReasonableSmall  | 35.51% |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 61.97% |
| Average Miss Rate  (MR) @ All              | 41.30% |

| AP & AR                 | IoU           | area        | maxDets     | Value |
|-------------------------|---------------|-------------|-------------|-------|
| Average Precision  (AP) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.422 |
| Average Precision  (AP) | IoU=0.50      | area=   all | maxDets=100 | 0.790 |
| Average Precision  (AP) | IoU=0.75      | area=   all | maxDets=100 | 0.399 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.210 |
| Average Precision  (AP) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.457 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.629 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=  1 | 0.218 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets= 10 | 0.502 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.521 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.332 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.567 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.695 |


### 1.2. configs/atss/atss_r50_fpn_mstrain_800-1024_3x_coco.py

训练命令：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_mstrain_800-1024_3x_coco.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```

| Miss Rate                                  | Value  |
|--------------------------------------------|--------|
| Average Miss Rate  (MR) @ Reasonable       | 23.03% |
| Average Miss Rate  (MR) @ ReasonableSmall  | 30.34% |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 59.58% |
| Average Miss Rate  (MR) @ All              | 38.79% |

| AP & AR                 | IoU           | area        | maxDets     | Value |
|-------------------------|---------------|-------------|-------------|-------|
| Average Precision  (AP) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.447 |
| Average Precision  (AP) | IoU=0.50      | area=   all | maxDets=100 | 0.814 |
| Average Precision  (AP) | IoU=0.75      | area=   all | maxDets=100 | 0.430 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.239 |
| Average Precision  (AP) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.480 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.639 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=  1 | 0.229 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets= 10 | 0.528 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.545 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.368 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.588 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.707 |

### 1.3. configs/atss/atss_r101_fpn_1x_coco.py

训练命令：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```

| Miss Rate                                  | Value  |
|--------------------------------------------|--------|
| Average Miss Rate  (MR) @ Reasonable       | 24.61% |
| Average Miss Rate  (MR) @ ReasonableSmall  | 34.19% |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 64.36% |
| Average Miss Rate  (MR) @ All              | 41.47% |

| AP & AR                 | IoU           | area        | maxDets     | Value |
|-------------------------|---------------|-------------|-------------|-------|
| Average Precision  (AP) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.422 |
| Average Precision  (AP) | IoU=0.50      | area=   all | maxDets=100 | 0.788 |
| Average Precision  (AP) | IoU=0.75      | area=   all | maxDets=100 | 0.398 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.200 |
| Average Precision  (AP) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.455 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.642 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=  1 | 0.221 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets= 10 | 0.505 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.521 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.329 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.564 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.713 |

### 1.4. configs/atss/atss_r101_fpn_mstrain_800-1024_3x_dhd_ped_traffic.py

训练命令：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=18181 tools/dist_train.sh configs/atss/atss_r101_fpn_mstrain_800-1024_3x_dhd_ped_traffic.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```

| Miss Rate                                  | Value  |
|--------------------------------------------|--------|
| Average Miss Rate  (MR) @ Reasonable       | 23.06% |
| Average Miss Rate  (MR) @ ReasonableSmall  | 30.92% |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 61.12% |
| Average Miss Rate  (MR) @ All              | 39.14% |

| AP & AR                 | IoU           | area        | maxDets     | Value |
|-------------------------|---------------|-------------|-------------|-------|
| Average Precision  (AP) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.443 |
| Average Precision  (AP) | IoU=0.50      | area=   all | maxDets=100 | 0.803 |
| Average Precision  (AP) | IoU=0.75      | area=   all | maxDets=100 | 0.433 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.225 |
| Average Precision  (AP) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.474 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.639 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=  1 | 0.227 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets= 10 | 0.520 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.534 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.354 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.576 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.707 |


### 1.5. configs/atss/atss_r50_fpn_dcd_1x_dhd_ped_traffic.py

**注意：** 详细说明请见 [dev-dyconv](https://github.com/YiqunChen1999/pedestrian-counting/tree/dev-dyconv) 分支。

训练命令：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=18181 tools/dist_train.sh configs/atss/atss_r50_fpn_dcd_1x_dhd_ped_traffic.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```

| Miss Rate                                  | baseline  | fpn_dcd (ALL) (1x) | fpn_dcd (ALL) (mstrain-3x) |
|--------------------------------------------|-----------|---------------     |----------------------------|
| Average Miss Rate  (MR) @ Reasonable       | 25.01%    | **24.65%**         | **24.65%**                 |
| Average Miss Rate  (MR) @ ReasonableSmall  | 35.51%    | **34.28%**         | **34.28%**                 |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 61.97%    | **61.00%**         | **61.00%**                 |
| Average Miss Rate  (MR) @ All              | 41.30%    | **41.02%**         | **41.02%**                 |

### 1.6. configs/atss/atss_r50_fpn_dcd_3x_dhd_ped_traffic.py

**注意：** 详细说明请见 [dev-dyconv](https://github.com/YiqunChen1999/pedestrian-counting/tree/dev-dyconv) 分支。

训练命令：[配置文件未给出]

| Miss Rate                                  | baseline  | fpn_dcd (ALL) (3x) |
|--------------------------------------------|-----------|--------------------|
| Average Miss Rate  (MR) @ Reasonable       | 25.01%    | **22.17%**         |
| Average Miss Rate  (MR) @ ReasonableSmall  | 35.51%    | **29.90%**         |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 61.97%    | **59.18%**         |
| Average Miss Rate  (MR) @ All              | 41.30%    | **38.20%**         |

### 1.7. configs/atss/atss_r50_fpn_1x_coco_head_dcd.py

**注意：** 详细说明请见 [dconv](https://github.com/YiqunChen1999/pedestrian-counting/tree/dconv) 分支。

训练命令：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=19197 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco_head_dcd.py 4 --cfg-options "data.samples_per_gpu=2 optimizer.lr=0.0025"
```

| Miss Rate                                  | baseline  | atss_head_dcd (1x) |
|--------------------------------------------|-----------|--------------------|
| Average Miss Rate  (MR) @ Reasonable       | 25.01%    | **24.95%**         |
| Average Miss Rate  (MR) @ ReasonableSmall  | 35.51%    | **35.26%**         |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 61.97%    | **61.60%**         |
| Average Miss Rate  (MR) @ All              | 41.30%    | **41.08%**         |

## 2. TJU-DHD-Pedestrian-Campus

### 2.1. configs/atss/atss_r50_fpn_1x_dhd_ped_campus.py

训练命令：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_dhd_ped_campus.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```

| Miss Rate                                  | Value  |
|--------------------------------------------|--------|
| Average Miss Rate  (MR) @ Reasonable       | 32.45% |
| Average Miss Rate  (MR) @ ReasonableSmall  | 67.55% |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 68.79% |
| Average Miss Rate  (MR) @ All              | 41.96% |

| AP & AR                 | IoU           | area        | maxDets     | Value |
|-------------------------|---------------|-------------|-------------|-------|
| Average Precision  (AP) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.551 |
| Average Precision  (AP) | IoU=0.50      | area=   all | maxDets=100 | 0.832 |
| Average Precision  (AP) | IoU=0.75      | area=   all | maxDets=100 | 0.594 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.048 |
| Average Precision  (AP) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.339 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.696 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=  1 | 0.121 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets= 10 | 0.515 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.627 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.127 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.473 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.757 |

### 2.1. configs/atss/atss_r50_fpn_mstrain_800-1024_3x_dhd_ped_campus.py

训练命令：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_mstrain_800-1024_3x_dhd_ped_campus.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```

| Miss Rate                                  | Value  |
|--------------------------------------------|--------|
| Average Miss Rate  (MR) @ Reasonable       | 29.43% |
| Average Miss Rate  (MR) @ ReasonableSmall  | 59.96% |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 65.14% |
| Average Miss Rate  (MR) @ All              | 39.00% |

| AP & AR                 | IoU           | area        | maxDets     | Value |
|-------------------------|---------------|-------------|-------------|-------|
| Average Precision  (AP) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.583 |
| Average Precision  (AP) | IoU=0.50      | area=   all | maxDets=100 | 0.858 |
| Average Precision  (AP) | IoU=0.75      | area=   all | maxDets=100 | 0.630 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.069 |
| Average Precision  (AP) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.382 |
| Average Precision  (AP) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.720 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=  1 | 0.124 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets= 10 | 0.540 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.654 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.170 |
| Average Recall     (AR) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.508 |
| Average Recall     (AR) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.777 |

## 3. 其他数据集
### 3.1. 配置文件路径


