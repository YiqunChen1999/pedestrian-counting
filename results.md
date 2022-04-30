
# Results

本文档记录了各个实验的实验结果，其中，AP & AR 仅作为参考。

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

| Model                | lr    | batch size | policy     | MR@R   | MR@S   | MR@RO  | MR@All |
|----------------------|-------|------------|------------|--------|--------|--------|--------|
| ATSS-Res50           | 0.005 | 16         | 1x         | 25.01% | 35.51% | 61.97% | 41.30% |
| ATSS-Res50           | 0.005 | 16         | mstrain-3x | 23.03% | 30.34% | 59.58% | 38.79% |
| ATSS-Res50-DyConvFPN | 0.005 | 16         | 1x         | 25.07% | 34.38% | 61.20% | 41.48% |
| ATSS-Res101          | 0.005 | 16         | 1x         | 24.61% | 34.19% | 64.36% | 41.47% |
| ATSS-Res101          | 0.005 | 16         | mstrain-3x | 23.06% | 30.92% | 61.12% | 39.14% |

**注意：** 从实验日志来看，似乎 ATSS-Res101 存在过拟合的情况，其在验证集上的 MR 略呈 U 形。

### TJU-DHD-Ped-campus

| Model       | lr    | batch size | policy     | MR@R   | MR@S   | MR@RO  | MR@All |
|-------------|-------|------------|------------|--------|--------|--------|--------|
| ATSS-Res101 | 0.005 | 16         | 1x         | 32.45% | 67.55% | 68.79% | 41.96% |
| ATSS-Res101 | 0.005 | 16         | mstrain-3x |        |        |        |        |

## 1. TJU-DHD-Pedestrian-Traffic 

### 1.1. configs/atss/atss_r50_fpn_1x_coco.py

- 训练命令：CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"

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

- 训练命令：CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_mstrain_800-1024_3x_coco.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"

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

- 训练命令：CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"

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

- 训练命令：CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=18181 tools/dist_train.sh configs/atss/atss_r101_fpn_mstrain_800-1024_3x_dhd_ped_traffic.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"

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

| Miss Rate                                  | baseline  | fpn_dcd (only smooth)    |
|--------------------------------------------|-----------|--------------------------|
| Average Miss Rate  (MR) @ Reasonable       | 25.01%    | 25.07%                   |
| Average Miss Rate  (MR) @ ReasonableSmall  | 35.51%    | **34.38%**               |
| Average Miss Rate  (MR) @ ReasonableHeavy  | 61.97%    | **61.20%**               |
| Average Miss Rate  (MR) @ All              | 41.30%    | 41.48%                   |

## 2. TJU-DHD-Pedestrian-Campus

### 2.1. configs/atss/atss_r50_fpn_1x_dhd_ped_campus.py

- 训练命令：CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_dhd_ped_campus.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"


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

## 3. 其他数据集
### 3.1. 配置文件路径


