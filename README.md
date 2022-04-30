
# Pedestrian Counting

本文档记录了如何零基础运行本项目，均以 Linux 为例。

**注意** 以下所有以 `[全大写英文]` 出现的均为需要按照自己的机器情况修改的地方，例如：`[PROJECT_ROOT]` 表示本项目文件夹在自己机器的位置，实际中可能为：`~/projects/pedestrian-counting/`。

**注意：** 如果不会用 open-mmlab 相关工具，强烈建议按照本文档先跑起来。

**注意：** 所有的实验结果均已迁移到 [results.md](./results.md) 文档中。

## 更新日志

- 2022-04-30：更新了 [results.md](./results.md) 文档，所有实验结果均存放于此。
- 2022-04-30：更新了 baseline （多尺度训练，3x policy）并增加了动态卷积的模块与配置。
- 2022-04-29：新增了 [Issues 腾讯文档](https://docs.qq.com/doc/DY0VRVU5Pb0RkTExU?scene=f37afff1f6077c92c2401eff8Rsnn1)，方便记录已有问题及解决方案。若遇到问题，请先到腾讯文档中查看是否已有解决方案。
- 2022-04-28：更新 README 文件并新增 mmdet 版本警告。
- 2022-04-27：发布 baseline 与代码使用说明。

## TODO 

- [ ] 增加 DHD-Ped-campus 子集的 baseline。

## 1. 如何开始

### 1.1. 将本项目代码下载到服务器

```bash
cd [FOLDER]
git clone https://github.com/yiqunchen1999/pedestrian-counting.git
```

### 1.2. 下载数据集

新建一个 `[DATA_ROOT]`，例如：`/data1/share/TJU-DHD/`。 此处以 TJU-DHD-traffic 为例，在 `[DATA_ROOT]` 下新建一个文件夹 `TJU-Ped-traffic`：
```bash
cd [DATA_ROOT]
mkdir TJU-Ped-traffic
```
从 [TJU-DHD](https://github.com/tjubiit/TJU-DHD) 官方连接处下载以下文件并上传到服务器文件夹 `[DATA_ROOT]/TJU-Ped-traffic/` 下 (例如，`/data1/share/TJU-DHD/TJU-Ped-traffic/`)：

- dhd_traffic_trainval_images.zip
- dhd_pedestrian_traffic_trainval_annos.zip

如果需要 **使用 Linux 终端下载 OneDrive 文件** 的教程，可联系作者获取。

完成之后，应该在 `[DATA_ROOT]/TJU-Ped-traffic` 下有所需的 .zip 文件，例如：
```bash
/data1/share/TJU-DHD/TJU-Ped-traffic/dhd_traffic_trainval_images.zip
/data1/share/TJU-DHD/TJU-Ped-traffic/dhd_pedestrian_traffic_trainval_annos.zip
```

解压文件：
```bash
cd [DATA_ROOT]
unzip dhd_traffic_trainval_images.zip
unzip dhd_pedestrian_traffic_trainval_annos.zip
```

接下来将数据目录链接到项目目录下：

```bash
cd [PROJECT_ROOT]
ln -s [DATA_ROOT] data
```
例如：
```bash
cd ~/projects/pedestrian-counting
ln -s /data1/share/TJU-DHD/ data
```
完成这一步后，在项目目录下应该会出现:
```bash
data/TJU-Ped-traffic/dhd_traffic_trainval_images.zip
data/TJU-Ped-traffic/dhd_pedestrian_traffic_trainval_annos.zip
data/TJU-Ped-traffic/dhd_traffic
    |_ images
        |_ test
        |_ train
        |_ val
data/TJU-Ped-traffic/dhd_pedestrian
    |_ ped_traffic
        |_ annotations
            |_ dhd_pedestrian_traffic_train.json
            |_ dhd_pedestrian_traffic_val.json
```

**注意：** 这一步仅仅是进行路径关联，并不会把文件复制到项目目录下，对于家目录容量有限的情况很有帮助。

**可选：** 如果担心自己的家目录存储空间很小，也可以考虑将存储程序输出的地方链接到项目目录 `[PROJECT_ROOT]` 下的 `work_dirs` ，`work_dirs` 是 [openmm-lab](https://github.com/open-mmlab) 各种包保存程序输出的文件夹。

### 1.3. 准备运行环境

太长不看版：
```bash
cd [PROJECT_ROOT]
conda create -n dl python=3.9 -y
conda activate dl
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet
pip install -r requirements.txt
python setup.py develop
```


**可选但推荐执行：** 下载 anaconda 或者 miniconda，然后创建一个虚拟环境：
```bash
conda create -n [ENV] python=3.9 -y
conda activate [ENV]
```

例如：
```bash
conda create -n dl python=3.9 -y
conda activate dl
```

参见 [PyTorch 官网](https://pytorch.org/) 安装 PyTorch，例如：
```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

按照 [mmdetection 文档](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html#id2) 给出的安装流程创建安装有关依赖。这里给出一个示例（CUDA=11.3, PyTorch=1.11）：
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet
```

**警告：** 由于 `mmdet` 于 2022-04-27 晚更新了 pypi 的版本，因此与之前版本的启动工具不兼容，关于不同版本如何进行训练与测试将在后面说明。这里默认安装 2022-04-27 的新版本 `mmdet==2.24.0`。

接下来安装本项目的代码：
```bash
cd [PROJECT_ROOT]
pip install -r requirements.txt
python setup.py develop
```

### 1.4. 运行训练程序

**注意：** 本小节仅给出了能够运行起来的示例，关于训练脚本的更多用法请参见 [mmdetection 教程](https://mmdetection.readthedocs.io/zh_CN/latest/2_new_data_model.html#id4)

至此就可以开始训练模型了，接下来给出一个示例：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py 4 --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005"
```

各个参数解释如下：
- CUDA_VISIBLE_DEVICES=0,1,2,3 表示运行程序时只能使用标号为 0，1，2，3 的显卡，此时使用 PyTorch 观察时只有最多四张卡可用，即
```python
>>> import torch
>>> torch.cuda.device_count()
4
```
**注意：** 请不要插入额外的空格。

- PORT=19191 表示分布式训练时该程序用于通信的端口，如果报错显示该端口已经被占用，则将 19191 修改为可以运行的端口号即可；
- tools/dist_train.sh 表示运行脚本 `[PROJECT_ROOT]/tools/dist_train.sh`，这是用于分布式训练的启动脚本；
- configs/atss/atss_r50_fpn_1x_coco.py 表示使用 `[PROJECT_ROOT]/configs/atss/atss_r50_fpn_1x_coco.py` 这个配置文件，该文件中定义了模型结构，数据集路径，优化器参数，损失函数，训练策略等等，其他配置文件可以在同一文件夹下找到。关于配置文件的使用方法可以在 [mmdetection 教程](https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/config.html) 中找到，下面也给出了自己进行实验时可能会修改的地方；
- 4 表示用于分布式训练的 GPU 数目，不能多于可用的显卡数；
- --cfg-options "data.samples_per_gpu=4 optimizer.lr=0.005" 表示用终端输入的参数来替代配置文件中的参数，例如，这里指明每张卡有 4 个样本（4 张图片），优化器的学习率为 0.005。**注意：** 因为这里的空格具有特殊含义，等号两侧不能有空格；

如果只能单卡训练：
```bash
python tools/train.py configs/atss/atss_r50_fpn_1x_coco.py
```

**警告：** 由于 `mmdet` 于 2022-04-27 晚更新了 pypi 的版本，因此与之前版本的启动工具不兼容。如果安装的 `mmdet` 版本低于 `2.24.0` （例如，`2.23.0`），则只需要将以上命令中的 `tools/` 更换成 `.tools/` 即可（注意这个 `.`）。即：`tools` 下放的是 `mmdet==2.24.0` 的启动工具，`.tools` 下放的是 `mmdet==2.23.0` 的启动工具。

### 1.5. 测试已有模型

**注意：** 本小节仅给出了能够运行起来的示例，关于测试脚本的更多用法请参见 [mmdetection 教程](https://mmdetection.readthedocs.io/zh_CN/latest/1_exist_data_model.html#id10)

```bash
cd [PROJECT_ROOT]
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=19191 tools/dist_test.sh configs/atss/atss_r50_fpn_1x_coco.py work_dirs/atss_r50_fpn_1x_coco/latest.pth 4 --eval bbox miss_rate
```

- CUDA_VISIBLE_DEVICES=0,1,2,3 见上一小节；
- PORT=19191 见上一小节；
- tools/dist_test.sh 类似于上一小节；
- configs/atss/atss_r50_fpn_1x_coco.py 见上一小节；
- work_dirs/atss_r50_fpn_1x_coco/latest.pth 为训练好的模型的存放位置，存放模型参数的文件夹名称与配置文件的文件名一致，latest.pth 表示最新的模型，该目录下还放着终端输出日志，运行配置，检查点保存的模型；
- 4 见上一小节；
- --eval bbox miss_rate 表示测试时应该使用的评价指标， bbox 表示基于 bbox 交并比的 COCO 评价指标，miss_rate 表示行人检测的评价指标，作者已经适配该指标；

如果只能单卡测试：
```bash
python tools/test.py configs/atss/atss_r50_fpn_1x_coco.py work_dirs/atss_r50_fpn_1x_coco/latest.pth --eval bbox miss_rate
```

**警告：** 由于 `mmdet` 于 2022-04-27 晚更新了 pypi 的版本，因此与之前版本的启动工具不兼容。如果安装的 `mmdet` 版本低于 `2.24.0` （例如，`2.23.0`），则只需要将以上命令中的 `tools/` 更换成 `.tools/` 即可（注意这个 `.`）。即：`tools` 下放的是 `mmdet==2.24.0` 的启动工具，`.tools` 下放的是 `mmdet==2.23.0` 的启动工具。

## 2. 关于数据集应该注意的一些细节

**注意：** 这一部分可能会更新。

- 数据集的标注应该采用 COCO 格式的标注，如果不是，请转换为 COCO 格式。COCO 格式的数据集中 `bbox` 的标注为 `(x, y, w, h)`。
- 当新增数据集时，请先确保数据格式为 COCO 格式的格式，然后修改 `[PROJECT_ROOT]/configs/_base_/datasets/pedestrian.py` 中 `data_root = 'data/TJU-Ped-traffic/'`.
- `mmdet.datasets.api_wrappers` 中导入的 `COCO, COCOeval` 与 `counter.utils.eval_MR_multisetup, counter.utils.coco` 中导入的 `COCO, COCOeval` 并不完全一致，请勿相互替代，因为前者用于 `bbox, seg` 的评估，后者用于 `miss_rate` 的评估。

## 3. 关于代码应该注意的一些细节

- TJU-DHD 原始论文中给出的训练时与测试时的图像的分辨率均设为 `2048 x 1024`，与 `RetinaNet, FCOS, ATSS` 等的原始配置都不一样。如果使用原始配置，会导致 `miss_rate` 很不正常（小目标不能被识别）。
- TJU-DHD 原始论文给出 `RetinaNet` 的学习率设置为 `0.005`，是默认参数的一半，由于 `RetinaNet` 衍生版的配置一般与 `RetinaNet` 相同，故 `FCOS, ATSS` 一般而言也应该与 `RetinaNet` 保持一致（学习率修改为 0.005）。

## 4. 进一步开发前必读

本项目依赖于 `git` 进行版本控制，如果服务器无法访问 `github`，可联系作者将本项目迁移至 `gitlab` 或者 `gitee`。使用 `git` 对本项目的好处主要有：
- 现有的多个带推进工作可以同时在多个分支进行，只需要最终合并即可；
- 方便回退到之前的版本；

### 4.1. 分支管理

主分支（master branch）用于保存最新的已验证的代码，因此进行开发时请各自新建一个分支，避免代码混乱：
```bash
cd [PROJECT_ROOT]
git checkout -b [BRANCH]
```
其中，`[BRANCH]` 表示自己的分支名称，例如，动态卷积开发可命名为 `dev-dyconv`，语义分割可命名为 `dev-seg`。

### 4.2. 代码提交

本地分支上传到 `GitHub` 时，请指定分支名称，如：
```bash
git add --all
git commit -m [YOUR_COMMENT]
git push https://github.com/yiqunchen1999/pedestrian-counting.git [BRANCH]
```
其中 `[YOUR_COMMENT]` 是对本次提交的修改的说明，`[BRANCH]` 是自己的分支名称。

当当前分支确实可用时，可以将其合并到主分支。

### 4.3. 示例

此处以新建一个文档分支为例，分支名为 `docs`：
```bash
git checkout -b docs
# 输出：Switched to a new branch 'docs'
# 自己进行某些修改，例如，完善文档
git add --all
git commit -m "update readme"
git push https://github.com/yiqunchen1999/pedestrian-counting.git docs
```

## 5. 评价指标

**注意：** 以下评价指标均为越小越好。

| Target Attribute      | Height Range  | Visibility Range  |
|-----------------------|---------------|-------------------|
| All	                | [20,+∞)	    | [0.2,+∞)          |
| Reasonable (R)	    | [50,+∞)	    | [0.65,+∞)         |
| Large	                | [100,+∞)	    | [0.65,+∞)         |
| Medium	            | [75,100)	    | [0.65,+∞)         |
| Small	                | [50,75)	    | [0.65,+∞)         |
| Bare	                | [50,+∞)	    | [0.9,1]           |
| Partial	            | [50,+∞)	    | [0.65,0.9)        |
| Heavy (HO)	        | [50,+∞)	    | [0,0.65)          |
| Reasonable Heavy (RO)	| [50,+∞)	    | [0.2,0.65)        |

## 6. 实验数据

参见 [results.md](./results.md)

