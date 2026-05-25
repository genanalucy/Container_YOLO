# 集装箱检测 YOLO11 目标检测项目

本项目使用 YOLO11 实现集装箱目标检测，提供从环境准备、数据集配置、训练验证到推理预测的完整流程。文档面向 YOLO 入门和项目复现实验，适用于 Kaggle、服务器或本地 GPU 环境。

数据集来源：[Roboflow Universe container-sq4mu](https://universe.roboflow.com/cont-vkss7/container-sq4mu/dataset/1)，许可协议为 CC BY 4.0。

## 项目功能

本项目包含以下内容：

- YOLO11 集装箱检测训练流程
- YOLO 格式数据集检查脚本
- 基线训练命令和 smoke training 命令
- 推理预测脚本
- Kaggle 运行流程
- 服务器或本地 GPU 运行流程
- tmux 训练进度查看方法
- 多 GPU 训练示例
- 常见问题排查和优化建议

本项目检测 4 种集装箱类型，类别名称在 `data.yaml` 中以 `['0', '1', '2', '3']` 表示。

## 实验结果参考

| 实验 | 模型 | GPU 用法 | epochs | imgsz | batch | P | R | mAP50 | mAP50-95 | 用时 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 基线实验 | yolo11n | 单 GPU | 50 | 640 | 16 | 0.980 | 0.975 | 0.984 | 0.759 | 约 39 分钟 |
| 大模型实验 | yolo11l | 4 GPU | 100 | 640 | 64 | 0.982 | 0.980 | 0.985 | 0.777 | 约 1.027 小时 |

指标说明：

| 指标 | yolo11n | yolo11l 4GPU | 说明 |
|:---:|:---:|:---:|:---|
| 精确率 P | 0.980 | 0.982 | 检测出的目标中真实目标所占比例 |
| 召回率 R | 0.975 | 0.980 | 实际目标中被模型检出的比例 |
| mAP50 | 0.984 | 0.985 | IoU=0.5 时的平均精度 |
| mAP50-95 | 0.759 | 0.777 | 更严格的定位精度指标 |

yolo11l 相比 yolo11n 有提升，但幅度不大。当前数据集的 mAP50 已经较高，mAP50-95 的提升空间更可能与标注质量、图像分辨率、目标框贴合程度和数据集难度有关。

## 项目结构

```text
Container_YOLO/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data_server.yaml
│   └── data_kaggle.yaml
└── scripts/
    ├── train.py
    ├── predict.py
    └── check_dataset.py
```

训练后会生成 `runs/` 目录，常见内容如下：

```text
runs/
└── exp/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.png
    ├── confusion_matrix.png
    ├── results.csv
    └── ...
```

## 环境准备

### 1. 安装 Python

Python 版本要求为 3.8 或更高版本。

```bash
python --version
# Python 3.8.x 或更高版本即可
```

### 2. 安装依赖

```bash
pip install ultralytics PyYAML
```

如果下载速度较慢，可使用国内镜像源：

```bash
pip install ultralytics PyYAML -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 验证 Ultralytics 安装

```bash
yolo version
# 输出类似: ultralytics 8.x.x
```

也可以在 Python 中验证：

```python
from ultralytics import YOLO
print("Ultralytics 安装成功！")
```

## 数据集准备

### YOLO 数据集目录格式

YOLO 数据集需要按以下结构组织：

```text
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

标签文件为 `.txt` 格式，每一行代表一个目标：

```text
类别ID  中心点X  中心点Y  宽度  高度
```

坐标值使用 0 到 1 之间的归一化比例。示例：

```text
0 0.451563 0.641667 0.156250 0.433333
2 0.710938 0.508333 0.120313 0.383333
```

第一列是类别 ID，后四列分别表示目标框中心点和宽高相对于图片尺寸的比例。

### data.yaml 示例

```yaml
train: /path/to/dataset/train/images
val: /path/to/dataset/valid/images
test: /path/to/dataset/test/images

nc: 4
names: ['0', '1', '2', '3']
```

若数据集中没有 `test/images`，可以删除 `test:` 这一行，不影响训练。

### 获取数据集

1. 打开数据集页面：<https://universe.roboflow.com/cont-vkss7/container-sq4mu/dataset/1>
2. 下载格式选择 YOLOv8。
3. 解压后确认包含 `train/`、`valid/`、`test/` 和 `data.yaml`。

如需制作自定义数据集，可使用 [Roboflow](https://roboflow.com) 或 [LabelImg](https://github.com/heartexlabs/labelImg) 标注图片，导出为 YOLO 格式后按上述目录结构组织。

## 快速开始

### 1. 克隆项目并安装依赖

```bash
git clone https://github.com/genanalucy/Container_YOLO.git
cd Container_YOLO
pip install ultralytics PyYAML
```

### 2. 配置数据集路径

根据实际数据集目录修改 `configs/data_kaggle.yaml` 或 `configs/data_server.yaml`，确保 `train:` 和 `val:` 指向 `images/` 目录。

示例：

```yaml
train: /path/to/container_dataset/train/images
val: /path/to/container_dataset/valid/images
test: /path/to/container_dataset/test/images

nc: 4
names: ['0', '1', '2', '3']
```

### 3. 检查数据集

训练前建议先运行数据集检查：

```bash
python scripts/check_dataset.py --data configs/data_kaggle.yaml
```

如果输出 `数据集检查通过`，可继续训练。若出现 `No images found` 或 `train path does not exist`，说明 `train:` 或 `val:` 路径需要修改为实际路径。

### 4. 运行 smoke training

Smoke training 用于快速确认代码、数据和 GPU 是否正常，不作为正式结果。

```bash
python scripts/train.py \
    --model yolo11n.pt \
    --data configs/data_kaggle.yaml \
    --epochs 1 \
    --batch 4 \
    --imgsz 640 \
    --device 0 \
    --name smoke_test
```

### 5. 运行 baseline 训练

```bash
python scripts/train.py \
    --model yolo11n.pt \
    --data configs/data_kaggle.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name baseline_yolo11n
```

显存不足时，可将 `--batch 16` 改为 `--batch 8` 或 `--batch 4`。

也可以使用 YOLO 命令行训练：

```bash
yolo detect train model=yolo11n.pt data=configs/data_kaggle.yaml epochs=50 batch=16
```

### 6. 推理预测

训练完成后，最佳权重通常位于：

```text
runs/baseline_yolo11n/weights/best.pt
```

检测单张图片：

```bash
python scripts/predict.py \
    --model runs/baseline_yolo11n/weights/best.pt \
    --source test_image.jpg
```

检测整个文件夹：

```bash
python scripts/predict.py \
    --model runs/baseline_yolo11n/weights/best.pt \
    --source test_images/
```

调整置信度阈值：

```bash
python scripts/predict.py \
    --model runs/baseline_yolo11n/weights/best.pt \
    --source test_image.jpg \
    --conf 0.5
```

预测结果会保存在 `runs/predict/` 目录下。

### 7. 查看训练指标

```bash
tail -n 5 runs/baseline_yolo11n/results.csv
```

`results.csv` 中常用列如下：

- `metrics/precision(B)`：精确率 P
- `metrics/recall(B)`：召回率 R
- `metrics/mAP50(B)`：mAP50
- `metrics/mAP50-95(B)`：mAP50-95

## Kaggle 运行流程

Kaggle 提供免费 GPU 资源，常见配置为单张 T4、P100 或 T4 x2。可选择以下流程完成训练。

### 1. 下载 Roboflow YOLO 数据集

1. 打开数据集页面：<https://universe.roboflow.com/cont-vkss7/container-sq4mu/dataset/1>
2. 点击下载，格式选择 YOLOv8。
3. 得到压缩包后确认其中包含 `train/`、`valid/`、`test/` 和 `data.yaml`。

### 2. 上传或添加数据集到 Kaggle

可选择以下运行路线：

1. 在 Kaggle 页面点击 Datasets，再点击 New Dataset，上传下载得到的压缩包。
2. 如果 Kaggle 已存在对应数据集，可在 Notebook 右侧点击 Add Input，搜索并添加数据集。

添加完成后，数据集通常位于 `/kaggle/input/` 下。

### 3. 新建 Kaggle Notebook 并开启 GPU

1. 点击 New Notebook。
2. 在右侧 Settings 中打开 Accelerator。
3. 选择 GPU。Kaggle 常见配置为 T4、P100 或 T4 x2。

### 4. 安装依赖并克隆项目

在 Kaggle Notebook 中依次运行以下 Cell：

```python
!pip install ultralytics PyYAML
```

```python
!git clone https://github.com/genanalucy/Container_YOLO.git
%cd Container_YOLO
```

### 5. 查找 data.yaml

先搜索 Kaggle 输入目录，不建议手动猜路径。

```python
!find /kaggle/input -name data.yaml -print
```

输出示例：

```text
/kaggle/input/container-dataset/data.yaml
```

记录该路径，后续配置会使用。

### 6. 新建 Kaggle 绝对路径配置

Roboflow 下载的 `data.yaml` 有时使用相对路径，例如：

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

在 Kaggle 中，这类相对路径可能失效。建议新建一个使用绝对路径的配置文件。先查看数据集目录：

```python
!find /kaggle/input -maxdepth 3 -type d | sort
```

然后将下面三行路径修改为实际数据集目录：

```python
%%writefile data_kaggle_fixed.yaml
train: /kaggle/input/实际数据集目录/train/images
val: /kaggle/input/实际数据集目录/valid/images
test: /kaggle/input/实际数据集目录/test/images

nc: 4
names: ['0', '1', '2', '3']
```

若数据集中没有 `test/images`，可以删除 `test:` 这一行。

### 7. 检查数据集

```python
!python scripts/check_dataset.py --data data_kaggle_fixed.yaml
```

如果检查通过，可继续训练。若出现 `No images found` 或 `train path does not exist`，返回上一步检查绝对路径。

### 8. 运行 smoke training

```python
!python scripts/train.py \
    --model yolo11n.pt \
    --data data_kaggle_fixed.yaml \
    --epochs 1 \
    --batch 4 \
    --imgsz 640 \
    --device 0 \
    --name smoke_test
```

### 9. 运行 baseline 训练

```python
!python scripts/train.py \
    --model yolo11n.pt \
    --data data_kaggle_fixed.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name baseline_yolo11n
```

如果 Kaggle 显存不足，将 `--batch 16` 改为 `--batch 8` 或 `--batch 4`。

### 10. 使用 best.pt 预测

```python
!python scripts/predict.py \
    --model runs/baseline_yolo11n/weights/best.pt \
    --source /kaggle/input/实际数据集目录/valid/images \
    --conf 0.25
```

预测图片会保存在 `runs/predict/` 下。

### 11. 查看指标并打包结果

```python
!ls runs/baseline_yolo11n
!tail -n 5 runs/baseline_yolo11n/results.csv
```

如需下载结果，可以打包 `runs/` 目录：

```python
import shutil
shutil.make_archive('training_results', 'zip', 'runs')
```

打包完成后，可在 Kaggle 右侧 Output 区域下载 `training_results.zip`。

Kaggle 注意事项：

- 免费版 GPU 使用时长有限。
- 单个 Session 有最长运行时间限制。
- 数据集上传有文件大小限制。
- 训练完成后建议及时下载结果，Session 结束后临时文件可能被清理。

## 服务器或本地 GPU 运行流程

该流程适用于服务器或本地 GPU 环境。所有路径均使用通用占位写法，运行时需要修改为实际路径。

### 1. 准备代码和环境

```bash
git clone https://github.com/genanalucy/Container_YOLO.git
cd Container_YOLO
pip install ultralytics PyYAML
```

### 2. 下载并解压数据集

从 Roboflow 下载 YOLOv8 格式后，解压到任意位置，例如：

```text
/path/to/container_dataset/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
└── data.yaml
```

### 3. 创建数据配置文件

```bash
cat > data_server_fixed.yaml <<'EOF'
train: /path/to/container_dataset/train/images
val: /path/to/container_dataset/valid/images
test: /path/to/container_dataset/test/images

nc: 4
names: ['0', '1', '2', '3']
EOF
```

若没有 `test/images`，删除 `test:` 那一行即可。

### 4. 检查数据集

```bash
python scripts/check_dataset.py --data data_server_fixed.yaml
```

### 5. 运行 smoke training

```bash
python scripts/train.py \
    --model yolo11n.pt \
    --data data_server_fixed.yaml \
    --epochs 1 \
    --batch 4 \
    --imgsz 640 \
    --device 0 \
    --name smoke_test
```

### 6. 运行 baseline 训练

```bash
python scripts/train.py \
    --model yolo11n.pt \
    --data data_server_fixed.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name baseline_yolo11n
```

### 7. 预测并查看结果

```bash
python scripts/predict.py \
    --model runs/baseline_yolo11n/weights/best.pt \
    --source /path/to/container_dataset/valid/images \
    --conf 0.25

tail -n 5 runs/baseline_yolo11n/results.csv
```

## 训练进度查看

### Kaggle 进度

Kaggle Notebook 会直接显示训练日志。进度输出示例：

```text
      4/100      13.2G      0.751      0.432      0.903         47        640:  52% 47/91 [00:15<00:14, 3.0it/s]
```

字段含义：

- `4/100`：当前为第 4 轮，总共训练 100 轮。
- `47/91`：当前 epoch 共 91 个 batch，已运行到第 47 个。
- `3.0it/s`：每秒约处理 3 个 batch。
- `<00:14`：当前 epoch 预计剩余约 14 秒。

### 使用 tmux 运行训练

服务器训练时间较长，推荐使用 `tmux`，避免 SSH 断开导致训练停止。

新建会话：

```bash
tmux new -s yolo_train
```

在 tmux 中启动训练并保存日志：

```bash
python scripts/train.py \
    --model yolo11n.pt \
    --data data_server_fixed.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name baseline_yolo11n 2>&1 | tee baseline_yolo11n.log
```

临时离开 tmux：先按 `Ctrl+b`，松开后再按 `d`。

重新进入训练窗口：

```bash
tmux attach -t yolo_train
```

只查看日志时运行：

```bash
tail -f baseline_yolo11n.log
```

## 多 GPU 训练

`--device` 控制使用的 GPU 编号：

- `--device 0`：使用第 0 张 GPU。
- `--device 0,1,2,3`：同时使用第 0、1、2、3 张 GPU。

使用四张 GPU 时，可将 batch 从 16 提高到 64。若显存不足，可改为 48、32 或 16。

四 GPU 训练 yolo11l 的命令如下：

```bash
python scripts/train.py \
    --model yolo11l.pt \
    --data /path/to/data_server_fixed.yaml \
    --epochs 100 \
    --batch 64 \
    --imgsz 640 \
    --device 0,1,2,3 \
    --name baseline_yolo11l_4gpu
```

训练完成后查看结果：

```bash
tail -n 5 runs/baseline_yolo11l_4gpu/results.csv
```

本次四 GPU 训练结果为：100 轮约 1.027 小时，P=0.982，R=0.980，mAP50=0.985，mAP50-95=0.777。

## 训练参数说明

| 参数 | 默认值 | 说明 |
|:---|:---:|:---|
| `--model` | `yolo11n.pt` | 预训练模型。从小到大包括 yolo11n、yolo11s、yolo11m、yolo11l、yolo11x。 |
| `--data` | 必填 | 数据集配置文件路径。 |
| `--epochs` | `50` | 训练轮数。 |
| `--imgsz` | `640` | 输入图像尺寸，常用值包括 320、416、640、1280。 |
| `--batch` | `16` | 每次处理的图片数量，显存不足时调小。 |
| `--device` | `0` | 训练设备，`0` 表示第一块 GPU，`cpu` 表示使用 CPU。 |
| `--patience` | `20` | 早停耐心值。 |
| `--project` | `runs` | 结果保存根目录。 |
| `--name` | `exp` | 实验名称，结果保存在 `project/name/` 下。 |

YOLO 模型大小参考：

| 模型 | 参数量 | 相对速度 | 相对精度 | 适用场景 |
|:---:|:---:|:---:|:---:|:---|
| yolo11n | 2.6M | 很快 | 基础 | 快速验证、轻量部署 |
| yolo11s | 9.4M | 快 | 较高 | 速度和精度平衡 |
| yolo11m | 20.1M | 中等 | 高 | 更高精度实验 |
| yolo11l | 25.3M | 较慢 | 更高 | 高精度需求 |
| yolo11x | 56.9M | 慢 | 最高 | 大显存环境 |

## 指标解释

### 精确率 Precision

```text
P = 正确检测数 / 总检测数
```

P = 0.98 表示模型输出的 100 个检测框中，大约 98 个是真实目标。

### 召回率 Recall

```text
R = 正确检测数 / 实际目标数
```

R = 0.975 表示实际存在的 100 个目标中，大约 97.5 个被模型检出。

### mAP50

mAP50 是 IoU 阈值为 0.5 时的平均精度。IoU 表示预测框和真实框的重叠程度，IoU 越高，目标框位置越准确。

### mAP50-95

mAP50-95 会在 IoU 0.5 到 0.95 的多个阈值下计算 mAP 后取平均，因此比 mAP50 更严格。该指标较低时，通常需要检查标注框质量、图像分辨率和小目标表现。

## 常见问题 FAQ

### Q1: `FileNotFoundError: data.yaml not found`

原因：数据集配置文件路径不正确，或文件不存在。

处理方法：

```bash
# 1. 确认配置文件存在
ls configs/data_kaggle.yaml

# 2. 使用绝对路径
python scripts/train.py --data /完整/路径/configs/data_kaggle.yaml

# 3. 检查当前工作目录
pwd  # 应位于 Container_YOLO/ 目录下
```

### Q2: `No images found` 或 `train path does not exist`

原因：`data.yaml` 中的 `train:` 或 `val:` 路径不存在，或未指向 `images/` 目录。

处理方法：

```bash
# 1. 检查 data.yaml 中的路径
cat configs/data_kaggle.yaml

# 2. 确认图片目录存在且包含图片
ls /kaggle/input/实际数据集路径/train/images/ | head -5

# 3. 修改 train 和 val 路径，路径应指向 images/ 目录
```

### Q3: `class index out of range`

原因：标签文件中的类别 ID 超过 `nc - 1`。例如 `nc=4` 时，合法类别 ID 为 0、1、2、3。

处理方法：

```bash
python scripts/check_dataset.py --data configs/data_kaggle.yaml
```

根据检查结果修复对应标签文件中的类别 ID。

### Q4: `CUDA out of memory`

原因：batch 太大、图片尺寸太大，或模型过大。

处理方法：

```bash
# 减小 batch size
python scripts/train.py --data ... --batch 8
python scripts/train.py --data ... --batch 4

# 减小图片尺寸
python scripts/train.py --data ... --imgsz 416

# 使用更小的模型
python scripts/train.py --model yolo11n.pt ...
```

### Q5: `No labels found` 或标签目录不存在

原因：YOLO 会从 `images/` 同级目录查找 `labels/`，目录缺失或结构不符合要求时会报错。

处理方法：

```bash
# 检查目录结构
ls /path/to/dataset/train/
# 应看到: images/  labels/

# 确认标签文件格式
ls /path/to/dataset/train/labels/ | head -5
```

标签文件应为 `.txt` 格式，不能是 `.xml` 或 `.json`。

### Q6: `No detections` 或检测结果为空

原因：模型未检测到满足置信度阈值的目标。

处理方法：

```bash
# 降低置信度阈值
python scripts/predict.py --model best.pt --source test.jpg --conf 0.1

# 检查训练结果
ls runs/exp/results.csv
```

如果训练 mAP 很低，需要增加训练轮数、检查数据集质量或补充样本。

### Q7: 下载预训练模型失败或网络错误

原因：首次运行时 YOLO 会自动下载预训练权重，网络访问失败会导致下载中断。

处理方法：

```bash
# 方法1: 手动下载模型权重
# 访问 https://github.com/ultralytics/assets/releases
# 下载 yolo11n.pt，放到项目根目录或 ~/.config/Ultralytics/ 下

# 方法2: 使用代理
export https_proxy=http://your-proxy:port
python scripts/train.py --data ...

# 方法3: 直接指定本地权重路径
python scripts/train.py --model ./yolo11n.pt --data ...
```

## 优化建议

### 使用更大的模型

基线模型为 yolo11n，速度快，适合验证流程。追求更高精度时，可尝试 yolo11s、yolo11m、yolo11l 或 yolo11x。

```bash
# yolo11l
python scripts/train.py --model yolo11l.pt --data ... --epochs 100

# yolo11x，需要更大显存
python scripts/train.py --model yolo11x.pt --data ... --epochs 100 --batch 4
```

| 升级路线 | 命令 | 预期效果 |
|:---:|:---|:---|
| n 到 s | `--model yolo11s.pt` | mAP50-95 可能提升 |
| s 到 m | `--model yolo11m.pt` | 定位和分类能力进一步提升 |
| m 到 l | `--model yolo11l.pt` | 高精度场景更合适 |
| l 到 x | `--model yolo11x.pt` | 精度上限更高，但速度较慢 |

### 增加训练轮数

数据量较小时，可适当增加训练轮数：

```bash
python scripts/train.py --model yolo11s.pt --data ... --epochs 200 --patience 50
```

### 提高图像分辨率

更大的输入尺寸有助于小目标检测，但会增加显存占用和训练时间。

```bash
python scripts/train.py --model yolo11s.pt --data ... --imgsz 1280 --batch 4
```

### 提高标注质量

标注质量会直接影响 mAP50-95。建议检查以下内容：

- 是否存在漏标或错标。
- 目标框是否贴合目标边缘。
- 类别 ID 是否在合法范围内。
- 同类目标的标注标准是否一致。

### 增加难例样本

可补充模型容易出错的场景，例如：

- 遮挡严重的图片。
- 远距离或小目标图片。
- 光照不足或过曝图片。
- 俯视、侧视等特殊角度图片。

### 做错误分析

训练结束后，建议查看以下文件：

- `confusion_matrix.png`
- `results.csv`
- `runs/predict/` 中的预测图片
- 漏检或误检样本

先确认错误类型，再决定补数据、改参数或更换模型。

### 尝试 yolo11x

yolo11x 更大、更慢，也需要更多显存。建议在数据质量和标注质量确认后，再使用该模型做最终实验。

## 许可证与致谢

本项目代码用于学习、实验和项目复现。数据集遵循 Roboflow Universe 对应数据集的 CC BY 4.0 许可证。

感谢以下项目和平台：

- [Ultralytics](https://github.com/ultralytics/ultralytics)，提供 YOLO11 目标检测框架。
- [Roboflow Universe](https://universe.roboflow.com/)，提供集装箱检测数据集。
