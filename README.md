# 🚢 集装箱检测：YOLO11 目标检测项目

> 📚 **面向本科生的 YOLO 入门实战项目**：从数据准备到模型训练到推理预测，手把手教你完成一个完整的目标检测项目。

---

## 📖 项目简介

本项目使用 **YOLO11**（You Only Look Once）实现集装箱（Container）的自动检测与分类。目标检测是计算机视觉中的核心任务之一，它的目标是：

> 给定一张图片，自动找出图片中所有感兴趣的目标，并用矩形框标注它们的位置和类别。

**本项目检测 4 种集装箱类型**，使用的数据集来自 [Roboflow Universe](https://universe.roboflow.com/cont-vkss7/container-sq4mu/dataset/1)（CC BY 4.0 许可）。

---

## 🏆 效果展示

### 本次实验结果对比

| 实验 | 模型 | GPU 用法 | epochs | imgsz | batch | P | R | mAP50 | mAP50-95 | 用时 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 基线实验 | yolo11n | 单 GPU | 50 | 640 | 16 | 0.980 | 0.975 | 0.984 | 0.759 | 约 39 分钟 |
| 大模型实验 | yolo11l | 4 GPU | 100 | 640 | 64 | 0.982 | 0.980 | 0.985 | 0.777 | 约 1.027 小时 |

### 指标怎么理解

| 指标 | yolo11n | yolo11l 4GPU | 说明 |
|:---:|:---:|:---:|:---|
| **精确率 (P)** | 0.980 | 0.982 | 检测出的目标中有多少是真的 |
| **召回率 (R)** | 0.975 | 0.980 | 实际存在的目标中有多少被找到了 |
| **mAP50** | 0.984 | 0.985 | IoU=0.5 时的平均精度，已经很高 |
| **mAP50-95** | 0.759 | 0.777 | 更严格的位置精度指标，大模型有小幅提升 |

> 💡 **这意味着什么？** yolo11l 确实比 yolo11n 更好，但提升并不夸张。mAP50 已经接近数据集上限，mAP50-95 的瓶颈更可能来自标注质量、图片分辨率、目标框是否贴合、数据集本身难度，而不只是模型大小。

---

## 🧑‍🎓 学生怎么复刻我的操作

这一节给学生直接照着跑。你可以选一条路线：

- **路线 A：Kaggle 路线**，适合学生，没有本地 GPU 也能跑。Kaggle 常见是单张 T4 或 P100，也可能给 T4 x2。
- **路线 B：服务器或本地 GPU 路线**，适合老师、助教、实验室机器，路径要换成自己的数据集路径。

### 路线 A：Kaggle 从零复刻流程

#### 1. 下载 Roboflow YOLO 数据集

1. 打开数据集页面：<https://universe.roboflow.com/cont-vkss7/container-sq4mu/dataset/1>
2. 点击下载，格式选择 **YOLOv8**。
3. 得到一个压缩包，里面通常会有 `train/`、`valid/`、`test/` 和 `data.yaml`。

#### 2. 上传或添加数据集到 Kaggle

有两种做法，任选一种：

1. 在 Kaggle 页面点击 **Datasets**，再点 **New Dataset**，上传刚才下载的压缩包。
2. 如果已经有人上传过这个数据集，也可以在 Notebook 右侧点击 **Add Input**，搜索并添加数据集。

添加完成后，数据集会出现在 `/kaggle/input/` 下面。

#### 3. 新建 Kaggle Notebook 并打开 GPU

1. 点击 **New Notebook**。
2. 右侧 **Settings** 里打开 **Accelerator**。
3. 选择 GPU。学生一般用单张 T4 或 P100 就够跑 baseline。

#### 4. 安装依赖并克隆项目

在 Kaggle Notebook 里依次运行下面的 Cell：

```python
!pip install ultralytics PyYAML
```

```python
!git clone https://github.com/genanalucy/Container_YOLO.git
%cd Container_YOLO
```

#### 5. 找到 Kaggle 里的 data.yaml

先不要猜路径，直接搜索：

```python
!find /kaggle/input -name data.yaml -print
```

你会看到类似输出：

```text
/kaggle/input/container-dataset/data.yaml
```

把这个路径记下来，下面会用到。

#### 6. 如果原始 data.yaml 的相对路径失效，就新建 data_kaggle_fixed.yaml

Roboflow 下载的 `data.yaml` 有时写的是相对路径，例如：

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

在 Kaggle 里，这种相对路径容易失效。最稳妥的做法是新建一个使用绝对路径的配置文件。先用下面命令看清楚数据集目录：

```python
!find /kaggle/input -maxdepth 3 -type d | sort
```

然后按你的真实目录改下面三行路径：

```python
%%writefile data_kaggle_fixed.yaml
train: /kaggle/input/你的数据集目录/train/images
val: /kaggle/input/你的数据集目录/valid/images
test: /kaggle/input/你的数据集目录/test/images

nc: 4
names: ['0', '1', '2', '3']
```

如果你的数据集中没有 `test/images`，可以删掉 `test:` 那一行，不影响训练。

#### 7. 训练前检查数据集

```python
!python scripts/check_dataset.py --data data_kaggle_fixed.yaml
```

如果看到数据集检查通过，就继续。要是出现 `No images found` 或 `train path does not exist`，说明 `train:` 或 `val:` 路径写错了，回到上一步重新改绝对路径。

#### 8. 先跑 1 轮 smoke training

Smoke training 的作用是快速确认代码、数据、GPU 都能正常工作。它不是正式结果。

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

#### 9. 跑 baseline 正式训练

学生复刻建议先跑这个版本，速度快，结果稳定：

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

如果 Kaggle 显存不足，把 `--batch 16` 改成 `--batch 8` 或 `--batch 4`。

#### 10. 用 best.pt 做预测

训练完成后，最佳权重一般在：

```text
runs/baseline_yolo11n/weights/best.pt
```

对验证集图片预测：

```python
!python scripts/predict.py \
    --model runs/baseline_yolo11n/weights/best.pt \
    --source /kaggle/input/你的数据集目录/valid/images \
    --conf 0.25
```

预测图片会保存在 `runs/predict/` 下面。

#### 11. 查看 results.csv 和指标

```python
!ls runs/baseline_yolo11n
!tail -n 5 runs/baseline_yolo11n/results.csv
```

`results.csv` 里重点看这些列：

- `metrics/precision(B)`：精确率 P。
- `metrics/recall(B)`：召回率 R。
- `metrics/mAP50(B)`：mAP50。
- `metrics/mAP50-95(B)`：mAP50-95。

如果你想下载结果，可以把 `runs/` 打包：

```python
import shutil
shutil.make_archive('training_results', 'zip', 'runs')
```

然后在 Kaggle 右侧 Output 区域下载 `training_results.zip`。

### 路线 B：服务器或本地 GPU 复刻流程

这条路线适合老师、助教或有 GPU 的同学。下面所有路径都用通用写法，请换成你自己的目录，不要直接复制成真实服务器路径。

#### 1. 准备代码和环境

```bash
git clone https://github.com/genanalucy/Container_YOLO.git
cd Container_YOLO
pip install ultralytics PyYAML
```

#### 2. 下载并解压 Roboflow YOLO 数据集

从 Roboflow 下载 YOLOv8 格式后，解压到任意位置，例如：

```text
/path/to/container_dataset/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
└── data.yaml
```

#### 3. 写一个服务器用的数据配置

```bash
cat > data_server_fixed.yaml <<'EOF'
train: /path/to/container_dataset/train/images
val: /path/to/container_dataset/valid/images
test: /path/to/container_dataset/test/images

nc: 4
names: ['0', '1', '2', '3']
EOF
```

如果没有 `test/images`，删掉 `test:` 那一行即可。

#### 4. 检查数据集

```bash
python scripts/check_dataset.py --data data_server_fixed.yaml
```

#### 5. 先跑 smoke training

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

#### 6. 跑 baseline

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

#### 7. 用 best.pt 预测并检查结果

```bash
python scripts/predict.py \
    --model runs/baseline_yolo11n/weights/best.pt \
    --source /path/to/container_dataset/valid/images \
    --conf 0.25

tail -n 5 runs/baseline_yolo11n/results.csv
```

---

## 👀 怎么看训练进度

### Kaggle 上怎么看

Kaggle Notebook 会直接显示训练日志。训练时你会看到类似这样的进度：

```text
      4/100      13.2G      0.751      0.432      0.903         47        640:  52% 47/91 [00:15<00:14, 3.0it/s]
```

这行可以这样读：

- `4/100`：现在是第 4 轮，一共要训练 100 轮。
- `47/91`：当前这一轮有 91 个 batch，现在跑到第 47 个。
- `3.0it/s`：每秒大约跑 3 个 batch，越高越快。
- `<14.9s` 或 `<00:14`：当前这一轮预计还要 14 秒左右。

### 服务器上用 tmux 看进度

服务器训练时间比较长，推荐用 `tmux`，这样断开 SSH 后训练不会停。

新建会话：

```bash
tmux new -s yolo_train
```

在 tmux 里启动训练，并把日志保存下来：

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

如果你只想看日志，不进入 tmux：

```bash
tail -f baseline_yolo11n.log
```

---

## 🧩 怎么用四张 GPU

`--device` 控制用哪些 GPU：

- `--device 0`：只用第 0 张 GPU，也就是单 GPU。
- `--device 0,1,2,3`：同时用第 0、1、2、3 张 GPU，也就是四张 GPU。

用四张 GPU 时，建议把 batch 从 16 提高到 64。直观理解是：单张卡每次吃 16 张图，四张卡合起来就可以尝试每次吃 64 张图。显存不够时再改成 48、32 或 16。

四 GPU 训练 yolo11l 的通用命令如下：

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

训练完成后看结果：

```bash
tail -n 5 runs/baseline_yolo11l_4gpu/results.csv
```

本次四 GPU 训练结果是：100 轮约 1.027 小时，P=0.982，R=0.980，mAP50=0.985，mAP50-95=0.777。它比 yolo11n 有提升，但提升幅度不大，说明当前数据集可能已经接近模型能轻松达到的上限。

---

## 🧪 下一步怎么优化但先不跑

下面这些是后续实验方向，先记录，不建议一开始就全部跑。先把 baseline 跑通，再逐个验证。

1. **把 `imgsz` 提高到 1280**  
   更大的输入分辨率能让模型看到更多细节，可能提升 mAP50-95。代价是显存和训练时间明显增加。

2. **检查标签质量**  
   重点看框是否贴合目标边缘，有没有漏标、错标、类别混乱。mAP50-95 对框的位置很敏感，标注不准会直接压低这个指标。

3. **补充难例样本**  
   收集模型容易错的图片，例如遮挡、远距离、小目标、强光、阴影、特殊角度。难例通常比盲目换大模型更有价值。

4. **做混淆和错误分析**  
   查看 `confusion_matrix.png`、预测图片和漏检图片，记录模型到底错在哪里。先分析错误，再决定补数据还是改参数。

5. **之后再尝试 yolo11x**  
   yolo11x 更大、更慢，也更吃显存。建议等标签和数据问题先处理完，再用它做最后的冲刺实验。

---

## 🛠️ 环境准备

### 1. 安装 Python

确保你的 Python 版本 **≥ 3.8**：

```bash
python --version
# Python 3.8.x 或更高版本即可
```

### 2. 安装 Ultralytics

Ultralytics 是 YOLO 的官方 Python 库，包含了训练、推理等所有功能：

```bash
pip install ultralytics
```

### 3. 验证安装

```bash
yolo version
# 输出类似: ultralytics 8.x.x
```

也可以在 Python 中验证：

```python
from ultralytics import YOLO
print("Ultralytics 安装成功！")
```

> ⚠️ 如果 `pip install` 很慢，可以使用国内镜像源：
> ```bash
> pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 📂 数据集准备

### YOLO 数据集格式

YOLO 要求数据集按以下目录结构组织：

```
dataset/
├── train/
│   ├── images/          # 训练集图片
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/          # 训练集标签（与图片一一对应）
│       ├── img001.txt   # 每张图片对应一个 .txt 标签文件
│       ├── img002.txt
│       └── ...
├── valid/
│   ├── images/          # 验证集图片
│   └── labels/          # 验证集标签
└── data.yaml            # 数据集配置文件
```

### 标签文件格式（.txt）

每行代表一个目标，格式为：

```
类别ID  中心点X  中心点Y  宽度  高度
```

所有坐标值都是**归一化**的（0 到 1 之间），表示相对于图片尺寸的比例。

示例：
```
0 0.451563 0.641667 0.156250 0.433333
2 0.710938 0.508333 0.120313 0.383333
```

> 解释：第一行表示类别 0 的集装箱，中心在图片 (45.2%, 64.2%) 处，宽度占 15.6%，高度占 43.3%。

### data.yaml 配置文件

```yaml
# 数据集路径（相对于 data.yaml 所在目录，或使用绝对路径）
train: /path/to/dataset/train/images
val: /path/to/dataset/valid/images

# 类别信息
nc: 4                          # 类别数量
names: ['0', '1', '2', '3']   # 类别名称（对应 4 种集装箱类型）
```

### 如何获取数据集

本项目数据集来自 **Roboflow Universe**，你可以通过以下方式获取：

1. 访问数据集页面：[container-sq4mu](https://universe.roboflow.com/cont-vkss7/container-sq4mu/dataset/1)
2. 选择 **YOLOv8** 格式下载
3. 解压后即可使用

如果你想制作自己的数据集：
1. 收集图片并使用标注工具（推荐 [Roboflow](https://roboflow.com) 或 [LabelImg](https://github.com/heartexlabs/labelImg)）标注
2. 导出为 YOLO 格式
3. 按上面的目录结构组织

---

## 🚀 快速开始

### Step 1：安装依赖

```bash
# 安装 ultralytics（包含 YOLO11）和 PyYAML
pip install ultralytics PyYAML

# 如果下载慢，使用清华镜像源
pip install ultralytics PyYAML -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step 2：准备数据集

下载并解压数据集，确认目录结构正确。

**健康检查**（推荐！训练前先检查数据集是否有问题）：

```bash
python scripts/check_dataset.py --data configs/data_kaggle.yaml
```

如果看到 `✅ 数据集检查通过`，就可以开始训练了。

> 📝 记得根据你的环境修改 `configs/data_kaggle.yaml` 或 `configs/data_server.yaml` 中的数据集路径。

### Step 3：快速验证（Smoke Test）

在正式训练前，建议先跑 1-2 轮确认一切正常：

```bash
# Smoke test：只训练 1 轮，确认数据加载、模型训练没问题
python scripts/train.py \
    --data configs/data_kaggle.yaml \
    --epochs 1 \
    --batch 4 \
    --name smoke_test
```

如果没有报错，就可以正式训练了。如果报错，先根据错误提示排查（参考下方 FAQ）。

### Step 4：正式训练（Baseline）

```bash
# 基线训练：yolo11n，50 轮（约 39 分钟 / RTX 3090）
python scripts/train.py \
    --data configs/data_kaggle.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --name baseline_exp

# 自定义参数训练（如使用更大的模型）
python scripts/train.py \
    --model yolo11s.pt \
    --data configs/data_kaggle.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0

# 也可以直接用 yolo 命令行
yolo detect train model=yolo11n.pt data=configs/data_kaggle.yaml epochs=50 batch=16
```

训练过程中，终端会实时显示损失值和指标：

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances     Size
  1/50      5.2G     1.2345     0.8765     1.1234         45      640
  2/50      5.2G     1.0234     0.7654     0.9876         52      640
  ...
```

### Step 5：推理预测

训练完成后，使用最佳权重进行预测：

```bash
# 检测单张图片
python scripts/predict.py \
    --model runs/exp/weights/best.pt \
    --source test_image.jpg

# 检测整个文件夹
python scripts/predict.py \
    --model runs/exp/weights/best.pt \
    --source test_images/

# 调整置信度阈值
python scripts/predict.py \
    --model runs/exp/weights/best.pt \
    --source test_image.jpg \
    --conf 0.5
```

预测结果（带标注框的图片）会保存在 `runs/predict/` 目录下。

---

## 📊 训练参数说明

| 参数 | 默认值 | 说明 |
|:---|:---:|:---|
| `--model` | `yolo11n.pt` | 预训练模型。从大到小: yolo11n → s → m → l → x。越大的模型越准但越慢。 |
| `--data` | （必填） | 数据集配置文件路径（data.yaml） |
| `--epochs` | `50` | 训练轮数。数据量大时 30-50 轮可能就够了，数据量小时可以设 100-200。 |
| `--imgsz` | `640` | 输入图像尺寸。越大越准但越慢。常用: 320, 416, 640, 1280。 |
| `--batch` | `16` | 每次处理的图片数量。显存不够就调小（如 8 或 4），显存够可以调大。 |
| `--device` | `0` | 训练设备。`0` 表示第一块 GPU，`cpu` 表示用 CPU（很慢）。 |
| `--patience` | `20` | 早停耐心值。连续多少轮指标没有提升就自动停止训练，防止过拟合。 |
| `--project` | `runs` | 结果保存的根目录 |
| `--name` | `exp` | 本次实验名称。结果保存在 `project/name/` 下 |

### YOLO 模型大小对比

| 模型 | 参数量 | 相对速度 | 相对精度 | 适用场景 |
|:---:|:---:|:---:|:---:|:---|
| yolo11n | 2.6M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 快速验证、嵌入式设备 |
| yolo11s | 9.4M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 平衡速度和精度 |
| yolo11m | 20.1M | ⭐⭐⭐ | ⭐⭐⭐⭐ | 追求更高精度 |
| yolo11l | 25.3M | ⭐⭐ | ⭐⭐⭐⭐⭐ | 高精度需求 |
| yolo11x | 56.9M | ⭐ | ⭐⭐⭐⭐⭐+ | 最高精度，需要大显存 |

---

## 📈 指标解释

### 精确率 (Precision, P)

> **"我检测出来的结果中，有多少是对的？"**

$$P = \frac{正确检测数}{总检测数}$$

- P = 0.98 表示：模型报告了 100 个检测框，其中 98 个是真正的集装箱。
- 越高越好。

### 召回率 (Recall, R)

> **"实际上有多少目标被我找到了？"**

$$R = \frac{正确检测数}{实际目标数}$$

- R = 0.975 表示：图片中实际有 100 个集装箱，模型找到了 97.5 个。
- 越高越好。

### mAP50（Mean Average Precision @ IoU=0.5）

> **"在'重叠度≥50%'的标准下，模型整体表现如何？"**

- IoU（Intersection over Union）衡量预测框和真实框的重叠程度。
- IoU ≥ 0.5 意味着预测框和真实框至少有 50% 的重叠，就算检测正确。
- mAP50 = 0.984 是非常好的结果。

### mAP50-95（Mean Average Precision @ IoU=0.5:0.95）

> **"在不同严格程度下（IoU 从 0.5 到 0.95），模型的综合表现如何？"**

- 这是一个更严格的指标，在 10 个不同的 IoU 阈值下（0.5, 0.55, 0.60, ..., 0.95）计算 mAP 然后取平均。
- mAP50-95 = 0.759 表示模型在定位精度上还有提升空间（预测框和真实框的位置还不够精确）。
- 这个值通常比 mAP50 低很多，是正常的。

### 简单理解

```
精确率高、召回率高 → 模型很厉害！
mAP50 高            → 模型能找到大多数目标
mAP50-95 高         → 模型不仅找到了，而且框的位置很准
```

---

## ☁️ Kaggle 使用指南

Kaggle 提供免费的 GPU 资源，非常适合学生在没有 GPU 的电脑上训练模型。

### 步骤

1. **注册 Kaggle 账号**：访问 [kaggle.com](https://www.kaggle.com)

2. **上传数据集**：
   - 在 Kaggle → Datasets → New Dataset
   - 上传你下载的集装箱数据集 ZIP 文件
   - 记下数据集名称（如 `your-username/container-dataset`）

3. **创建 Notebook**：
   - New Notebook → Settings → Accelerator → **GPU T4 x2**（免费）

4. **在 Notebook 中运行**：

```python
# === Cell 1: 安装依赖 ===
!pip install ultralytics PyYAML

# === Cell 2: 克隆项目代码 ===
!git clone https://github.com/genanalucy/Container_YOLO.git
%cd Container_YOLO

# === Cell 3: 查看数据集路径 ===
# 把下面的 your-dataset-name 替换为你上传的数据集名称
!ls /kaggle/input/your-dataset-name/

# === Cell 4: 修改 data.yaml 路径 ===
import yaml

# 用你实际的数据集路径替换
data_config = {
    'train': '/kaggle/input/your-dataset-name/train/images',
    'val': '/kaggle/input/your-dataset-name/valid/images',
    'nc': 4,
    'names': ['0', '1', '2', '3']
}

with open('configs/data_kaggle.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("data.yaml 已更新！")

# === Cell 5: 检查数据集 ===
!python scripts/check_dataset.py --data configs/data_kaggle.yaml

# === Cell 6: 开始训练 ===
!python scripts/train.py \
    --model yolo11n.pt \
    --data configs/data_kaggle.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name kaggle_exp

# === Cell 7: 查看训练结果 ===
!ls runs/kaggle_exp/

# === Cell 8: 运行预测 ===
!python scripts/predict.py \
    --model runs/kaggle_exp/weights/best.pt \
    --source /kaggle/input/your-dataset-name/valid/images \
    --conf 0.25

# === Cell 9: 下载结果 ===
# 在 Kaggle Notebook 中，点击右侧 Output 面板可以下载 runs/ 目录
import shutil
shutil.make_archive('training_results', 'zip', 'runs')
print("结果已打包为 training_results.zip，在 Output 中下载")
```

> ⚠️ **Kaggle 注意事项**：
> - 免费版每周有 **30 小时** GPU 使用时长
> - 每个 Session 最长 **12 小时**
> - 数据集上传单文件不能超过 **20GB**
> - 训练完成后记得及时下载结果，Session 过期后数据会清除

---

## ❓ 常见问题 FAQ

### Q1: `FileNotFoundError: data.yaml not found`

**原因**：数据集配置文件路径不对，或者文件不存在。

**解决方法**：
```bash
# 1. 确认配置文件确实存在
ls configs/data_kaggle.yaml

# 2. 使用绝对路径确保能找到
python scripts/train.py --data /完整/路径/configs/data_kaggle.yaml

# 3. 检查当前工作目录是否正确
pwd  # 应该在 Container_YOLO/ 目录下
```

### Q2: `No images found` / `train path does not exist`

**原因**：data.yaml 中的 train/val 路径指向的图片目录不存在或路径写错了。

**解决方法**：
```bash
# 1. 检查 data.yaml 中的路径是否正确
cat configs/data_kaggle.yaml

# 2. 确认图片目录存在且有图片
ls /kaggle/input/你的数据集路径/train/images/ | head -5

# 3. 如果路径不对，修改 data.yaml 中的 train 和 val 路径
# 注意：路径要指向 images/ 目录，不是 labels/ 目录
```

### Q3: `class index out of range` / 类别 ID 超出范围

**原因**：标签文件中的类别 ID 超过了 `nc - 1`（比如 nc=4 但出现了类别 ID=5）。

**解决方法**：
```bash
# 1. 用 check_dataset.py 检查标签
python scripts/check_dataset.py --data configs/data_kaggle.yaml

# 2. 查看哪些标签文件有问题
# check_dataset.py 会列出类别 ID 超出范围的文件名和行号

# 3. 手动修复有问题的标签文件（把错误的类别 ID 改为正确的值）
```

> ⚠️ 类别 ID 从 **0** 开始，如果 nc=4，合法的类别 ID 是 0、1、2、3。

### Q4: `CUDA out of memory`（显存不足）

**原因**：batch 太大或图片尺寸太大，GPU 显存放不下。

**解决方法**：
```bash
# 方法1: 减小 batch size
python scripts/train.py --data ... --batch 8    # 从 16 改为 8
python scripts/train.py --data ... --batch 4    # 还不行就改为 4

# 方法2: 减小图片尺寸
python scripts/train.py --data ... --imgsz 416  # 从 640 改为 416

# 方法3: 使用更小的模型
python scripts/train.py --model yolo11n.pt ...  # 用 n 而不是 l
```

### Q5: `No labels found` / 标签目录不存在

**原因**：YOLO 期望标签文件在 images 同级的 `labels/` 目录下，但找不到。

**解决方法**：
```bash
# 1. 检查目录结构，确认 labels/ 和 images/ 是同级目录
ls /path/to/dataset/train/
# 应该看到: images/  labels/

# 2. 如果标签文件在别的位置，需要移动或创建符号链接
# 例如，假设标签在 train/labels/ 但数据集路径指向 train/images/
# YOLO 会自动把 images/ 替换为 labels/ 来查找标签

# 3. 确认标签文件是 .txt 格式，不是 .xml 或 .json
ls /path/to/dataset/train/labels/ | head -5
```

### Q6: `No detections` / 检测结果为空

**原因**：模型没有检测到任何目标。

**排查步骤**：
```bash
# 1. 尝试降低置信度阈值
python scripts/predict.py --model best.pt --source test.jpg --conf 0.1

# 2. 检查模型是否训练充分（训练了多少轮？mAP 是多少？）
ls runs/exp/results.csv

# 3. 确认测试图片中确实有集装箱目标

# 4. 如果训练 mAP 很低，说明模型还没学好，需要多训练几轮
```

### Q7: 下载预训练模型失败 / 网络错误

**原因**：第一次运行时 YOLO 会自动从 GitHub 下载预训练权重，国内网络可能无法访问。

**解决方法**：
```bash
# 方法1: 手动下载模型权重
# 访问 https://github.com/ultralytics/assets/releases
# 下载 yolo11n.pt，放到项目根目录或 ~/.config/Ultralytics/ 下

# 方法2: 使用镜像或代理
export https_proxy=http://your-proxy:port
python scripts/train.py --data ...

# 方法3: 如果已经有权重文件，直接指定路径
python scripts/train.py --model ./yolo11n.pt --data ...
```

---

## 💡 优化建议

如果基线模型效果不够好，可以尝试以下方法：

### 📦 使用更大的模型

基线用的是最小的 yolo11n（2.59M 参数），升级到更大的模型通常能显著提升精度：

```bash
# yolo11l：大模型，精度更高
python scripts/train.py --model yolo11l.pt --data ... --epochs 100

# yolo11x：超大模型，最高精度（需要大显存 ≥16GB）
python scripts/train.py --model yolo11x.pt --data ... --epochs 100 --batch 4
```

| 升级路线 | 命令 | 预期效果 |
|:---:|:---|:---|
| n → s | `--model yolo11s.pt` | mAP50-95 提升约 2-5% |
| s → m | `--model yolo11m.pt` | mAP50-95 继续提升 |
| m → l | `--model yolo11l.pt` | mAP50-95 显著提升 |
| l → x | `--model yolo11x.pt` | 最高精度，但非常慢 |

### ⏱️ 增加训练轮数

数据量小时，增加训练轮数可以让模型学得更充分：

```bash
# 增加到 100-200 轮
python scripts/train.py --model yolo11s.pt --data ... --epochs 200 --patience 50
```

### 🔍 提高图像分辨率

更大的图像尺寸能让模型看到更多细节，特别是小目标：

```bash
# 使用 1280 分辨率（显存需求翻倍，注意调小 batch）
python scripts/train.py --model yolo11s.pt --data ... --imgsz 1280 --batch 4
```

### 🏷️ 提高标注质量

标注质量直接影响模型效果：
- 检查是否有遗漏标注（有目标但没画框）
- 确保框紧贴目标边缘，不要太大或太小
- 用 `check_dataset.py` 检查是否有异常标签
- 确保同类目标的标注标准一致

### 🎯 增加难例样本（Hard Examples）

模型在哪些场景下表现不好，就多加一些那种场景的图片：
- 遮挡严重的场景
- 远距离/小目标的场景
- 光照不足或过曝的场景
- 不同角度（俯视、侧视）的场景

> 💡 这是最有效的优化方法！增加难例数据往往比换更大模型效果更好。

---

## 📁 项目结构

```
Container_YOLO/
├── README.md                      # 📖 项目说明文档（你正在看的这个文件）
├── requirements.txt               # 📦 Python 依赖
├── .gitignore                     # 🚫 Git 忽略规则
├── configs/                       # ⚙️ 配置文件
│   ├── data_server.yaml           # 服务器训练用数据集配置（绝对路径）
│   └── data_kaggle.yaml           # Kaggle 训练用数据集配置（需修改路径）
└── scripts/                       # 📜 脚本文件
    ├── train.py                   # 🚂 训练脚本
    ├── predict.py                 # 🔍 推理预测脚本
    └── check_dataset.py           # ✅ 数据集健康检查工具
```

### 训练后生成的文件（不上传到 Git）

```
runs/
└── exp/                           # 训练输出目录
    ├── weights/
    │   ├── best.pt                # 🏅 最佳模型权重（按 mAP 排序）
    │   └── last.pt                # 📌 最后一轮的权重
    ├── results.png                # 📈 训练曲线图
    ├── confusion_matrix.png       # 📊 混淆矩阵
    ├── results.csv                # 📋 训练数据
    └── ...                        # 其他训练产出
```

---

## 🙏 致谢

- **[Ultralytics](https://github.com/ultralytics/ultralytics)**：提供 YOLO11 目标检测框架
- **[Roboflow Universe](https://universe.roboflow.com/)**：提供集装箱检测数据集（[container-sq4mu](https://universe.roboflow.com/cont-vkss7/container-sq4mu/dataset/1)，CC BY 4.0 许可）
- 感谢所有为开源社区贡献的开发者们 ❤️

---

## 📄 许可证

本项目的代码仅供学习和教学使用。数据集遵循 Roboflow 的 CC BY 4.0 许可证。

---

> 🎓 **祝你在 YOLO 的学习之旅中收获满满！遇到问题不要慌，多看报错信息，善用搜索引擎。**
