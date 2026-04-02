# MNIST 手写数字识别项目

这是一个基于 PyTorch 的手写数字识别练习项目，围绕 `MNIST` 数据集完成了从模型训练、模型保存，到本地手写画板识别与样本回收的完整闭环。

项目当前已经包含：

- `MNIST` 数据集本地缓存
- 训练完成的模型权重 `runs_mnist/best_model.pt`
- 两个可直接运行的手写识别应用
- 若干手工采集样本 `my_doodles/`

## 项目功能

- 使用 `torch` + `torchvision` 训练 MNIST 分类模型
- 提供多种训练方案：基础版、快速版、增强版
- 支持加载训练好的模型进行本地手写数字识别
- 支持将自己绘制的数字保存为样本，便于后续扩充数据

## 目录结构

```text
mnist_project/
├─ data/                  # MNIST 数据集缓存目录
├─ my_doodles/            # 手工绘制并保存的 28x28 数字样本
├─ runs_mnist/            # 训练输出目录
│  └─ best_model.pt       # 当前最佳模型权重
├─ train_mnist.py         # 基础 CNN 训练脚本
├─ train_fast.py          # 更快收敛的 FastCNN 训练脚本
├─ train_strong_mnist.py  # 更强模型 SmallResNet 训练脚本
├─ mnist_draw.py          # Tkinter 手写识别画板（兼容多种模型结构）
└─ app_handwrite.py       # Tkinter 手写识别应用（偏向 SmallResNet 权重）
```

## 环境依赖

建议使用 `Python 3.10+`。

核心依赖：

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `tkinter`（标准库，Windows 一般自带）

可用下面的命令安装：

```bash
pip install torch torchvision numpy pillow
```

如果你使用的是 GPU 环境，建议根据自己的 CUDA 版本，从 PyTorch 官网选择对应安装命令。

## 快速开始

### 1. 训练模型

项目提供三套训练脚本：

#### 方案一：基础训练

```bash
python train_mnist.py
```

特点：

- 使用较小的 CNN
- 逻辑简单，适合入门
- 会自动下载或复用 `./data` 下的 MNIST 数据

#### 方案二：快速训练

```bash
python train_fast.py
```

特点：

- 使用 `FastCNN`
- 加入数据增强、`OneCycleLR`、`AdamW`
- 目标是在较短时间内达到较高验证准确率

#### 方案三：增强训练

```bash
python train_strong_mnist.py
```

特点：

- 使用轻量 `SmallResNet`
- 模型表达能力更强
- 更适合作为最终部署模型

### 2. 模型输出

训练完成后，最佳模型会保存到：

```text
runs_mnist/best_model.pt
```

该文件中通常包含：

- 模型参数 `model`
- 网络结构标识 `arch`（部分脚本会写入）
- 归一化参数 `normalize_mean`
- 归一化参数 `normalize_std`

### 3. 启动手写识别

#### 方式一：通用画板识别

```bash
python mnist_draw.py
```

说明：

- 提供一个 Tkinter 画板
- 可直接手写数字并点击“识别”
- 支持“清空”“保存样本”等功能
- 会自动尝试识别当前权重对应的模型结构，兼容性更强

#### 方式二：稳定版手写识别应用

```bash
python app_handwrite.py
```

说明：

- 同样基于 Tkinter
- 采用离屏绘制并预处理后再推理
- 更偏向与 `train_strong_mnist.py` 产出的 `SmallResNet` 权重配套使用

## 数据预处理逻辑

两个识别应用都做了接近 MNIST 的预处理，主要包括：

- 提取前景数字区域
- 按比例缩放到较小尺寸
- 居中放入 `28x28` 画布
- 轻微模糊，模拟手写笔迹边缘
- 按训练时的均值和方差进行标准化

这样可以减少“自己手写”和“MNIST 原始样本”之间的分布差异，提高识别效果。

## 样本保存

在识别应用中点击“保存样本”后，程序会要求输入标签 `0-9`，随后将当前预处理后的样本保存到：

```text
my_doodles/<标签>/
```

例如：

```text
my_doodles/8/1758961913442.png
```

这些样本可用于后续分析、扩充数据集，或继续迭代训练脚本。

## 适用场景

- 深度学习入门练习
- CNN 图像分类实验
- 手写数字识别 Demo
- 课程作业或个人练手项目

## 当前项目状态

根据当前 workspace 可确认：

- `data/` 中已存在 MNIST 原始数据
- `runs_mnist/best_model.pt` 已存在
- `my_doodles/` 下已有若干手工采集数字样本
- 主脚本已通过基础语法检查

## 常见使用流程

推荐按下面顺序体验项目：

1. 先运行 `python train_strong_mnist.py` 训练或刷新最佳模型
2. 再运行 `python app_handwrite.py` 或 `python mnist_draw.py`
3. 在画板中手写数字并测试识别结果
4. 对识别不稳定的样本进行保存，放入 `my_doodles/` 中积累数据

## 后续可改进方向

- 增加 `requirements.txt`
- 增加测试集评估报告与混淆矩阵
- 支持将 `my_doodles/` 融入再训练流程
- 增加 Web 版本前端或 API 服务
- 输出更详细的 Top-K 预测结果与置信度分析

## 许可证

当前仓库中未看到明确许可证声明；如需开源发布，建议补充 `LICENSE` 文件。
