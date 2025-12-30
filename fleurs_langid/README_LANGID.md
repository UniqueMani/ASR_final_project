# FLEURS Language Identification (Pure NumPy MFCC)

本项目是一个基于 Python 的**语言识别（Language Identification, LID）**系统，旨在区分不同种类的语言。

项目的核心特点在于**从零实现（From Scratch）**：

* **特征提取**：不依赖 `librosa` 或 `torchaudio`，完全使用 **Pure NumPy** 手写实现了 MFCC 特征提取（包含预加重、分帧、加窗、FFT、Mel 滤波器组、DCT、Delta 计算）。
* **模型算法**：提供了从经典的统计模型到鉴别式模型的多种基准实现。

这既是一个极佳的信号处理教学案例，也是一个轻量级、无需深度学习框架（如 PyTorch/TensorFlow）即可运行的语音分类基准。

## 项目结构

```text
.
├── backend/
│   └── langid/
│       ├── mfcc.py          # 核心：纯 Numpy 实现的 MFCC 特征提取
│       └── gauss_model.py   # 核心：纯 Numpy 实现的对角高斯模型
├── scripts/
│   ├── download_fleurs_subset.py # 下载数据集
│   ├── merge_fleurs.py           # 数据预处理与合并
│   ├── train_langid.py           # 训练入口（支持3种方案）
│   ├── eval_langid.py            # 评估入口
│   ├── predict_langid.py         # 单文件推理
│   └── check_dataset.py          # 数据完整性检查
└── data/                         # 数据存放目录（自动生成）

```

## 环境依赖

最低要求：

* Python 3.9+
* `numpy`

推荐安装（用于进度条和高级模型）：

* `tqdm`
* `scikit-learn` (方案二和方案三需要)
* `soundfile` (用于数据下载)

```bash
pip install numpy tqdm scikit-learn soundfile

```

---

## 快速上手

### 1. 准备数据

本项目使用 Google FLEURS 数据集的子集。

**第一步：下载数据**

```bash
python scripts/download_fleurs_subset.py \
  --langs en_us ja_jp cmn_hans_cn ko_kr yue_hant_hk \
  --splits train validation test \
  --format wav \
  --out data/FLEURS_5

```

**第二步：合并与标准化**
将不同语言的文件夹合并为一个统一的数据集，重命名文件以防冲突，并生成带有 `language` 标签的 `manifest.jsonl`。

```bash
python scripts/merge_fleurs.py \
  --root data/FLEURS_5 \
  --out_root data/FLEURS_merged \
  --langs en_us ja_jp cmn_hans_cn ko_kr yue_hant_hk \
  --overwrite

```

### 2. 设置环境变量

为了让脚本能正确引用 `backend` 模块，请将项目根目录加入 `PYTHONPATH`。

* **Linux/Mac:** `export PYTHONPATH=$PYTHONPATH:.`
* **Windows (PowerShell):** `$env:PYTHONPATH = (Get-Location).Path`

---

## 三种训练方案详解

本项目内置了三种不同层级的训练策略，可以通过 `train_langid.py` 的 `--method` 参数切换。

### 方案一：统计特征 + 单高斯模型 (`stats_gauss`)

这是最基础的方法，完全不依赖 `scikit-learn`，**纯 NumPy 实现**。

* **原理**：计算整句语音 MFCC 的统计量（均值、方差）以及一阶、二阶差分的统计量，拼接成一个固定长度的向量。每种语言训练一个对角协方差高斯模型（Diagonal Gaussian）。
* **特点**：极快，代码极简，适合理解概率生成模型。
* **缺点**：因为压缩了时序信息，容易欠拟合，准确率较低。

```bash
python scripts/train_langid.py \
  --manifest data/FLEURS_merged/train/metadata/manifest.jsonl \
  --out data/model_gauss \
  --method stats_gauss \
  --cache_dir data/.mfcc_cache

```

### 方案二：帧级特征 + 高斯混合模型 (`frame_gmm`)

这是非深度学习时代的**黄金标准（Gold Standard）**方法。

* **原理**：不再压缩成整句统计量，而是使用每一帧的 MFCC 特征。为每种语言训练一个 GMM（高斯混合模型）。推理时，计算测试语音所有帧在各语言 GMM 下的平均对数似然，取最大者。
* **配置**：推荐开启 `--cmvn`（倒谱均值方差归一化）以消除信道噪声。
* **特点**：准确率通常远高于方案一，能捕捉更细微的声学特征。
* **依赖**：需要 `scikit-learn`。

```bash
python scripts/train_langid.py \
  --manifest data/FLEURS_merged/train/metadata/manifest.jsonl \
  --out data/model_gmm \
  --method frame_gmm \
  --n_components 32 \
  --global_cmvn \
  --cache_dir data/.mfcc_cache_frames

```

### 方案三：统计特征 + 逻辑回归 (`utt_lr`)

这是**鲁棒性最强**的基线模型。

* **原理**：输入与方案一相同（整句统计量），但使用鉴别式模型（Discriminative Model）—— 逻辑回归（Logistic Regression）进行分类。
* **特点**：相比生成式的高斯模型，逻辑回归直接优化分类边界，通常在同样的简单特征下能获得比方案一高得多的准确率。
* **依赖**：需要 `scikit-learn`。

```bash
python scripts/train_langid.py \
  --manifest data/FLEURS_merged/train/metadata/manifest.jsonl \
  --out data/model_lr \
  --method utt_lr \
  --cache_dir data/.mfcc_cache

```

---

## 评估与推理

### 模型评估

使用 `eval_langid.py` 在测试集上计算准确率和混淆矩阵。

```bash
# 评估 GMM 模型示例
python scripts/eval_langid.py \
  --manifest data/FLEURS_merged/test/metadata/manifest.jsonl \
  --model_dir data/model_gmm \
  --method frame_gmm \
  --report per_label

```

*输出示例：*

```text
[RESULT] Method=frame_gmm Accuracy=XX%
per_label_accuracy:
  - cmn_hans_cn  acc= XX.XX% support=  200 top_confuse=yue_hant_hk(12)
  - en_us        acc= XX.XX% support=  200
  ...

```

### 单文件预测

使用 `predict_langid.py` 对任意 WAV 文件进行语言识别。

```bash
python scripts/predict_langid.py \
  --model_dir data/model_gmm \
  --wav "your_audio_sample.wav"

```

## 贡献与许可

本项目代码开源，旨在用于教育和研究目的。欢迎提交 Issue 或 Pull Request。