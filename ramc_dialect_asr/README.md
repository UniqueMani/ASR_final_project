# RAMC Dialect Classification & Real-time ASR Pipeline
# 基于 MagicData-RAMC 的方言分类与实时字幕系统

这是一个科研入门的语音处理项目。它构建了一个从数据清洗、特征提取、模型训练到 Web 端实时展示的完整流水线。

核心功能：
1.  **方言/口音分类**：基于经典语音信号处理特征（MFCC统计量）和机器学习模型（GMM）。
2.  **实时流式字幕**：基于 WebSocket/SSE 的前后端交互，模拟或集成真实 ASR 引擎。
3.  **数据工程**：提供了完整的非平衡数据集重采样、划分与清洗脚本。

---

## 方法论 (Methodology)

本项目没有直接调用“一行代码”的黑盒大模型 API 来做分类，而是实现了经典的语音识别/语种识别技术路线，便于理解底层原理。

### 1. 特征工程 (Feature Engineering)
- **声学特征**：提取 13 维 **MFCC**（梅尔频率倒谱系数）及其一阶差分（$\Delta$）和二阶差分（$\Delta\Delta$），共 39 维特征。
- **统计池化 (Statistical Pooling)**：由于语音长度不一，我们对整段语音（或切片）的帧级特征计算 **均值 (Mean)** 和 **标准差 (Std)**，拼接成一个固定维度的超向量（Supervector）作为输入。

### 2. 建模方法 (Modeling)
- **监督学习 (Supervised)**：采用 **高斯混合模型 (GMM)**。
    - 为每个方言类别（如 "Shang_Hai", "Guang_Dong"）单独训练一个 GMM。
    - 预测时，计算测试语音在所有 GMM 上的对数似然度 (Log-Likelihood)，取最大值对应的标签。
- **无监督学习 (Unsupervised)**：
    - 使用 **K-Means** 聚类算法对说话人嵌入向量进行聚类，探索数据的潜在分布。

---

## 环境安装

建议使用 Python 3.9+。

1. **克隆代码并创建虚拟环境**
   ```bash
   git clone [https://github.com/your-username/ramc_dialect_asr.git](https://github.com/your-username/ramc_dialect_asr.git)
   cd ramc_dialect_asr
   python -m venv .venv
   # Windows
   .\.venv\Scripts\Activate.ps1
   # Linux/Mac
   source .venv/bin/activate

```

2. **安装依赖**
```bash
pip install -r requirements.txt

```


3. **设置环境变量**
为了让脚本能正确导入 `backend` 模块，必须设置 `PYTHONPATH`。
```powershell
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path

```


```bash
# Linux/Mac
export PYTHONPATH=$(pwd)

```



---

## 数据准备流程 (Data Pipeline)

本项目使用 **MagicData-RAMC (OpenSLR 68)** 数据集。

### 1. 下载与解压

请从 OpenSLR 下载数据并解压，结构如下：

```text
data/RAMC/
  train/
  dev/
  test/
  metadata/TRANS.txt, SPKINFO.txt

```

### 2. 数据子采样 (Subsample)

原始数据量巨大，为加快实验速度，我们抽取每个说话人随机 N 条语音（例如 50 条）。

```bash
python scripts/subsample_ramc.py \
  --root ./data/RAMC \
  --splits train,dev,test \
  --n_per_spk 50 \
  --mode copy \
  --dst_root ./data/RAMC_medium

```

### 3. 数据集重平衡 (Rebalance)

**关键步骤**：解决原始数据中“训练集缺方言”、“验证集方言分布不均”的问题。该脚本会将所有数据汇总，按方言标签重新分层划分为 train/dev/test。

```bash
python scripts/rebalance_ramc_splits.py \
  --root ./data/RAMC_medium \
  --out_root ./data/RAMC_balanced \
  --train_ratio 0.8 --dev_ratio 0.1 --test_ratio 0.1 \
  --ensure_train --ensure_test \
  --overwrite

```

### 4. 生成清单 (Manifest)

生成 JSONL 格式的清单文件，供训练和 Web 后端使用。

```bash
# 生成训练集清单
python scripts/prepare_manifest_ramc.py --root ./data/RAMC_balanced --split train --out ./data/RAMC_balanced/train/metadata/manifest.jsonl
# 生成测试集清单
python scripts/prepare_manifest_ramc.py --root ./data/RAMC_balanced --split test --out ./data/RAMC_balanced/test/metadata/manifest.jsonl

```

---

## 模型训练 (Training)

使用 `scripts/train_dialect.py` 进行训练。支持多种筛选模式以优化模型表现。

### 1. 基础全量训练

训练所有出现的方言标签。

```bash
python scripts/train_dialect.py \
  --manifest ./data/RAMC_balanced/train/metadata/manifest.jsonl \
  --out ./data/dialect_model \
  --supervised \
  --max_utts_per_spk 50

```

### 2. Top-N 模式 (只训练主要方言)

数据集中存在长尾分布（某些方言样本极少），使用 `top10` 可以只训练样本量最多的 10 个方言，**显著提升分类准确率**。

```bash
python scripts/train_dialect.py \
  --manifest ./data/RAMC_balanced/train/metadata/manifest.jsonl \
  --out ./data/dialect_model \
  --supervised \
  --label_subset top10 \
  --log_every 10

```

### 3. 排除特定城市/方言 (Exclude Mode)

如果发现某些标签噪声大或者不需要（例如数据质量差的 "Others" 或特定的 "Shang_Hai"），可以使用 `--label_exclude` 将其剔除。

```bash
# 示例：排除上海话和未标注数据
python scripts/train_dialect.py \
  --manifest ./data/RAMC_balanced/train/metadata/manifest.jsonl \
  --out ./data/dialect_model \
  --supervised \
  --label_subset top10 \
  --label_exclude "Shang_Hai,unknown"

```

---

## 评估 (Evaluation)

评估脚本 `scripts/eval_dialect.py` 支持按“语句级别”或“说话人级别”进行评估。

```bash
# 说话人级别评估 (推荐)：会对同一个人的多条语音取平均特征后再预测
python scripts/eval_dialect.py \
  --manifest ./data/RAMC_balanced/test/metadata/manifest.jsonl \
  --level spk \
  --spk_mode embed \
  --report confusion

```

* `--level spk`: 按说话人评估。
* `--report confusion`: 输出混淆矩阵，便于分析哪些方言容易混淆。
* `--cache_dir ./data/mfcc_cache`: 开启缓存，加速二次评估。

---

## 实时 Web Demo

本项目包含一个前后端分离的演示系统。

### 启动后端

```bash
python backend/run.py

```

服务默认运行在 `http://127.0.0.1:8000`。

### 功能说明

1. **Mock 模式 (默认)**：不需要安装 Whisper。直接读取 `manifest.jsonl` 中的真实文本，模拟流式打字机效果。用于演示方言分类与前端交互。
2. **Whisper 模式**：
* 安装依赖：`pip install -r requirements-whisper.txt`
* 设置环境变量：`$env:ASR_ENGINE="whisper"`
* 进行真实的语音转写 + 方言预测。



### 前端界面

访问 `http://127.0.0.1:8000`。

* 上传 `.wav` 文件。
* 调节 `chunk_sec` (切片时长) 观察流式输出的稳定性。
* 界面会实时显示：**当前识别文本 (Partial/Final)** 以及 **预测的方言及其置信度**。

---

## 文件结构说明

```
ramc_dialect_asr/
├── backend/                # 后端代码
│   ├── asr/                # ASR 引擎接口 (Mock/Whisper)
│   ├── dialect/            # 方言预测核心 (GMM推理, 特征提取)
│   ├── streaming/          # 流式处理逻辑 (切片, 文本稳定器)
│   └── server.py           # FastAPI 服务端
├── frontend/               # 前端静态资源 (HTML/JS/CSS)
├── scripts/                # 数据处理与训练脚本
│   ├── subsample_ramc.py       # 1. 子采样
│   ├── rebalance_ramc_splits.py# 2. 数据平衡
│   ├── prepare_manifest_ramc.py# 3. 生成清单
│   ├── train_dialect.py        # 4. 训练模型 (含 Top10/Exclude)
│   └── eval_dialect.py         # 5. 评估模型
└── requirements.txt        # 项目依赖

```

```

```