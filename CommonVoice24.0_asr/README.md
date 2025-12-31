# 五语言语音分类项目

这是一个基于Mozilla Common Voice Scripted Speech 24.0数据集的五语言语音分类项目，支持对中文、日语、韩语、粤语和泰语进行自动分类。

## 项目结构

```
language-classification/
├── config.py                    # 配置文件
├── requirements.txt             # 依赖包
├── .env.example                 # 环境变量示例
├── scripts/
│   ├── download_data.py         # 数据下载脚本
│   ├── preprocess_data.py       # 数据预处理脚本
│   ├── split_dataset.py         # 数据集划分脚本
│   ├── train.py                 # 模型训练脚本
│   └── evaluate.py              # 模型评估脚本
├── utils/
│   ├── audio_processor.py       # 音频处理工具
│   ├── dataset.py               # PyTorch Dataset类
│   ├── models.py                # 模型定义
│   ├── metrics.py               # 评估指标
│   └── visualization.py         # 可视化工具
├── data/
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后的数据
│   └── splits/                  # 数据集划分
├── models/                      # 训练好的模型
└── results/                     # 评估结果
```

## 环境配置

### 使用Conda（推荐）

详细说明请参考 [CONDA_SETUP.md](CONDA_SETUP.md)

**快速设置：**

```bash
# 1. 创建conda环境
conda create -n language-classification python=3.10 -y
conda activate language-classification

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API密钥（见下方）
```

### 使用pip（如果没有conda）

```bash
pip install -r requirements.txt
```

### 配置说明

**注意**：如果您使用手动下载数据集（推荐），则**不需要**配置API密钥。

只有在使用API自动下载时才需要配置API密钥（详细说明请参考 [ENV_SETUP.md](ENV_SETUP.md)）。

## 使用流程

### 步骤1: 下载并整理数据

**手动下载数据集（推荐方式）**

1. **下载5个数据集的tar.gz文件**：
   - 中文（China）
   - 日语（Japanese）
   - 韩语（Korean）
   - 粤语（Hong Kong）
   - 泰语（Thai）

2. **将tar.gz文件放在项目目录**（项目根目录、data/或data/raw/都可以）

3. **运行整理脚本**：
```bash
python scripts/organize_manual_data.py
```

脚本会自动：
- 识别每个文件对应的语言
- 解压到正确的位置
- 整理目录结构
- 验证数据完整性

详细说明请参考 [MANUAL_DOWNLOAD.md](MANUAL_DOWNLOAD.md) 或 [DATA_SETUP.md](DATA_SETUP.md)

### 步骤2: 数据预处理

运行数据预处理脚本，整合数据并按说话人采样：

```bash
python scripts/preprocess_data.py
```

脚本会：
- 加载每种语言的`validated.tsv`文件
- 按说话人采样（每种语言：20个说话人训练 + 10个说话人测试）
- 预处理音频（统一采样率、裁剪/补零、归一化）
- 提取特征（melspectrogram/MFCC/raw）
- 生成元数据文件`data/processed/dataset_metadata.csv`

### 步骤3: 数据集划分

运行数据集划分脚本，生成最终的训练/测试集：

```bash
python scripts/split_dataset.py
```

脚本会：
- 验证训练/测试集说话人不重叠
- 生成`data/splits/train.csv`和`data/splits/test.csv`
- 保存标签映射文件

### 步骤4: 训练模型

运行训练脚本：

```bash
python scripts/train.py
```

训练脚本支持三种模型架构（在`config.py`中配置`MODEL_TYPE`）：
- `cnn`: CNN模型
- `cnn_bilstm`: CNN+BiLSTM模型
- `wav2vec`: wav2vec2预训练模型

训练过程会：
- 自动保存最佳模型到`models/`
- 记录训练曲线
- 支持早停机制

### 步骤5: 评估模型

运行评估脚本：

```bash
python scripts/evaluate.py
```

评估脚本会：
- 在测试集上评估模型性能
- 计算准确率、精确率、召回率、F1-score
- 生成混淆矩阵热力图
- 生成各类别性能对比图
- 保存评估报告到`results/`

## 配置说明

主要配置在`config.py`中：

### 模型配置
- `MODEL_TYPE`: 模型类型（"cnn", "cnn_bilstm", "wav2vec"）
- `BATCH_SIZE`: 批次大小（小样本建议8）
- `NUM_EPOCHS`: 训练轮数
- `LEARNING_RATE`: 学习率

### 数据采样配置
- `TRAIN_SPEAKERS_PER_LANGUAGE`: 每种语言训练集说话人数（默认20）
- `TEST_SPEAKERS_PER_LANGUAGE`: 每种语言测试集说话人数（默认10）
- `USE_SMALL_SAMPLE`: 是否使用小样本模式（默认True）

### 特征提取配置
- `FEATURE_TYPE`: 特征类型（"melspectrogram", "mfcc", "raw"）
- `SAMPLE_RATE`: 采样率（默认16kHz）
- `AUDIO_DURATION`: 音频时长（默认5秒）

## 模型架构

### 1. CNN模型
- 适用于melspectrogram/MFCC特征
- 多层卷积 + 池化 + 全连接分类

### 2. CNN+BiLSTM模型
- CNN提取空间特征
- BiLSTM提取时序特征
- 结合两种特征进行分类

### 3. wav2vec2模型
- 使用预训练的wav2vec2-base-960h
- 适用于原始音频波形
- 冻结特征提取器，只训练分类头

## 评估指标

项目支持以下评估指标：
- **准确率 (Accuracy)**: 整体分类准确率
- **精确率 (Precision)**: 宏平均和微平均
- **召回率 (Recall)**: 宏平均和微平均
- **F1-score**: 宏平均和微平均
- **混淆矩阵**: 各类别分类情况
- **各类别指标**: 每个语言的详细性能

## 注意事项

1. **API密钥安全**: 不要将`.env`文件提交到版本控制系统
2. **小样本训练**: 由于数据量小（100条训练样本），建议：
   - 使用较小的batch size（8）
   - 启用数据增强
   - 使用早停机制
3. **说话人分离**: 确保训练集和测试集的说话人不重叠，避免数据泄露
4. **特征选择**: 根据模型类型选择合适的特征（CNN/CNN+BiLSTM用melspectrogram，wav2vec用raw）

## 实验结果

训练完成后，可以在`results/`目录查看：
- 评估报告（文本格式）
- 混淆矩阵热力图
- 各类别性能对比图
- 训练曲线（在`models/`目录）

## 故障排除

### 下载失败
- 检查API密钥是否正确
- 检查网络连接
- 尝试手动下载数据

### 预处理失败
- 检查原始数据是否完整
- 确认`validated.tsv`文件存在
- 检查音频文件路径是否正确

### 训练失败
- 检查GPU/CPU可用性
- 减小batch size
- 检查特征文件是否存在

## 许可证

本项目用于课程实验和学习目的。

## 参考

- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [Mozilla Data Collective](https://datacollective.mozillafoundation.org/)

