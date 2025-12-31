# 语种识别系统

基于 ECAPA-TDNN 的三语种识别系统（普通话、藏语、维吾尔语）

## 项目结构

```
classify/
├── data/
│   └── processed/          # 处理后的数据
│       ├── train/          # 训练集
│       │   ├── mandarin/   # 普通话
│       │   ├── tibetan/    # 藏语
│       │   └── uyghur/     # 维吾尔语
│       └── test/           # 测试集
│           ├── mandarin/
│           ├── tibetan/
│           └── uyghur/
├── models/                 # 训练好的模型
│   └── ecapa_lang_id/
├── results/                # 可视化结果
├── scripts/                # 脚本文件
│   ├── preprocess_data.py  # 数据预处理
│   ├── train_ecapa.py      # 模型训练
│   ├── train_config.yaml   # 训练配置
│   ├── visualize_tsne.py   # t-SNE 可视化
│   ├── inference.py        # 推理脚本
│   └── run_all.py          # 主运行脚本
└── requirements.txt        # 依赖包
```

## 数据集

- **普通话**: SHALCAS22A (14,580 个音频文件)
- **藏语**: xbmu_amdo31 (22,630 个音频文件)
- **维吾尔语**: THUYG-20 (9,468 个音频文件)

## 安装依赖

```bash
cd classify
pip install -r requirements.txt
```

## 使用方法

### 方法一：完整流程（推荐）

```bash
cd scripts
python run_all.py
```

选择选项 4 执行完整流程。

### 方法二：分步执行

#### 1. 数据预处理

```bash
cd scripts
python preprocess_data.py
```

功能：
- 统一转换为 16kHz, Mono, WAV 格式
- 将长音频切片为 3-5 秒片段
- 以最少语种为基准进行下采样
- 按 80:20 划分训练集和测试集

#### 2. 模型训练

```bash
python train_ecapa.py
```

模型参数：
- 模型架构：ECAPA-TDNN
- Embedding 维度：192
- 训练轮数：30
- 批大小：32
- 学习率：0.001

#### 3. 可视化

```bash
python visualize_tsne.py
```

生成 t-SNE 降维可视化图，展示不同语种的 embedding 分布。

#### 4. 推理测试

```bash
# 交互模式
python inference.py

# 命令行模式
python inference.py <音频文件路径>
```

## 技术细节

### 数据预处理策略

1. **格式统一**：所有音频转换为 16kHz 单声道 WAV 格式
2. **音频切片**：将长音频切分为 3-5 秒的短片段
3. **数据平衡**：
   - 基准：维吾尔语（9,468 个文件，最少）
   - 对普通话和藏语进行下采样
   - 确保三个语种的训练数据量完全相同（1:1:1）

### 模型架构

- **特征提取**：80 维 Fbank 特征
- **Embedding 模型**：ECAPA-TDNN
  - Channels: [1024, 1024, 1024, 1024, 3072]
  - Kernel sizes: [5, 3, 3, 3, 1]
  - Dilations: [1, 2, 3, 4, 1]
  - Attention channels: 128
  - Output embedding: 192 维
- **分类器**：全连接层 (192 -> 3)
- **损失函数**：Negative Log Likelihood Loss

### 训练设置

- **优化器**：Adam (lr=0.001)
- **学习率调度**：NewBob Scheduler
- **Batch Size**：32
- **Epochs**：30
- **数据划分**：
  - 训练集：72% (80% of 90%)
  - 验证集：8% (10% of 90%)
  - 测试集：20%

## 可视化结果

t-SNE 可视化展示了三种语种在 embedding 空间中的分布：
- 红色：普通话
- 青色：藏语
- 黄色：维吾尔语

结果保存在 `classify/results/tsne_visualization.png`

## 性能评估

模型在测试集上的性能指标：
- 准确率 (Accuracy)
- 错误率 (Error Rate)
- 混淆矩阵

详见训练日志：`classify/models/ecapa_lang_id/train_log.txt`

## 注意事项

1. 确保有足够的磁盘空间（处理后的数据约占用数 GB）
2. 建议使用 GPU 进行训练以加快速度
3. 如果内存不足，可以减小 batch_size
4. 首次运行会下载 SpeechBrain 预训练模型

## 参考文献

- SpeechBrain: https://speechbrain.github.io/
- ECAPA-TDNN: Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"

## 作者

语音识别课程项目
