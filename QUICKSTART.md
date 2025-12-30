# 快速开始指南

## 前置条件检查清单

在运行项目之前，请确保：

- [ ] ✅ Conda环境已创建并激活
- [ ] ✅ 所有依赖包已安装
- [ ] ✅ 已手动下载5个数据集的tar.gz文件
- [ ] ✅ tar.gz文件已放在项目目录中

## 运行前检查

### 1. 检查conda环境

```bash
# 查看当前激活的环境
conda info --envs

# 激活环境
conda activate language-classification
```

### 2. 检查依赖安装

```bash
python -c "import torch; import librosa; import pandas; import sklearn; print('✓ 所有依赖已安装')"
```

### 3. 检查API密钥配置

```bash
# 检查项目目录下是否有tar.gz文件
# 应该能看到5个tar.gz文件（中文、日语、韩语、粤语、泰语）
```

## 完整运行流程

### 步骤1: 下载数据（需要API密钥）

```bash
python scripts/organize_manual_data.py
```

**说明**：
- 不需要提前准备数据集，脚本会从API自动下载
- 需要有效的Mozilla Data Collective API密钥
- 下载的数据会保存到 `data/raw/` 目录
- 每种语言的数据集较大（几GB），请确保有足够的磁盘空间

### 步骤2: 预处理数据

```bash
python scripts/preprocess_data.py
```

**说明**：
- 会自动从 `data/raw/` 读取数据
- 按说话人采样（每种语言20个说话人训练+10个说话人测试）
- 预处理音频并提取特征
- 生成 `data/processed/dataset_metadata.csv`

### 步骤3: 划分数据集

```bash
python scripts/split_dataset.py
```

**说明**：
- 验证训练/测试集说话人不重叠
- 生成 `data/splits/train.csv` 和 `data/splits/test.csv`

### 步骤4: 训练模型

```bash
python scripts/train.py
```

**说明**：
- 训练时间取决于模型类型和硬件
- 模型会保存到 `models/` 目录
- 训练曲线会保存为图片

### 步骤5: 评估模型

```bash
python scripts/evaluate.py
```

**说明**：
- 评估结果会保存到 `results/` 目录
- 包括混淆矩阵、性能报告等

## 关于数据集

**不需要手动准备数据集！**

- ✅ 脚本会自动从Mozilla Data Collective API下载
- ✅ 支持5种语言：中文、日语、韩语、粤语、泰语
- ✅ 下载后自动解压和处理

**唯一需要的是**：
- Mozilla Data Collective API密钥
- 足够的磁盘空间（每种语言约1-5GB）

## 故障排除

### 问题1: "未设置MOZILLA_API_KEY环境变量"

**解决方案**：
1. 创建 `.env` 文件在项目根目录
2. 添加内容：`MOZILLA_API_KEY=your_api_key`
3. 或设置环境变量（见ENV_SETUP.md）

### 问题2: "ModuleNotFoundError: No module named 'xxx'"

**解决方案**：
```bash
conda activate language-classification
pip install -r requirements.txt
```

### 问题3: 下载失败

**可能原因**：
- API密钥无效
- 网络连接问题
- API服务暂时不可用

**解决方案**：
- 检查API密钥是否正确
- 检查网络连接
- 稍后重试

### 问题4: 内存不足

**解决方案**：
- 减少并行处理进程数（在preprocess_data.py中）
- 使用更小的batch size（已在config.py中设置为8）

## 预期结果

运行成功后，项目结构应该是：

```
language-classification/
├── data/
│   ├── raw/              # 原始数据（下载后）
│   ├── processed/        # 处理后的特征
│   └── splits/           # 数据集划分
├── models/               # 训练好的模型（训练后）
└── results/              # 评估结果（评估后）
```

## 下一步

完成所有步骤后，您可以：
1. 查看 `results/` 目录中的评估报告
2. 查看 `models/` 目录中的训练曲线
3. 尝试不同的模型架构（修改config.py中的MODEL_TYPE）
4. 调整超参数（修改config.py中的配置）

