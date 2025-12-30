# Conda 环境配置指南

## 1. 创建Conda环境

```bash
# 创建新的conda环境
conda create -n language-classification python=3.10 -y

# 激活环境
conda activate language-classification
```

## 2. 安装依赖包

### 方法1：使用pip安装（推荐）

```bash
# 确保在项目根目录
cd D:\3_1\language-classification

# 安装所有依赖
pip install -r requirements.txt
```

### 方法2：使用conda安装部分包（可选）

```bash
# 使用conda安装一些基础包（可选，pip也可以）
conda install numpy pandas scikit-learn matplotlib seaborn -y
conda install pytorch torchaudio -c pytorch -y  # 如果需要GPU版本，添加 cudatoolkit

# 然后使用pip安装其余包
pip install librosa soundfile tqdm transformers requests python-dotenv
```

## 3. 配置API密钥

### 方法1：创建.env文件（推荐）

在项目根目录 `D:\3_1\language-classification\` 创建 `.env` 文件：

```bash
# Windows PowerShell
cd D:\3_1\language-classification
New-Item -Path .env -ItemType File
# 然后编辑.env文件，添加：
# MOZILLA_API_KEY=your_api_key_here
```

或者手动创建文件，内容为：
```
MOZILLA_API_KEY=your_api_key_here
```

### 方法2：设置环境变量

**Windows PowerShell:**
```powershell
$env:MOZILLA_API_KEY="your_api_key_here"
```

**Windows CMD:**
```cmd
set MOZILLA_API_KEY=your_api_key_here
```

## 4. 验证安装

```bash
# 激活环境
conda activate language-classification

# 验证Python版本
python --version

# 验证关键包是否安装
python -c "import torch; import librosa; import pandas; print('所有包安装成功！')"
```

## 5. 运行项目

```bash
# 确保在项目根目录
cd D:\3_1\language-classification

# 激活conda环境
conda activate language-classification

# 运行下载脚本
python scripts/download_data.py
```

## 常见问题

### Q: 如何检查conda环境是否激活？
A: 命令行提示符前应该显示 `(language-classification)`

### Q: PyTorch安装失败？
A: 如果使用CPU版本，可以：
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Q: librosa安装失败？
A: 可能需要先安装系统依赖（Windows通常不需要），或者：
```bash
pip install librosa --no-cache-dir
```

### Q: 如何退出conda环境？
A: 运行 `conda deactivate`

## 环境信息

- **Python版本**: 3.9 或 3.10（推荐）
- **操作系统**: Windows 10/11
- **主要依赖**: PyTorch, librosa, pandas, scikit-learn

