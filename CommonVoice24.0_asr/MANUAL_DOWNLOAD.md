# 手动下载数据集指南

## 数据集存放位置

请将下载的5个数据集按照以下结构放置：

```
data/
└── raw/
    ├── zh-CN/          # 中文（中国）
    │   ├── clips/      # 音频文件目录
    │   └── validated.tsv  # 元数据文件
    ├── ja/             # 日语
    │   ├── clips/
    │   └── validated.tsv
    ├── ko/             # 韩语
    │   ├── clips/
    │   └── validated.tsv
    ├── yue/            # 粤语（香港）
    │   ├── clips/
    │   └── validated.tsv
    └── th/             # 泰语
        ├── clips/
        └── validated.tsv
```

## 下载的数据集文件

您下载的应该是5个tar.gz压缩包，文件名可能类似：
- `Common Voice Scripted Speech 24.0 - Chinese (China).tar.gz`
- `Common Voice Scripted Speech 24.0 - Japanese.tar.gz`
- `Common Voice Scripted Speech 24.0 - Korean.tar.gz`
- `Common Voice Scripted Speech 24.0 - Chinese (Hong Kong).tar.gz`
- `Common Voice Scripted Speech 24.0 - Thai.tar.gz`

## 整理步骤

### 方法1：使用整理脚本（推荐）

运行整理脚本，自动解压和整理：

```bash
python scripts/organize_manual_data.py
```

脚本会自动：
1. 查找项目目录下的tar.gz文件
2. 识别语言类型
3. 解压到正确的位置
4. 验证数据结构

### 方法2：手动整理

1. **解压每个tar.gz文件**

   每个压缩包解压后通常包含：
   - `clips/` 目录（包含所有音频文件）
   - `validated.tsv` 文件（元数据）

2. **移动到正确位置**

   将解压后的内容移动到对应的语言目录：
   
   ```
   中文 → data/raw/zh-CN/
   日语 → data/raw/ja/
   韩语 → data/raw/ko/
   粤语 → data/raw/yue/
   泰语 → data/raw/th/
   ```

3. **验证结构**

   每个语言目录应该包含：
   - `clips/` 目录（存在且包含音频文件）
   - `validated.tsv` 文件（存在且可读）

## 验证数据

运行验证脚本检查数据完整性：

```bash
python scripts/verify_data.py
```

## 常见问题

### Q: 解压后的目录结构不对？

A: Common Voice数据集解压后可能有不同的目录结构。如果遇到问题：
1. 检查解压后的目录内容
2. 找到 `clips/` 目录和 `validated.tsv` 文件
3. 将它们移动到对应的语言目录

### Q: 找不到validated.tsv文件？

A: 可能文件名不同，常见名称：
- `validated.tsv`
- `validated.tsv.gz`（需要先解压）
- `train.tsv`（如果只有训练集）

### Q: clips目录为空？

A: 检查：
1. 解压是否完整
2. 音频文件是否在子目录中
3. 文件路径是否正确

## 下一步

数据整理完成后，运行：

```bash
# 预处理数据
python scripts/preprocess_data.py

# 划分数据集
python scripts/split_dataset.py

# 训练模型
python scripts/train.py
```

