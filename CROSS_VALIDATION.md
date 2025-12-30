# 交叉验证训练说明

## 概述

对于小样本数据集，使用交叉验证可以：
- **最大化利用数据**：每次都用大部分数据训练，小部分验证
- **提供更可靠的性能评估**：多次验证的平均结果更稳定
- **减少过拟合影响**：通过多次训练-验证循环评估模型泛化能力

## 使用方法

### 1. K折交叉验证（推荐，默认5折）

```bash
python scripts/train_cv.py --cv_type kfold --n_splits 5
```

或者简写：
```bash
python scripts/train_cv.py
```

### 2. 分层K折交叉验证（保持类别比例）

```bash
python scripts/train_cv.py --cv_type stratified --n_splits 5
```

### 3. 留一交叉验证（LOOCV）

**注意**：LOOCV会训练N次（N=样本数），对于265个样本会训练265次，耗时较长。

```bash
python scripts/train_cv.py --loo
```

或者：
```bash
python scripts/train_cv.py --cv_type loo
```

## 输出结果

交叉验证完成后，会在 `results/` 目录下生成：

1. **结果JSON文件**：`{MODEL_TYPE}_cv_{cv_type}_results.json`
   - 包含所有fold的准确率、损失等详细数据
   - 包含训练历史（每个epoch的loss和acc）

2. **评估报告**：`{MODEL_TYPE}_cv_{cv_type}_report.txt`
   - 各fold的准确率
   - 平均准确率和标准差
   - 详细的分类报告（Precision, Recall, F1）

3. **混淆矩阵**：`{MODEL_TYPE}_cv_{cv_type}_confusion_matrix.png`
   - 所有fold合并后的混淆矩阵

## 示例输出

```
交叉验证结果汇总
============================================================

各Fold验证准确率:
  Fold 1: 0.8235
  Fold 2: 0.8118
  Fold 3: 0.8353
  Fold 4: 0.8000
  Fold 5: 0.8176

平均验证准确率: 0.8176 ± 0.0123
平均验证损失: 0.5234 ± 0.0234

总体性能指标（所有fold合并）:
  准确率: 0.8151
  宏平均F1: 0.8123
  微平均F1: 0.8151
```

## 参数说明

- `--cv_type`: 交叉验证类型
  - `kfold`: K折交叉验证（默认）
  - `stratified`: 分层K折交叉验证（保持类别比例）
  - `loo`: 留一交叉验证

- `--n_splits`: K折交叉验证的折数（默认5）
  - 对于小样本，建议使用5折或10折
  - 折数越多，训练时间越长，但评估更稳定

- `--loo`: 使用留一交叉验证（会覆盖cv_type）

## 建议

对于你的数据集（约265个训练样本）：

1. **推荐使用5折交叉验证**：
   ```bash
   python scripts/train_cv.py --cv_type stratified --n_splits 5
   ```
   - 训练5次，每次用约212个样本训练，53个样本验证
   - 平衡了评估稳定性和训练时间

2. **如果想更快评估，使用3折**：
   ```bash
   python scripts/train_cv.py --cv_type stratified --n_splits 3
   ```

3. **如果想最准确的评估，使用LOOCV**（但会很慢）：
   ```bash
   python scripts/train_cv.py --loo
   ```

## 与普通训练的区别

| 特性 | 普通训练 (`train.py`) | 交叉验证 (`train_cv.py`) |
|------|---------------------|----------------------|
| 数据使用 | 固定训练/测试划分 | 多次训练/验证划分 |
| 模型数量 | 1个 | N个（N=折数） |
| 评估可靠性 | 中等 | 高 |
| 训练时间 | 快 | 慢（N倍） |
| 适用场景 | 大数据集 | 小样本数据集 |

## 注意事项

1. **训练时间**：交叉验证会训练N次模型，总时间是单次训练的N倍
2. **内存使用**：每次fold都会创建新模型，内存使用正常
3. **结果解释**：交叉验证的结果更可靠，但需要更多时间
4. **早停机制**：每个fold都有独立的早停机制，避免过拟合

