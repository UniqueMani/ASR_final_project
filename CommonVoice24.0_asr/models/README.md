# 模型保存目录

此目录用于保存训练好的模型文件。

## 文件说明

训练完成后，此目录会包含以下文件：

- `{model_type}_best.pth` - 最佳模型权重文件
  - 例如：`cnn_bilstm_best.pth`、`cnn_best.pth`、`wav2vec_best.pth`
  
- `{model_type}_label_mapping.json` - 标签映射文件
  - 包含语言标签到索引的映射关系，用于评估时还原标签

- `{model_type}_training_curves.png` - 训练曲线图
  - 显示训练过程中的损失和准确率变化

## 使用方法

1. **训练模型**：
   ```bash
   python scripts/train.py
   ```
   训练完成后，最佳模型会自动保存到此目录。

2. **加载模型**：
   评估脚本会自动从此目录加载模型：
   ```bash
   python scripts/evaluate.py
   ```

## 注意事项

- 模型文件较大，建议不要提交到版本控制系统
- 如果需要保存模型，请使用外部存储或云存储
- 模型文件包含完整的模型状态，可以在不同环境中加载使用

