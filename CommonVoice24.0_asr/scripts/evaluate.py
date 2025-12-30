"""
模型评估脚本：计算评估指标，生成混淆矩阵和可视化结果
"""
import os
import torch
import numpy as np
import json
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from config import (
    SPLITS_DIR,
    MODEL_SAVE_DIR,
    MODEL_TYPE,
    BATCH_SIZE,
    DEVICE,
    IDX_TO_LABEL
)
from utils.dataset import LanguageDataset
from utils.models import create_model
from utils.metrics import calculate_metrics, print_metrics, get_classification_report
from utils.visualization import plot_confusion_matrix, plot_per_class_metrics

def evaluate():
    """主评估函数"""
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 检查设备
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载测试集
    test_csv = os.path.join(SPLITS_DIR, 'test.csv')
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"测试集文件不存在: {test_csv}\n请先运行 split_dataset.py")
    
    # 加载标签映射
    label_mapping_path = os.path.join(SPLITS_DIR, 'label_mapping.json')
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
            label_to_idx = label_mapping['label_to_idx']
            label_to_idx = {k: int(v) for k, v in label_to_idx.items()}
            idx_to_label = label_mapping['idx_to_label']
            idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    else:
        # 尝试从模型目录加载
        model_label_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_TYPE}_label_mapping.json")
        if os.path.exists(model_label_path):
            with open(model_label_path, 'r', encoding='utf-8') as f:
                label_mapping = json.load(f)
                label_to_idx = label_mapping['label_to_idx']
                label_to_idx = {k: int(v) for k, v in label_to_idx.items()}
                idx_to_label = label_mapping['idx_to_label']
                idx_to_label = {int(k): v for k, v in idx_to_label.items()}
        else:
            # 使用默认映射
            from config import LABEL_TO_IDX
            label_to_idx = LABEL_TO_IDX
            idx_to_label = IDX_TO_LABEL
    
    print(f"\n模型类型: {MODEL_TYPE}")
    
    # 创建测试数据集
    test_dataset = LanguageDataset(
        test_csv,
        label_to_idx=label_to_idx,
        augment=False  # 评估时不使用数据增强
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 加载模型
    model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_TYPE}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先运行 train.py 训练模型")
    
    # 创建模型
    from config import NUM_CLASSES
    model = create_model(MODEL_TYPE, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 对于 CNNBiLSTMModel，需要先初始化 feature_projection 层
    # 因为该层是在 forward 中动态创建的
    if MODEL_TYPE == 'cnn_bilstm':
        # 先运行一次 dummy forward 来初始化 feature_projection
        # 获取一个样本来确定输入维度
        dummy_sample, _ = next(iter(test_loader))
        dummy_sample = dummy_sample.to(device)
        with torch.no_grad():
            _ = model(dummy_sample)  # 这会初始化 feature_projection
    
    # 现在可以安全加载 state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 加载模型: {model_path}")
    if 'best_acc' in checkpoint:
        print(f"  训练时最佳准确率: {checkpoint['best_acc']:.4f}")
    
    # 预测
    print("\n开始预测...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    print("\n计算评估指标...")
    labels_list = sorted(idx_to_label.keys())
    metrics = calculate_metrics(all_labels, all_preds, labels=labels_list)
    
    # 打印指标
    print_metrics(metrics, idx_to_label)
    
    # 分类报告
    report = get_classification_report(all_labels, all_preds, idx_to_label)
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(report)
    
    # 保存结果
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存评估报告
    report_path = os.path.join(results_dir, f"{MODEL_TYPE}_evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型类型: {MODEL_TYPE}\n")
        f.write(f"测试集大小: {len(test_dataset)}\n\n")
        f.write("评估指标:\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"\n宏平均:\n")
        f.write(f"  精确率: {metrics['precision_macro']:.4f}\n")
        f.write(f"  召回率: {metrics['recall_macro']:.4f}\n")
        f.write(f"  F1-score: {metrics['f1_macro']:.4f}\n")
        f.write(f"\n微平均:\n")
        f.write(f"  精确率: {metrics['precision_micro']:.4f}\n")
        f.write(f"  召回率: {metrics['recall_micro']:.4f}\n")
        f.write(f"  F1-score: {metrics['f1_micro']:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("分类报告\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    
    print(f"\n✓ 评估报告已保存: {report_path}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(results_dir, f"{MODEL_TYPE}_confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, idx_to_label, save_path=cm_path)
    
    # 绘制各类别性能对比
    per_class_path = os.path.join(results_dir, f"{MODEL_TYPE}_per_class_metrics.png")
    plot_per_class_metrics(metrics, idx_to_label, save_path=per_class_path)
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"结果保存目录: {results_dir}")
    print(f"  - 评估报告: {report_path}")
    print(f"  - 混淆矩阵: {cm_path}")
    print(f"  - 性能对比: {per_class_path}")

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        raise

