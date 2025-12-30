"""
评估指标计算：准确率、精确率、召回率、F1-score、混淆矩阵
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def calculate_metrics(y_true, y_pred, labels=None):
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签列表（用于多分类）
    
    Returns:
        metrics_dict: 指标字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # 多分类指标（宏平均和微平均）
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)
    
    # 每类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }
    
    return metrics

def print_metrics(metrics, idx_to_label=None):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        idx_to_label: 索引到标签的映射
    """
    print("=" * 60)
    print("评估指标")
    print("=" * 60)
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"\n宏平均 (Macro Average):")
    print(f"  精确率 (Precision): {metrics['precision_macro']:.4f}")
    print(f"  召回率 (Recall): {metrics['recall_macro']:.4f}")
    print(f"  F1-score: {metrics['f1_macro']:.4f}")
    print(f"\n微平均 (Micro Average):")
    print(f"  精确率 (Precision): {metrics['precision_micro']:.4f}")
    print(f"  召回率 (Recall): {metrics['recall_micro']:.4f}")
    print(f"  F1-score: {metrics['f1_micro']:.4f}")
    
    if idx_to_label:
        print(f"\n各类别指标:")
        for idx, (prec, rec, f1) in enumerate(zip(
            metrics['precision_per_class'],
            metrics['recall_per_class'],
            metrics['f1_per_class']
        )):
            label = idx_to_label.get(idx, f"Class {idx}")
            print(f"  {label}:")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall: {rec:.4f}")
            print(f"    F1-score: {f1:.4f}")

def get_classification_report(y_true, y_pred, idx_to_label=None):
    """
    获取分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        idx_to_label: 索引到标签的映射
    
    Returns:
        report: 分类报告字符串
    """
    if idx_to_label:
        target_names = [idx_to_label.get(i, f"Class {i}") for i in sorted(idx_to_label.keys())]
    else:
        target_names = None
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0
    )
    return report

