"""
可视化工具：混淆矩阵、训练曲线、性能对比图
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils.metrics import calculate_metrics

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_confusion_matrix(y_true, y_pred, idx_to_label, save_path=None, figsize=(10, 8)):
    """
    绘制混淆矩阵热力图
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        idx_to_label: 索引到标签的映射
        save_path: 保存路径
        figsize: 图像大小
    """
    from sklearn.metrics import confusion_matrix
    
    # 计算混淆矩阵
    labels = sorted(idx_to_label.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 转换为百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建标签名称
    label_names = [idx_to_label[i] for i in labels]
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={'label': '百分比 (%)'}
    )
    
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title('混淆矩阵 (Confusion Matrix)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    
    plt.close()

def plot_training_curves(train_losses, train_accs, val_losses=None, val_accs=None, 
                        save_path=None, figsize=(12, 5)):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        val_losses: 验证损失列表（可选）
        val_accs: 验证准确率列表（可选）
        save_path: 保存路径
        figsize: 图像大小
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
    if val_accs:
        ax2.plot(epochs, val_accs, 'r-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('训练准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    
    plt.close()

def plot_per_class_metrics(metrics, idx_to_label, save_path=None, figsize=(12, 6)):
    """
    绘制各类别性能对比图
    
    Args:
        metrics: 指标字典
        idx_to_label: 索引到标签的映射
        save_path: 保存路径
        figsize: 图像大小
    """
    labels = sorted(idx_to_label.keys())
    label_names = [idx_to_label[i] for i in labels]
    
    precision = metrics['precision_per_class']
    recall = metrics['recall_per_class']
    f1 = metrics['f1_per_class']
    
    x = np.arange(len(label_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-score', alpha=0.8)
    
    ax.set_xlabel('语言类别', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('各类别性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # 添加数值标签
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"性能对比图已保存: {save_path}")
    
    plt.close()

