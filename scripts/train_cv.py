"""
交叉验证训练脚本：使用K折交叉验证或留一交叉验证处理小样本数据
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SPLITS_DIR,
    MODEL_SAVE_DIR,
    MODEL_TYPE,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    DEVICE,
    RANDOM_SEED,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    EARLY_STOPPING_PATIENCE,
    BALANCE_DATASET
)
from utils.dataset import LanguageDataset
from utils.models import create_model
from utils.metrics import calculate_metrics

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_single_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, fold_num):
    """训练单个fold"""
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        if scheduler:
            scheduler.step(val_loss)
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                if epoch > 5:  # 至少训练5个epoch
                    break
    
    return best_val_acc, val_losses[-1], train_losses, train_accs, val_losses, val_accs

def train_cross_validation(cv_type='kfold', n_splits=5, use_loo=False):
    """
    交叉验证训练
    
    Args:
        cv_type: 交叉验证类型 ('kfold', 'stratified', 'loo')
        n_splits: K折交叉验证的折数（当cv_type='kfold'或'stratified'时使用）
        use_loo: 是否使用留一交叉验证（如果True，会覆盖cv_type）
    """
    print("=" * 60)
    print("交叉验证训练")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 检查设备
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    train_csv = os.path.join(SPLITS_DIR, 'train.csv')
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"训练集文件不存在: {train_csv}\n请先运行 split_dataset.py")
    
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
        label_to_idx = LABEL_TO_IDX
        idx_to_label = IDX_TO_LABEL
    
    # 加载数据
    metadata_df = pd.read_csv(train_csv)
    print(f"\n数据集大小: {len(metadata_df)}")
    
    # 创建完整数据集（不使用数据增强，因为会在训练时动态应用）
    full_dataset = LanguageDataset(
        train_csv,
        label_to_idx=label_to_idx,
        augment=False  # 在训练时再启用增强
    )
    
    # 获取标签用于分层
    labels = [full_dataset.metadata.iloc[i]['language_label'] for i in range(len(full_dataset))]
    label_indices = [label_to_idx[label] for label in labels]
    
    print(f"\n模型类型: {MODEL_TYPE}")
    print(f"类别数: {NUM_CLASSES}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    
    # 确定交叉验证类型
    if use_loo or cv_type == 'loo':
        print(f"\n使用留一交叉验证 (LOOCV)")
        cv = LeaveOneOut()
        n_splits = len(full_dataset)
    elif cv_type == 'stratified':
        print(f"\n使用分层K折交叉验证 (Stratified K-Fold, K={n_splits})")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    else:
        print(f"\n使用K折交叉验证 (K-Fold, K={n_splits})")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # 存储所有fold的结果
    all_fold_accs = []
    all_fold_losses = []
    all_fold_preds = []
    all_fold_labels = []
    fold_histories = []
    
    print("\n开始交叉验证...")
    print("-" * 60)
    
    # 执行交叉验证
    for fold, (train_indices, val_indices) in enumerate(cv.split(full_dataset, label_indices), 1):
        print(f"\nFold {fold}/{n_splits}")
        print(f"  训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")
        
        # 创建fold的数据集
        # 为训练集创建带增强的数据集
        train_dataset_aug = LanguageDataset(
            train_csv,
            label_to_idx=label_to_idx,
            augment=True  # 训练时使用数据增强
        )
        train_subset = Subset(train_dataset_aug, train_indices)
        
        # 验证集不使用增强
        val_subset = Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 创建新模型（每个fold使用新模型）
        model = create_model(MODEL_TYPE, num_classes=NUM_CLASSES)
        model = model.to(device)
        
        # 损失函数和优化器
        # 根据是否平衡数据集决定是否使用加权损失
        if BALANCE_DATASET:
            # 数据集已平衡，使用普通损失函数
            criterion = nn.CrossEntropyLoss()
        else:
            # 数据集不平衡，计算类别权重并使用加权损失
            import pandas as pd
            
            # 统计当前fold训练集中各类别的样本数
            train_indices_list = list(train_indices)
            train_labels = [full_dataset.metadata.iloc[i]['language_label'] for i in train_indices_list]
            label_counts = pd.Series(train_labels).value_counts()
            total = len(train_indices_list)
            
            # 计算权重：使用逆频率加权
            class_weights = []
            for label in sorted(label_to_idx.keys()):
                count = label_counts.get(label, 1)
                weight = total / (len(label_to_idx) * count)
                class_weights.append(weight)
            
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # 训练这个fold
        best_val_acc, final_val_loss, train_losses, train_accs, val_losses, val_accs = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device, fold
        )
        
        # 最终验证（获取预测结果）
        model.eval()
        fold_preds = []
        fold_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                fold_preds.extend(predicted.cpu().numpy())
                fold_labels.extend(labels.numpy())
        
        all_fold_preds.extend(fold_preds)
        all_fold_labels.extend(fold_labels)
        all_fold_accs.append(best_val_acc)
        all_fold_losses.append(final_val_loss)
        fold_histories.append({
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        })
        
        print(f"  Fold {fold} 最佳验证准确率: {best_val_acc:.4f}")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("交叉验证结果汇总")
    print("=" * 60)
    
    all_fold_preds = np.array(all_fold_preds)
    all_fold_labels = np.array(all_fold_labels)
    
    # 计算总体指标
    labels_list = sorted(idx_to_label.keys())
    overall_metrics = calculate_metrics(all_fold_labels, all_fold_preds, labels=labels_list)
    
    print(f"\n各Fold验证准确率:")
    for i, acc in enumerate(all_fold_accs, 1):
        print(f"  Fold {i}: {acc:.4f}")
    
    print(f"\n平均验证准确率: {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}")
    print(f"平均验证损失: {np.mean(all_fold_losses):.4f} ± {np.std(all_fold_losses):.4f}")
    
    print(f"\n总体性能指标（所有fold合并）:")
    print(f"  准确率: {overall_metrics['accuracy']:.4f}")
    print(f"  宏平均F1: {overall_metrics['f1_macro']:.4f}")
    print(f"  微平均F1: {overall_metrics['f1_micro']:.4f}")
    
    # 保存结果
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    cv_type_name = 'loo' if use_loo or cv_type == 'loo' else f'{n_splits}fold'
    results_path = os.path.join(results_dir, f"{MODEL_TYPE}_cv_{cv_type_name}_results.json")
    
    results = {
        'cv_type': cv_type_name,
        'n_splits': n_splits,
        'fold_accuracies': all_fold_accs,
        'mean_accuracy': float(np.mean(all_fold_accs)),
        'std_accuracy': float(np.std(all_fold_accs)),
        'mean_loss': float(np.mean(all_fold_losses)),
        'std_loss': float(np.std(all_fold_losses)),
        'overall_metrics': {k: float(v) for k, v in overall_metrics.items()},
        'fold_histories': fold_histories
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存: {results_path}")
    
    # 保存评估报告
    from utils.metrics import get_classification_report
    report = get_classification_report(all_fold_labels, all_fold_preds, idx_to_label)
    
    report_path = os.path.join(results_dir, f"{MODEL_TYPE}_cv_{cv_type_name}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"交叉验证评估报告 ({cv_type_name.upper()})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型类型: {MODEL_TYPE}\n")
        f.write(f"数据集大小: {len(full_dataset)}\n")
        f.write(f"交叉验证类型: {cv_type_name}\n")
        f.write(f"折数: {n_splits}\n\n")
        f.write("各Fold结果:\n")
        for i, acc in enumerate(all_fold_accs, 1):
            f.write(f"  Fold {i}: {acc:.4f}\n")
        f.write(f"\n平均准确率: {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}\n")
        f.write(f"平均损失: {np.mean(all_fold_losses):.4f} ± {np.std(all_fold_losses):.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("总体分类报告\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    
    print(f"✓ 评估报告已保存: {report_path}")
    
    # 绘制混淆矩阵
    from utils.visualization import plot_confusion_matrix
    cm_path = os.path.join(results_dir, f"{MODEL_TYPE}_cv_{cv_type_name}_confusion_matrix.png")
    plot_confusion_matrix(all_fold_labels, all_fold_preds, idx_to_label, save_path=cm_path)
    print(f"✓ 混淆矩阵已保存: {cm_path}")
    
    print("\n" + "=" * 60)
    print("交叉验证完成！")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='交叉验证训练')
    parser.add_argument('--cv_type', type=str, default='kfold', 
                       choices=['kfold', 'stratified', 'loo'],
                       help='交叉验证类型: kfold, stratified, loo')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='K折交叉验证的折数（默认5）')
    parser.add_argument('--loo', action='store_true',
                       help='使用留一交叉验证（会覆盖cv_type）')
    
    args = parser.parse_args()
    
    try:
        train_cross_validation(
            cv_type=args.cv_type,
            n_splits=args.n_splits,
            use_loo=args.loo
        )
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()

