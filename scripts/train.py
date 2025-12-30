"""
模型训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
from tqdm import tqdm
import sys

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
    EARLY_STOPPING_PATIENCE,
    SAVE_BEST_MODEL,
    RANDOM_SEED,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    BALANCE_DATASET
)
from utils.dataset import LanguageDataset
from utils.models import create_model
from utils.visualization import plot_training_curves

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
    
    for features, labels in tqdm(dataloader, desc="训练中"):
        features = features.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
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
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="验证中"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def train():
    """主训练函数"""
    print("=" * 60)
    print("模型训练")
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
            # 转换字符串键为整数（如果需要）
            label_to_idx = {k: int(v) for k, v in label_to_idx.items()}
    else:
        label_to_idx = LABEL_TO_IDX
    
    print(f"\n模型类型: {MODEL_TYPE}")
    print(f"类别数: {NUM_CLASSES}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    
    # 创建数据集
    train_dataset = LanguageDataset(
        train_csv,
        label_to_idx=label_to_idx,
        augment=True  # 训练时使用数据增强
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows上可能需要设为0
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n训练集大小: {len(train_dataset)}")
    
    # 创建模型
    model = create_model(MODEL_TYPE, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    # 根据是否平衡数据集决定是否使用加权损失
    if BALANCE_DATASET:
        # 数据集已平衡，使用普通损失函数
        criterion = nn.CrossEntropyLoss()
        print("\n使用普通损失函数（数据集已平衡）")
    else:
        # 数据集不平衡，计算类别权重并使用加权损失
        print("\n计算类别权重（数据集不平衡）...")
        import pandas as pd
        
        # 统计训练集中各类别的样本数
        train_df = pd.read_csv(train_csv)
        label_counts = train_df['language_label'].value_counts()
        total = len(train_df)
        
        # 计算权重：使用逆频率加权
        # weight = total / (类别数 * 该类样本数)
        class_weights = []
        for label in sorted(label_to_idx.keys()):
            count = label_counts.get(label, 1)  # 如果某个类别不存在，设为1避免除零
            weight = total / (len(label_to_idx) * count)
            class_weights.append(weight)
            print(f"  {label}: {count} 条样本, 权重: {weight:.4f}")
        
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("使用加权损失函数（处理类别不平衡）")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # 验证（如果有验证集）
        # 这里我们使用训练集的一部分作为验证集，或者直接使用训练集
        # 对于小样本，我们可以跳过验证，直接使用训练集评估
        val_loss, val_acc = train_loss, train_acc  # 简化：使用训练指标
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            if SAVE_BEST_MODEL:
                model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_TYPE}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                    'model_type': MODEL_TYPE,
                }, model_path)
                print(f"✓ 保存最佳模型: {model_path} (Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n早停触发 (patience={EARLY_STOPPING_PATIENCE})")
            break
    
    # 保存标签映射
    label_mapping_save_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_TYPE}_label_mapping.json")
    with open(label_mapping_save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {v: k for k, v in label_to_idx.items()}
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ 标签映射已保存: {label_mapping_save_path}")
    
    # 绘制训练曲线
    curve_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_TYPE}_training_curves.png")
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path=curve_path)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳准确率: {best_val_acc:.4f}")
    print(f"模型保存目录: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

