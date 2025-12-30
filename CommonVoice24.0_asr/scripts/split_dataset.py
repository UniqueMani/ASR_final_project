"""
数据集划分脚本：验证说话人不重叠，生成最终的train.csv和test.csv
"""
import os
import pandas as pd
from config import (
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    LABEL_TO_IDX
)

def verify_speaker_separation(train_df, test_df):
    """
    验证训练集和测试集的说话人不重叠
    
    Args:
        train_df: 训练集数据框
        test_df: 测试集数据框
    
    Returns:
        bool: 如果无重叠返回True，否则抛出异常
    """
    if 'client_id' not in train_df.columns or 'client_id' not in test_df.columns:
        print("警告: 数据框中缺少'client_id'列，无法验证说话人分离")
        return True
    
    train_speakers = set(train_df['client_id'].unique())
    test_speakers = set(test_df['client_id'].unique())
    overlap = train_speakers & test_speakers
    
    if overlap:
        raise ValueError(f"发现说话人重叠: {len(overlap)} 个说话人同时出现在训练集和测试集")
    
    print(f"✓ 验证通过: 训练集和测试集说话人完全分离")
    print(f"  训练集说话人数: {len(train_speakers)}")
    print(f"  测试集说话人数: {len(test_speakers)}")
    return True

def generate_final_splits():
    """生成最终的数据集划分文件"""
    print("=" * 60)
    print("数据集划分：生成train.csv和test.csv")
    print("=" * 60)
    
    # 加载元数据
    metadata_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}\n请先运行 preprocess_data.py")
    
    metadata_df = pd.read_csv(metadata_path)
    print(f"\n加载元数据: {len(metadata_df)} 条记录")
    
    # 按split列划分
    if 'split' not in metadata_df.columns:
        raise ValueError("元数据中缺少'split'列，请检查预处理脚本")
    
    train_df = metadata_df[metadata_df['split'] == 'train'].copy()
    test_df = metadata_df[metadata_df['split'] == 'test'].copy()
    
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    
    # 验证说话人不重叠
    verify_speaker_separation(train_df, test_df)
    
    # 统计各类别样本数
    print("\n训练集各类别样本数:")
    train_lang_counts = train_df.groupby('language_label').size()
    for lang, count in train_lang_counts.items():
        print(f"  {lang}: {count} 条")
    
    print("\n测试集各类别样本数:")
    test_lang_counts = test_df.groupby('language_label').size()
    for lang, count in test_lang_counts.items():
        print(f"  {lang}: {count} 条")
    
    # 保存划分结果
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    train_path = os.path.join(SPLITS_DIR, 'train.csv')
    test_path = os.path.join(SPLITS_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ 划分文件已保存:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    
    # 保存标签映射（用于训练和评估）
    label_mapping_path = os.path.join(SPLITS_DIR, 'label_mapping.json')
    import json
    with open(label_mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'label_to_idx': LABEL_TO_IDX,
            'idx_to_label': {v: k for k, v in LABEL_TO_IDX.items()}
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  {label_mapping_path}")
    
    return train_path, test_path

def main():
    """主函数"""
    try:
        train_path, test_path = generate_final_splits()
        
        print("\n" + "=" * 60)
        print("数据集划分完成！")
        print("=" * 60)
        print(f"训练集: {train_path}")
        print(f"测试集: {test_path}")
        print("\n下一步: 运行 train.py 开始训练模型")
    except Exception as e:
        print(f"\n✗ 数据集划分失败: {e}")
        raise

if __name__ == "__main__":
    main()

