"""
PyTorch Dataset类：用于加载语音特征和标签
"""
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from config import FEATURE_TYPE, PROCESSED_DATA_DIR

class LanguageDataset(Dataset):
    """
    语言分类数据集类
    """
    def __init__(self, metadata_csv, label_to_idx, feature_dir=None, 
                 feature_type=None, augment=False):
        """
        初始化数据集
        
        Args:
            metadata_csv: 元数据CSV文件路径
            label_to_idx: 标签到索引的映射字典 {"zh": 0, "ja": 1, ...}
            feature_dir: 特征文件目录，如果为None则从元数据中推断
            feature_type: 特征类型，如果为None则使用config中的设置
            augment: 是否使用数据增强
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        self.augment = augment
        
        # 确定特征目录
        if feature_dir is None:
            self.feature_dir = os.path.join(PROCESSED_DATA_DIR, 'features')
        else:
            self.feature_dir = feature_dir
        
        # 确定特征类型
        if feature_type is None:
            self.feature_type = FEATURE_TYPE
        else:
            self.feature_type = feature_type
        
        # 验证特征文件存在性
        self._validate_features()
    
    def _validate_features(self):
        """验证特征文件是否存在"""
        missing_files = []
        for idx, row in self.metadata.iterrows():
            feature_path = os.path.join(self.feature_dir, row['feature_path'])
            if not os.path.exists(feature_path):
                missing_files.append(feature_path)
        
        if missing_files:
            print(f"警告: 发现 {len(missing_files)} 个缺失的特征文件")
            if len(missing_files) <= 10:
                for f in missing_files[:10]:
                    print(f"  {f}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            feature: 特征张量
            label: 标签索引（整数）
        """
        row = self.metadata.iloc[idx]
        
        # 加载特征
        feature_path = os.path.join(self.feature_dir, row['feature_path'])
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        feature = np.load(feature_path)
        
        # 转换为张量
        if isinstance(feature, np.ndarray):
            feature = torch.FloatTensor(feature)
        else:
            feature = torch.FloatTensor([feature])
        
        # 如果是2D特征（如melspectrogram, MFCC），添加channel维度
        # melspectrogram: (n_mels, time_frames) -> (1, n_mels, time_frames)
        if len(feature.shape) == 2:
            feature = feature.unsqueeze(0)  # (1, n_mels, time_frames)
        
        # 数据增强（如果启用）
        if self.augment:
            feature = self.apply_augmentation(feature)
        
        # 转换标签
        label_code = row['language_label']  # "zh", "ja"等
        label_idx = self.label_to_idx[label_code]
        
        return feature, label_idx
    
    def apply_augmentation(self, feature):
        """
        应用数据增强
        
        Args:
            feature: 特征张量，形状为 (1, n_mels, time_frames) 或 (1, n_mfcc, time_frames)
        
        Returns:
            augmented_feature: 增强后的特征
        """
        # 时间拉伸（Time Stretching）
        if np.random.rand() > 0.5:
            stretch_factor = np.random.uniform(0.9, 1.1)
            original_length = feature.shape[-1]
            new_length = int(original_length * stretch_factor)
            
            # feature应该是3D: (1, n_mels, time_frames)
            # interpolate需要3D输入: (batch, channels, length)
            # 对于melspectrogram/MFCC，我们需要在时间维度（最后一个维度）上插值
            # 所以需要将 (1, n_mels, time) 转换为 (1, n_mels, time) 格式（已经是正确的）
            # 但interpolate的linear模式期望 (batch, channels, length)，其中length是时间维度
            
            # 如果feature是3D (1, channels, time)，直接使用
            # 如果feature是2D (channels, time)，添加batch维度
            if len(feature.shape) == 2:
                feature = feature.unsqueeze(0)  # (1, channels, time)
            
            # 现在feature应该是3D (1, channels, time)
            # interpolate的linear模式需要3D输入，正好匹配
            feature = torch.nn.functional.interpolate(
                feature,
                size=new_length,
                mode='linear',
                align_corners=False
            )
            
            # 如果变长了，裁剪；如果变短了，补零
            if feature.shape[-1] > original_length:
                feature = feature[..., :original_length]
            elif feature.shape[-1] < original_length:
                padding = original_length - feature.shape[-1]
                feature = torch.nn.functional.pad(feature, (0, padding))
        
        # 添加噪声（对特征图）
        if np.random.rand() > 0.5:
            noise = torch.randn_like(feature) * 0.01
            feature = feature + noise
        
        return feature
    
    def get_label_mapping(self):
        """获取标签映射"""
        return {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }

