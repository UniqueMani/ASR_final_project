"""
模型定义：三种模型架构（CNN、CNN+BiLSTM、wav2vec2）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from config import (
    NUM_CLASSES,
    CNN_HIDDEN_DIM,
    CNN_NUM_LAYERS,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    WAV2VEC_MODEL_NAME,
    N_MELS,
    N_MFCC
)

class CNNModel(nn.Module):
    """
    CNN模型：适用于melspectrogram/MFCC特征
    输入: (batch, 1, n_features, time_frames)
    输出: (batch, num_classes)
    """
    def __init__(self, num_classes=NUM_CLASSES, input_channels=1, 
                 hidden_dim=CNN_HIDDEN_DIM, num_layers=CNN_NUM_LAYERS):
        super(CNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 卷积层
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            ])
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 1, n_features, time_frames)
        x = self.conv_layers(x)
        x = self.global_pool(x)  # (batch, hidden_dim, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, hidden_dim)
        x = self.classifier(x)
        return x

class CNNBiLSTMModel(nn.Module):
    """
    CNN+BiLSTM模型：结合空间和时序特征
    输入: (batch, 1, n_features, time_frames)
    输出: (batch, num_classes)
    """
    def __init__(self, num_classes=NUM_CLASSES, input_channels=1,
                 cnn_hidden_dim=CNN_HIDDEN_DIM, lstm_hidden_dim=LSTM_HIDDEN_DIM,
                 lstm_num_layers=LSTM_NUM_LAYERS):
        super(CNNBiLSTMModel, self).__init__()
        
        self.num_classes = num_classes
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn_hidden_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # BiLSTM处理时序特征
        # 假设经过CNN后，特征维度变为 (batch, cnn_hidden_dim, reduced_n_features, reduced_time)
        # 我们需要将空间维度展平，然后按时间维度输入LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_hidden_dim,  # LSTM输入维度
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 特征投影层（将在forward中动态创建）
        self.feature_projection = None
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),  # *2因为双向
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 1, n_features, time_frames)
        # CNN特征提取
        x = self.cnn(x)  # (batch, cnn_hidden_dim, reduced_n_features, reduced_time)
        
        # 重塑为LSTM输入格式
        batch_size = x.size(0)
        # 将空间维度展平，保留时间维度
        # x: (batch, cnn_hidden_dim, reduced_n_features, reduced_time)
        # 我们需要按时间维度输入LSTM，所以需要将 (cnn_hidden_dim, reduced_n_features) 展平
        x = x.permute(0, 3, 1, 2)  # (batch, time, cnn_hidden_dim, reduced_n_features)
        x = x.contiguous().view(batch_size, x.size(1), -1)  # (batch, time, cnn_hidden_dim * reduced_n_features)
        
        # 使用线性层将特征维度调整为LSTM输入维度
        # 动态计算输入维度
        feature_dim = x.size(2)
        if self.feature_projection is None or self.feature_projection.in_features != feature_dim:
            # 创建或更新特征投影层
            self.feature_projection = nn.Linear(feature_dim, self.lstm.input_size).to(x.device)
        
        x = self.feature_projection(x)  # (batch, time, lstm.input_size)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, lstm_hidden_dim * 2)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_hidden_dim * 2)
        
        # 分类
        output = self.classifier(lstm_out)
        return output

class Wav2Vec2Model(nn.Module):
    """
    wav2vec2模型：使用预训练的wav2vec2模型
    输入: 原始音频波形 (batch, audio_length)
    输出: (batch, num_classes)
    """
    def __init__(self, num_classes=NUM_CLASSES, model_name=WAV2VEC_MODEL_NAME):
        super(Wav2Vec2Model, self).__init__()
        
        self.num_classes = num_classes
        
        # 加载预训练的wav2vec2模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # 冻结特征提取器（可选：只训练分类头）
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # 获取wav2vec2的输出维度
        wav2vec2_dim = self.wav2vec2.config.hidden_size
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(wav2vec2_dim, wav2vec2_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(wav2vec2_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, audio_length) - 原始音频波形
        # wav2vec2期望输入为(batch, sequence_length)
        outputs = self.wav2vec2(x)
        
        # 获取最后一层的输出
        # outputs.last_hidden_state: (batch, sequence_length, hidden_size)
        # 使用全局平均池化
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # (batch, hidden_size)
        
        # 分类
        output = self.classifier(pooled_output)
        return output

def create_model(model_type, num_classes=NUM_CLASSES, **kwargs):
    """
    创建模型实例
    
    Args:
        model_type: 模型类型 ("cnn", "cnn_bilstm", "wav2vec")
        num_classes: 类别数
        **kwargs: 其他模型参数
    
    Returns:
        model: 模型实例
    """
    if model_type == "cnn":
        return CNNModel(num_classes=num_classes, **kwargs)
    elif model_type == "cnn_bilstm":
        return CNNBiLSTMModel(num_classes=num_classes, **kwargs)
    elif model_type == "wav2vec":
        return Wav2Vec2Model(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

