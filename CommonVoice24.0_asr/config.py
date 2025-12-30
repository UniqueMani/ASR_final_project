"""
配置文件：定义项目中的各种参数
"""
import os
from pathlib import Path

# 尝试加载.env文件（如果存在）
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv未安装，跳过

# 数据路径配置
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

# 语言配置
LANGUAGES = {
    "zh-CN": "中文",
    "ja": "日语",
    "ko": "韩语",
    "yue": "粤语",
    "th": "泰语"
}

# 音频处理配置
SAMPLE_RATE = 16000  # 统一采样率 16kHz
AUDIO_DURATION = 5.0  # 音频时长（秒），超过则裁剪，不足则补零
N_MELS = 128  # Mel频谱图的mel bins数量
N_MFCC = 13  # MFCC特征数量

# 特征提取配置
FEATURE_TYPE = "melspectrogram"  # 可选: "melspectrogram", "mfcc", "raw"
N_FFT = 2048
HOP_LENGTH = 512

# 数据集划分配置
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42
SPEAKER_AWARE_SPLIT = True  # 是否按说话人划分，避免数据泄露

# 模型配置
MODEL_TYPE = "cnn_bilstm"  # 可选: "cnn", "cnn_bilstm", "wav2vec"
NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
DEVICE = "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu"

# CNN模型配置
CNN_HIDDEN_DIM = 128
CNN_NUM_LAYERS = 3

# CNN+BiLSTM模型配置
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2

# wav2vec2配置
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"

# 训练配置
EARLY_STOPPING_PATIENCE = 10
SAVE_BEST_MODEL = True
MODEL_SAVE_DIR = "models"

# 评估配置
EVAL_METRICS = ["accuracy", "precision", "recall", "f1"]

# API配置
MOZILLA_API_KEY = os.getenv("MOZILLA_API_KEY", "f4fea0ddbeba7b5149e4d8c453ee056afed573fc4e1d42f9eda88ec38ce23368")
MOZILLA_CLIENT_ID = os.getenv("MOZILLA_CLIENT_ID", "mdc_9787ff4695ab6c9c6dc77159c7f74963")
DATA_COLLECTIVE_BASE_URL = "https://datacollective.mozillafoundation.org/api"

# 下载配置
DOWNLOAD_CHUNK_SIZE = 65536  # 64KB，可加速下载（可调整为128KB或更大）
DOWNLOAD_MAX_RETRIES = 3  # 最大重试次数

# 数据集ID映射
DATASET_IDS = {
    "zh-CN": "cmj8u3q2n00vhnxxbzrjcugwc",
    "ja": "cmj8u3p9x00clnxxbsv97o45e",
    "ko": "cmj8u3pbv00dxnxxbbsd2udth",
    "yue": "cmj8u3q2s00vlnxxb1jrujrxf",
    "th": "cmj8u3pvx00r9nxxb1yo5z53z"
}

# 小样本采样配置
TRAIN_SPEAKERS_PER_LANGUAGE = 500  # 每种语言训练集说话人数
TEST_SPEAKERS_PER_LANGUAGE = 200   # 每种语言测试集说话人数
USE_SMALL_SAMPLE = False

# 数据平衡配置
BALANCE_DATASET = False  # 是否进行下采样平衡数据集
# 如果BALANCE_DATASET=True：会下采样到最小样本数，训练时使用普通损失函数
# 如果BALANCE_DATASET=False：保持原始不平衡数据，训练时使用加权损失函数

# 语言标签映射（语言代码 → 标签代码）
LANGUAGE_LABEL_MAP = {
    "zh-CN": "zh",
    "ja": "ja",
    "ko": "ko",
    "yue": "yue",
    "th": "th"
}

# 标签到索引的映射（用于模型训练）
LABEL_TO_IDX = {
    "zh": 0,
    "ja": 1,
    "ko": 2,
    "yue": 3,
    "th": 4
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# 训练配置（小样本适配）
BATCH_SIZE = 8  # 减小batch size（100条训练数据）
EARLY_STOPPING_PATIENCE = 5  # 小样本数据，降低patience

