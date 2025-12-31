"""
音频处理工具：音频预处理和特征提取
"""
import os
import numpy as np
from pathlib import Path
import torch
import torchaudio
from config import SAMPLE_RATE, AUDIO_DURATION, N_MELS, N_MFCC, N_FFT, HOP_LENGTH

# 尝试导入librosa，如果失败则使用torchaudio作为备选
HAS_LIBROSA = False
try:
    import librosa
    import soundfile as sf
    # 直接使用librosa，不检查pkg_resources（在运行时处理错误）
    HAS_LIBROSA = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_LIBROSA = False
    print(f"警告: librosa未安装或缺少依赖 ({e})，将使用torchaudio作为备选")

def load_audio(file_path, target_sr=SAMPLE_RATE):
    """
    加载音频文件并统一采样率
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率
    
    Returns:
        audio: 音频波形数组
        sr: 采样率
    """
    # 优先使用librosa，如果失败则使用torchaudio
    if HAS_LIBROSA:
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio, sr
        except (Exception, ModuleNotFoundError) as e:
            # 如果librosa失败（如缺少pkg_resources），fallback到torchaudio
            pass
    
    # 使用torchaudio（作为主要方法或备选）
    try:
        # torchaudio.load 支持MP3，尝试不同的后端
        # 对于MP3文件，优先使用sox_io，如果失败则尝试soundfile
        waveform = None
        sr = None
        
        # 方法1: 尝试sox_io后端（支持MP3）
        try:
            waveform, sr = torchaudio.load(file_path, backend="sox_io")
        except Exception:
            pass
        
        # 方法2: 尝试soundfile后端
        if waveform is None:
            try:
                waveform, sr = torchaudio.load(file_path, backend="soundfile")
            except Exception:
                pass
        
        # 方法3: 使用默认后端（自动选择）
        if waveform is None:
            try:
                waveform, sr = torchaudio.load(file_path)
            except Exception:
                pass
        
        # 方法4: 尝试使用pydub（如果torchaudio都失败）
        if waveform is None:
            try:
                from pydub import AudioSegment  # type: ignore
                import warnings
                # 抑制pydub的ffmpeg警告
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio_seg = AudioSegment.from_mp3(file_path)
                # 转换为单声道
                audio_seg = audio_seg.set_channels(1)
                # 转换为numpy数组
                audio = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
                # 归一化到[-1, 1]
                if audio_seg.sample_width == 2:  # 16-bit
                    audio = audio / 32768.0
                elif audio_seg.sample_width == 4:  # 32-bit
                    audio = audio / 2147483648.0
                else:  # 8-bit
                    audio = (audio - 128.0) / 128.0
                sr = audio_seg.frame_rate
                
                # 重采样到目标采样率
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
                    audio_resampled = resampler(audio_tensor)
                    audio = audio_resampled.squeeze(0).numpy()
                
                return audio, target_sr
            except ImportError:
                pass
            except Exception as e:
                # pydub失败，继续尝试其他方法
                pass
        
        # 方法5: 使用ffmpeg subprocess（最后的备选）
        if waveform is None:
            try:
                import subprocess
                import tempfile
                # 检查ffmpeg是否可用
                try:
                    subprocess.run(['ffmpeg', '-version'], 
                                 capture_output=True, check=True, timeout=5)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    raise Exception("ffmpeg不可用，请安装: conda install -c conda-forge ffmpeg")
                
                # 使用ffmpeg转换为WAV，然后加载
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_wav = tmp_file.name
                
                try:
                    # 使用ffmpeg转换MP3到WAV
                    result = subprocess.run([
                        'ffmpeg', '-i', file_path, '-ar', str(target_sr),
                        '-ac', '1', '-f', 'wav', '-y', tmp_wav
                    ], capture_output=True, check=True, timeout=30)
                    
                    # 加载WAV文件
                    waveform, sr = torchaudio.load(tmp_wav)
                    os.unlink(tmp_wav)  # 删除临时文件
                except subprocess.CalledProcessError as e:
                    if os.path.exists(tmp_wav):
                        os.unlink(tmp_wav)
                    raise Exception(f"ffmpeg转换失败: {e.stderr.decode()[:200]}")
                except Exception as e:
                    if os.path.exists(tmp_wav):
                        os.unlink(tmp_wav)
                    raise
            except Exception:
                pass
        
        if waveform is None:
            raise Exception("所有加载方法都失败。请安装ffmpeg: conda install -c conda-forge ffmpeg")
        
        # 转换为numpy数组
        audio = waveform.numpy()
        
        # 如果是多声道，转换为单声道
        if len(audio.shape) > 1 and audio.shape[0] > 1:
            audio = np.mean(audio, axis=0)
        elif len(audio.shape) > 1:
            audio = audio[0]  # 取第一个声道
        else:
            audio = audio.flatten()
        
        # 确保是1D数组
        audio = audio.flatten()
        
        # 确保数据类型正确
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 归一化到[-1, 1]范围（如果不在这个范围）
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.max(np.abs(audio))
        
        # 重采样到目标采样率
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            audio_resampled = resampler(audio_tensor)
            audio = audio_resampled.squeeze(0).numpy()
        
        return audio, target_sr
    except Exception as e:
        # 静默失败，不打印详细错误（避免输出过多）
        return None, None

def normalize_audio(audio):
    """
    音量归一化
    
    Args:
        audio: 音频波形数组
    
    Returns:
        normalized_audio: 归一化后的音频
    """
    if len(audio) == 0:
        return audio
    
    # 计算RMS并归一化到-1到1之间
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    return audio

def pad_or_truncate_audio(audio, target_length=None, target_duration=AUDIO_DURATION, sr=SAMPLE_RATE):
    """
    音频裁剪或补零，使其达到目标长度
    
    Args:
        audio: 音频波形数组
        target_length: 目标长度（采样点数），如果为None则根据target_duration计算
        target_duration: 目标时长（秒）
        sr: 采样率
    
    Returns:
        processed_audio: 处理后的音频
    """
    if target_length is None:
        target_length = int(target_duration * sr)
    
    current_length = len(audio)
    
    if current_length > target_length:
        # 裁剪：取中间部分
        start = (current_length - target_length) // 2
        audio = audio[start:start + target_length]
    elif current_length < target_length:
        # 补零：在两端补零
        pad_length = target_length - current_length
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0)
    
    return audio

def preprocess_audio(file_path, target_sr=SAMPLE_RATE, target_duration=AUDIO_DURATION, normalize=True):
    """
    完整的音频预处理流程
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率
        target_duration: 目标时长
        normalize: 是否归一化
    
    Returns:
        processed_audio: 处理后的音频数组，失败返回None
    """
    # 1. 加载音频
    audio, sr = load_audio(file_path, target_sr)
    if audio is None:
        return None
    
    # 2. 裁剪或补零
    audio = pad_or_truncate_audio(audio, target_duration=target_duration, sr=target_sr)
    
    # 3. 音量归一化
    if normalize:
        audio = normalize_audio(audio)
    
    return audio

def extract_melspectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    提取Mel频谱图特征
    
    Args:
        audio: 音频波形数组
        sr: 采样率
        n_mels: Mel bins数量
        n_fft: FFT窗口大小
        hop_length: 帧移
    
    Returns:
        mel_spec: Mel频谱图 (n_mels, time_frames)
    """
    if HAS_LIBROSA:
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )
            # 转换为对数刻度
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception:
            # 如果librosa失败，使用torchaudio
            pass
    
    # 使用torchaudio作为备选
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(audio_tensor).squeeze(0).numpy()
    # 转换为对数刻度
    mel_spec_db = 20 * np.log10(np.maximum(mel_spec, 1e-10)) - 20 * np.log10(np.max(mel_spec))
    return mel_spec_db

def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    提取MFCC特征
    
    Args:
        audio: 音频波形数组
        sr: 采样率
        n_mfcc: MFCC系数数量
        n_fft: FFT窗口大小
        hop_length: 帧移
    
    Returns:
        mfcc: MFCC特征 (n_mfcc, time_frames)
    """
    if HAS_LIBROSA:
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            return mfcc
        except Exception:
            # 如果librosa失败，使用torchaudio
            pass
    
    # 使用torchaudio作为备选
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': 128
        }
    )
    mfcc = mfcc_transform(audio_tensor).squeeze(0).numpy()
    return mfcc

def extract_features(audio, feature_type="melspectrogram", sr=SAMPLE_RATE):
    """
    根据特征类型提取特征
    
    Args:
        audio: 音频波形数组
        feature_type: 特征类型 ("melspectrogram", "mfcc", "raw")
        sr: 采样率
    
    Returns:
        features: 提取的特征，失败返回None
    """
    try:
        if feature_type == "melspectrogram":
            return extract_melspectrogram(audio, sr)
        elif feature_type == "mfcc":
            return extract_mfcc(audio, sr)
        elif feature_type == "raw":
            return audio
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
    except Exception as e:
        # 静默失败，返回None
        return None

def save_processed_audio(audio, save_path, sr=SAMPLE_RATE):
    """
    保存处理后的音频
    
    Args:
        audio: 音频数组
        save_path: 保存路径
        sr: 采样率
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if HAS_LIBROSA:
        try:
            sf.write(save_path, audio, sr)
            return
        except Exception:
            pass
    
    # 使用torchaudio作为备选
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    torchaudio.save(save_path, audio_tensor, sr)

def save_features(features, save_path):
    """
    保存提取的特征
    
    Args:
        features: 特征数组
        save_path: 保存路径
    """
    # 确保目录存在（如果路径包含目录）
    dir_path = os.path.dirname(save_path)
    if dir_path:  # 只有当目录路径不为空时才创建
        os.makedirs(dir_path, exist_ok=True)
    np.save(save_path, features)

