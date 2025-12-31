"""
测试音频处理功能
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_processor import load_audio, preprocess_audio, extract_features
from config import RAW_DATA_DIR, SAMPLE_RATE, AUDIO_DURATION, FEATURE_TYPE

def test_single_audio():
    """测试单个音频文件处理"""
    # 找一个测试音频文件
    test_lang = "zh-CN"
    clips_dir = Path(RAW_DATA_DIR) / test_lang / "cv-corpus-24.0-2025-12-05" / test_lang / "clips"
    
    # 查找第一个mp3文件
    audio_files = list(clips_dir.glob("*.mp3"))
    if not audio_files:
        print(f"未找到音频文件: {clips_dir}")
        return False
    
    test_file = audio_files[0]
    print(f"测试文件: {test_file}")
    print(f"文件存在: {test_file.exists()}")
    
    # 测试1: 加载音频
    print("\n1. 测试加载音频...")
    try:
        # 先测试torchaudio直接加载
        print("  尝试使用torchaudio直接加载...")
        import torchaudio
        try:
            waveform, sr = torchaudio.load(str(test_file))
            print(f"  torchaudio加载成功: shape={waveform.shape}, sr={sr}")
        except Exception as e:
            print(f"  torchaudio加载失败: {e}")
            print("  提示: MP3文件可能需要ffmpeg，尝试安装: pip install ffmpeg-python")
        
        # 测试load_audio函数
        print("  使用load_audio函数...")
        audio, sr = load_audio(str(test_file), target_sr=SAMPLE_RATE)
        if audio is None:
            print("✗ 音频加载失败")
            return False
        print(f"✓ 音频加载成功: shape={audio.shape}, sr={sr}")
    except Exception as e:
        print(f"✗ 音频加载异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试2: 预处理音频
    print("\n2. 测试预处理音频...")
    try:
        processed = preprocess_audio(str(test_file), target_sr=SAMPLE_RATE, 
                                     target_duration=AUDIO_DURATION, normalize=True)
        if processed is None:
            print("✗ 音频预处理失败")
            return False
        print(f"✓ 音频预处理成功: shape={processed.shape}")
    except Exception as e:
        print(f"✗ 音频预处理异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试3: 提取特征
    print(f"\n3. 测试提取特征 (类型: {FEATURE_TYPE})...")
    try:
        features = extract_features(processed, feature_type=FEATURE_TYPE, sr=SAMPLE_RATE)
        if features is None:
            print("✗ 特征提取失败")
            return False
        print(f"✓ 特征提取成功: shape={features.shape}, dtype={features.dtype}")
    except Exception as e:
        print(f"✗ 特征提取异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试4: 保存特征
    print("\n4. 测试保存特征...")
    try:
        from utils.audio_processor import save_features
        test_output = Path("test_feature.npy")
        save_features(features, str(test_output))
        if test_output.exists():
            print(f"✓ 特征保存成功: {test_output}")
            test_output.unlink()  # 删除测试文件
            return True
        else:
            print("✗ 特征保存失败: 文件未创建")
            return False
    except Exception as e:
        print(f"✗ 特征保存异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("音频处理功能测试")
    print("=" * 60)
    
    success = test_single_audio()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过！")
    else:
        print("✗ 测试失败，请检查上述错误信息")
    print("=" * 60)

