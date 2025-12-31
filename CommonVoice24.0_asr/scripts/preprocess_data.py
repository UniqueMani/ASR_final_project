"""
数据预处理脚本：整合5种语言数据，按说话人采样，预处理音频并提取特征
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import json

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LANGUAGES,
    LANGUAGE_LABEL_MAP,
    TRAIN_SPEAKERS_PER_LANGUAGE,
    TEST_SPEAKERS_PER_LANGUAGE,
    RANDOM_SEED,
    FEATURE_TYPE,
    SAMPLE_RATE,
    AUDIO_DURATION,
    BALANCE_DATASET
)
from utils.audio_processor import preprocess_audio, extract_features, save_features

def analyze_speaker_distribution(df):
    """
    分析每个说话人的样本数量
    
    Args:
        df: 数据框
    
    Returns:
        speaker_counts: 说话人样本数量统计
    """
    if 'client_id' not in df.columns:
        raise ValueError("数据框中缺少'client_id'列（说话人ID）")
    
    speaker_counts = df.groupby('client_id').size()
    return speaker_counts

def sample_by_speakers(df, n_speakers_train, n_speakers_test, random_seed=42):
    """
    按说话人采样数据
    
    Args:
        df: 数据框
        n_speakers_train: 训练集需要的说话人数量
        n_speakers_test: 测试集需要的说话人数量
        random_seed: 随机种子
    
    Returns:
        train_df: 训练集数据框
        test_df: 测试集数据框
    """
    if 'client_id' not in df.columns:
        raise ValueError("数据框中缺少'client_id'列（说话人ID）")
    
    # 1. 获取所有唯一的说话人ID
    unique_speakers = df['client_id'].unique()
    total_speakers = len(unique_speakers)
    
    required_speakers = n_speakers_train + n_speakers_test
    
    if total_speakers < required_speakers:
        print(f"警告: 只有 {total_speakers} 个说话人，需要 {required_speakers} 个")
        print(f"将使用所有可用说话人")
        n_speakers_train = min(n_speakers_train, total_speakers // 2)
        n_speakers_test = min(n_speakers_test, total_speakers - n_speakers_train)
    
    # 2. 随机打乱
    np.random.seed(random_seed)
    shuffled_speakers = np.random.permutation(unique_speakers)
    
    # 3. 划分说话人（确保不重叠）
    train_speakers = shuffled_speakers[:n_speakers_train]
    test_speakers = shuffled_speakers[n_speakers_train:n_speakers_train+n_speakers_test]
    
    # 4. 从每个说话人的样本中随机选择1条
    train_samples = []
    for speaker in train_speakers:
        speaker_samples = df[df['client_id'] == speaker]
        if len(speaker_samples) > 1:
            # 如果有多条，随机选择1条
            selected = speaker_samples.sample(n=1, random_state=random_seed)
        else:
            selected = speaker_samples
        train_samples.append(selected)
    
    # 同样处理测试集
    test_samples = []
    for speaker in test_speakers:
        speaker_samples = df[df['client_id'] == speaker]
        if len(speaker_samples) > 1:
            selected = speaker_samples.sample(n=1, random_state=random_seed)
        else:
            selected = speaker_samples
        test_samples.append(selected)
    
    train_df = pd.concat(train_samples, ignore_index=True)
    test_df = pd.concat(test_samples, ignore_index=True)
    
    # 添加集合类型标记
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    return train_df, test_df

def process_single_audio(args):
    """
    处理单个音频文件（用于多进程）
    
    Args:
        args: 元组 (row, lang_code, lang_label, raw_audio_dir, feature_dir, feature_type, clips_dir)
    
    Returns:
        result_dict: 处理结果字典，失败返回None
    """
    row, lang_code, lang_label, raw_audio_dir, feature_dir, feature_type, clips_dir = args
    
    try:
        # 构建音频文件路径
        # Common Voice的音频文件通常在clips目录下
        audio_filename = row['path'] if 'path' in row else row.get('file_name', '')
        if not audio_filename:
            return None
        
        # 使用传入的clips_dir路径
        audio_path = os.path.join(clips_dir, audio_filename)
        
        # 如果文件不存在，尝试其他可能的路径
        if not os.path.exists(audio_path):
            # 尝试标准路径
            standard_path = os.path.join(raw_audio_dir, lang_code, 'clips', audio_filename)
            if os.path.exists(standard_path):
                audio_path = standard_path
            else:
                # 尝试cv-corpus子目录
                lang_dir = Path(raw_audio_dir) / lang_code
                for cv_dir in lang_dir.glob("cv-corpus-*"):
                    # 检查语言代码目录
                    lang_subdir = cv_dir / lang_code
                    if lang_subdir.exists():
                        potential_path = lang_subdir / "clips" / audio_filename
                        if potential_path.exists():
                            audio_path = str(potential_path)
                            break
                    
                    # 检查zh-HK（粤语）
                    if lang_code == "yue":
                        hk_subdir = cv_dir / "zh-HK"
                        if hk_subdir.exists():
                            potential_path = hk_subdir / "clips" / audio_filename
                            if potential_path.exists():
                                audio_path = str(potential_path)
                                break
                
                if not os.path.exists(audio_path):
                    return None
        
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            return None
        
        # 预处理音频
        processed_audio = preprocess_audio(audio_path, target_sr=SAMPLE_RATE, 
                                          target_duration=AUDIO_DURATION, normalize=True)
        if processed_audio is None:
            return None
        
        # 提取特征
        features = extract_features(processed_audio, feature_type=feature_type, sr=SAMPLE_RATE)
        if features is None:
            return None
        
        # 保存特征
        # 使用唯一ID作为文件名
        unique_id = row.get('path', '').replace('.mp3', '').replace('.wav', '').replace('/', '_')
        if not unique_id:
            unique_id = f"{lang_code}_{row.name}"
        
        feature_filename = f"{unique_id}.npy"
        feature_path = os.path.join(feature_dir, feature_filename)
        
        # 确保目录存在
        os.makedirs(feature_dir, exist_ok=True)
        
        save_features(features, feature_path)
        
        # 验证文件是否保存成功
        if not os.path.exists(feature_path):
            if not hasattr(process_single_audio, '_save_error_printed'):
                print(f"特征文件保存失败: {feature_path}")
                process_single_audio._save_error_printed = True
            return None
        
        # 返回元数据
        return {
            'feature_path': feature_filename,
            'language_code': lang_code,
            'language_label': lang_label,
            'client_id': row.get('client_id', ''),
            'split': row.get('split', ''),
            'original_path': audio_path
        }
    except Exception as e:
        # 静默失败，不打印详细错误（避免输出过多）
        return None

def find_validated_tsv(lang_code):
    """
    查找validated.tsv文件（支持多种目录结构）
    
    Args:
        lang_code: 语言代码
    
    Returns:
        validated_path: validated.tsv文件路径，如果不存在返回None
    """
    lang_dir = Path(RAW_DATA_DIR) / lang_code
    
    # 检查标准位置
    standard_path = lang_dir / "validated.tsv"
    if standard_path.exists():
        return str(standard_path)
    
    # 检查cv-corpus子目录
    for cv_dir in lang_dir.glob("cv-corpus-*"):
        # 检查语言代码目录
        lang_subdir = cv_dir / lang_code
        if lang_subdir.exists():
            validated_path = lang_subdir / "validated.tsv"
            if validated_path.exists():
                return str(validated_path)
        
        # 检查zh-HK（粤语的特殊情况）
        if lang_code == "yue":
            hk_subdir = cv_dir / "zh-HK"
            if hk_subdir.exists():
                validated_path = hk_subdir / "validated.tsv"
                if validated_path.exists():
                    return str(validated_path)
    
    return None

def find_clips_dir(lang_code):
    """
    查找clips目录（支持多种目录结构）
    
    Args:
        lang_code: 语言代码
    
    Returns:
        clips_dir: clips目录路径，如果不存在返回None
    """
    lang_dir = Path(RAW_DATA_DIR) / lang_code
    
    # 检查标准位置
    standard_clips = lang_dir / "clips"
    if standard_clips.exists():
        return str(standard_clips)
    
    # 检查cv-corpus子目录
    for cv_dir in lang_dir.glob("cv-corpus-*"):
        # 检查语言代码目录
        lang_subdir = cv_dir / lang_code
        if lang_subdir.exists():
            clips_dir = lang_subdir / "clips"
            if clips_dir.exists():
                return str(clips_dir)
        
        # 检查zh-HK（粤语的特殊情况）
        if lang_code == "yue":
            hk_subdir = cv_dir / "zh-HK"
            if hk_subdir.exists():
                clips_dir = hk_subdir / "clips"
                if clips_dir.exists():
                    return str(clips_dir)
    
    return None

def load_language_data(lang_code):
    """
    加载指定语言的validated.tsv文件
    
    Args:
        lang_code: 语言代码
    
    Returns:
        df: 数据框，如果文件不存在返回None
    """
    validated_path = find_validated_tsv(lang_code)
    
    if not validated_path:
        print(f"警告: {lang_code} 的validated.tsv文件不存在")
        return None
    
    try:
        df = pd.read_csv(validated_path, sep='\t', low_memory=False)
        print(f"✓ 加载 {lang_code}: {len(df)} 条记录 (来自: {validated_path})")
        return df
    except Exception as e:
        print(f"✗ 加载 {lang_code} 失败: {e}")
        return None

def preprocess_all_languages():
    """预处理所有语言的数据"""
    print("=" * 60)
    print("数据预处理：整合5种语言数据")
    print("=" * 60)
    
    # 创建输出目录
    feature_dir = os.path.join(PROCESSED_DATA_DIR, 'features')
    os.makedirs(feature_dir, exist_ok=True)
    
    all_metadata = []
    
    # 处理每种语言
    for lang_code, lang_name in LANGUAGES.items():
        print(f"\n处理语言: {lang_name} ({lang_code})")
        print("-" * 60)
        
        # 加载数据
        df = load_language_data(lang_code)
        if df is None or len(df) == 0:
            print(f"跳过 {lang_name} ({lang_code}): 无数据")
            continue
        
        # 分析说话人分布
        speaker_counts = analyze_speaker_distribution(df)
        print(f"说话人总数: {len(speaker_counts)}")
        print(f"每个说话人平均样本数: {speaker_counts.mean():.2f}")
        
        # 按说话人采样
        lang_label = LANGUAGE_LABEL_MAP.get(lang_code, lang_code)
        
        # 根据BALANCE_DATASET决定采样策略
        if BALANCE_DATASET:
            # 平衡模式：使用固定说话人数采样（目标：每种语言相同数量）
            train_df, test_df = sample_by_speakers(
                df, 
                TRAIN_SPEAKERS_PER_LANGUAGE, 
                TEST_SPEAKERS_PER_LANGUAGE,
                random_seed=RANDOM_SEED
            )
        else:
            # 不平衡模式：仍然使用固定说话人数采样
            # 注意：这里使用固定说话人数是为了保持采样策略一致
            # 实际的不平衡会在音频处理失败后自然形成（各语言处理成功率不同）
            # 最终的不平衡数据会在后续阶段保持，不会下采样
            # 这样既保持了采样策略的一致性，又允许自然形成的不平衡
            train_df, test_df = sample_by_speakers(
                df, 
                TRAIN_SPEAKERS_PER_LANGUAGE, 
                TEST_SPEAKERS_PER_LANGUAGE,
                random_seed=RANDOM_SEED
            )
        
        print(f"采样结果:")
        print(f"  训练集: {len(train_df)} 条 (来自 {train_df['client_id'].nunique()} 个说话人)")
        print(f"  测试集: {len(test_df)} 条 (来自 {test_df['client_id'].nunique()} 个说话人)")
        
        # 验证说话人不重叠
        train_speakers = set(train_df['client_id'].unique())
        test_speakers = set(test_df['client_id'].unique())
        overlap = train_speakers & test_speakers
        if overlap:
            raise ValueError(f"{lang_code} 训练集和测试集说话人重叠: {overlap}")
        
        # 合并训练集和测试集用于处理
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # 添加语言标签
        combined_df['language_code'] = lang_code
        combined_df['language_label'] = lang_label
        
        # 获取clips目录路径（用于处理音频）
        clips_dir_path = find_clips_dir(lang_code)
        if not clips_dir_path:
            clips_dir_path = os.path.join(RAW_DATA_DIR, lang_code, 'clips')
        
        print(f"  使用clips目录: {clips_dir_path}")
        
        # 准备处理参数（传递clips目录路径）
        process_args = [
            (row, lang_code, lang_label, RAW_DATA_DIR, feature_dir, FEATURE_TYPE, clips_dir_path)
            for _, row in combined_df.iterrows()
        ]
        
        # 多进程处理音频
        print(f"开始处理 {len(process_args)} 个音频文件...")
        # Windows上需要设置multiprocessing启动方法
        if hasattr(mp, 'set_start_method'):
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # 已经设置过了
        
        num_workers = min(mp.cpu_count(), 8)
        # Windows上减少进程数以避免问题
        if os.name == 'nt':
            num_workers = min(num_workers, 4)
        
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_audio, process_args),
                total=len(process_args),
                desc=f"处理 {lang_name}"
            ))
        
        # 收集成功的结果
        successful_results = [r for r in results if r is not None]
        print(f"✓ 成功处理 {len(successful_results)}/{len(process_args)} 个文件")
        
        # 添加到总元数据
        all_metadata.extend(successful_results)
    
    # 保存元数据
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        
        # 按语言和split统计样本数
        print("\n" + "=" * 60)
        print("处理完成后的样本统计:")
        print("=" * 60)
        
        # 统计各语言的训练集和测试集样本数（音频处理后的实际成功数）
        print("\n各语言样本数（音频处理后的实际成功数）:")
        lang_stats_before = {}
        for lang in LANGUAGES.keys():
            lang_label = LANGUAGE_LABEL_MAP.get(lang, lang)
            train_count = len(metadata_df[(metadata_df['language_label'] == lang_label) & (metadata_df['split'] == 'train')])
            test_count = len(metadata_df[(metadata_df['language_label'] == lang_label) & (metadata_df['split'] == 'test')])
            lang_stats_before[lang_label] = {'train': train_count, 'test': test_count}
            print(f"  {lang_label}: 训练集 {train_count} 条, 测试集 {test_count} 条")
        
        # 根据配置决定是否进行下采样平衡
        # 注意：这里的不平衡是由于音频处理失败率不同导致的
        # 如果BALANCE_DATASET=True：会下采样到最小值，使各语言样本数一致
        # 如果BALANCE_DATASET=False：保持原始不平衡数据，不进行任何下采样
        if BALANCE_DATASET:
            # 找到各语言训练集和测试集的最小样本数
            train_counts = []
            test_counts = []
            for lang in LANGUAGES.keys():
                lang_label = LANGUAGE_LABEL_MAP.get(lang, lang)
                train_count = len(metadata_df[(metadata_df['language_label'] == lang_label) & (metadata_df['split'] == 'train')])
                test_count = len(metadata_df[(metadata_df['language_label'] == lang_label) & (metadata_df['split'] == 'test')])
                if train_count > 0:
                    train_counts.append(train_count)
                if test_count > 0:
                    test_counts.append(test_count)
            
            if train_counts and test_counts:
                min_train = min(train_counts)
                min_test = min(test_counts)
                
                print(f"\n最小样本数: 训练集 {min_train} 条, 测试集 {min_test} 条")
                print("正在统一各语言样本数（删除多余的样本）...")
                print("注意: 已启用数据集平衡，训练时将使用普通损失函数")
                
                # 对每种语言，删除多余的样本，使其等于最小值
                balanced_metadata = []
                for lang in LANGUAGES.keys():
                    lang_label = LANGUAGE_LABEL_MAP.get(lang, lang)
                    
                    # 训练集
                    lang_train = metadata_df[(metadata_df['language_label'] == lang_label) & (metadata_df['split'] == 'train')]
                    original_train_count = len(lang_train)
                    if len(lang_train) > min_train:
                        lang_train = lang_train.sample(n=min_train, random_state=RANDOM_SEED).reset_index(drop=True)
                        print(f"  {lang_label} 训练集: {original_train_count} → {len(lang_train)} 条 (删除了 {original_train_count - min_train} 条)")
                    elif len(lang_train) > 0:
                        print(f"  {lang_label} 训练集: {len(lang_train)} 条 (保持不变)")
                    balanced_metadata.append(lang_train)
                    
                    # 测试集
                    lang_test = metadata_df[(metadata_df['language_label'] == lang_label) & (metadata_df['split'] == 'test')]
                    original_test_count = len(lang_test)
                    if len(lang_test) > min_test:
                        lang_test = lang_test.sample(n=min_test, random_state=RANDOM_SEED + 10000).reset_index(drop=True)
                        print(f"  {lang_label} 测试集: {original_test_count} → {len(lang_test)} 条 (删除了 {original_test_count - min_test} 条)")
                    elif len(lang_test) > 0:
                        print(f"  {lang_label} 测试集: {len(lang_test)} 条 (保持不变)")
                    balanced_metadata.append(lang_test)
                
                # 合并平衡后的数据
                metadata_df = pd.concat(balanced_metadata, ignore_index=True)
        else:
            print("\n注意: 已禁用数据集平衡，将保持原始不平衡数据")
            print("训练时将使用加权损失函数来处理类别不平衡")
        
        # 保存元数据
        metadata_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        
        print("\n" + "=" * 60)
        print("最终统计:")
        print("=" * 60)
        print(f"总样本数: {len(metadata_df)}")
        print(f"  训练集: {len(metadata_df[metadata_df['split'] == 'train'])} 条")
        print(f"  测试集: {len(metadata_df[metadata_df['split'] == 'test'])} 条")
        
        # 统计各语言样本数
        print("\n各语言样本统计（平衡后）:")
        for lang in LANGUAGES.keys():
            lang_label = LANGUAGE_LABEL_MAP.get(lang, lang)
            train_count = len(metadata_df[(metadata_df['language_label'] == lang_label) & (metadata_df['split'] == 'train')])
            test_count = len(metadata_df[(metadata_df['language_label'] == lang_label) & (metadata_df['split'] == 'test')])
            print(f"  {lang_label}: 训练集 {train_count} 条, 测试集 {test_count} 条")
        
        return metadata_path
    else:
        print("\n✗ 没有成功处理任何样本")
        return None

def main():
    """主函数"""
    np.random.seed(RANDOM_SEED)
    
    metadata_path = preprocess_all_languages()
    
    if metadata_path:
        print("\n" + "=" * 60)
        print("数据预处理完成！")
        print("=" * 60)
        print(f"元数据文件: {metadata_path}")
        print(f"特征目录: {os.path.join(PROCESSED_DATA_DIR, 'features')}")
        print("\n下一步: 运行 split_dataset.py 进行最终的数据集划分")
    else:
        print("\n数据预处理失败，请检查错误信息")

if __name__ == "__main__":
    main()

