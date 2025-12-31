"""
数据完整性检查脚本：检查所有语言的数据是否下载并解压正确
"""
import os
from pathlib import Path
from config import RAW_DATA_DIR, LANGUAGES

def check_language_data(lang_code, lang_name):
    """
    检查单个语言的数据完整性
    
    Returns:
        dict: 检查结果
    """
    lang_dir = Path(RAW_DATA_DIR) / lang_code
    
    result = {
        'lang_code': lang_code,
        'lang_name': lang_name,
        'status': 'incomplete',
        'clips_found': False,
        'validated_found': False,
        'clips_path': None,
        'validated_path': None,
        'audio_count': 0,
        'validated_count': 0,
        'issues': []
    }
    
    # 查找clips目录和validated.tsv文件
    # 可能的位置：
    # 1. data/raw/{lang_code}/clips/
    # 2. data/raw/{lang_code}/cv-corpus-*/{lang_code}/clips/
    # 3. data/raw/{lang_code}/cv-corpus-*/{lang_code_alt}/clips/
    
    clips_paths = []
    validated_paths = []
    
    # 检查标准位置
    standard_clips = lang_dir / "clips"
    standard_validated = lang_dir / "validated.tsv"
    
    if standard_clips.exists():
        clips_paths.append(standard_clips)
    if standard_validated.exists():
        validated_paths.append(standard_validated)
    
    # 检查cv-corpus目录
    for cv_dir in lang_dir.glob("cv-corpus-*"):
        for subdir in cv_dir.iterdir():
            if subdir.is_dir():
                potential_clips = subdir / "clips"
                potential_validated = subdir / "validated.tsv"
                
                if potential_clips.exists():
                    clips_paths.append(potential_clips)
                if potential_validated.exists():
                    validated_paths.append(potential_validated)
    
    # 使用找到的第一个路径
    if clips_paths:
        result['clips_path'] = str(clips_paths[0])
        result['clips_found'] = True
        # 统计音频文件
        audio_files = list(clips_paths[0].glob("*.mp3")) + list(clips_paths[0].glob("*.wav"))
        result['audio_count'] = len(audio_files)
    
    if validated_paths:
        result['validated_path'] = str(validated_paths[0])
        result['validated_found'] = True
        # 尝试读取validated.tsv
        try:
            import pandas as pd
            df = pd.read_csv(validated_paths[0], sep='\t', low_memory=False)
            result['validated_count'] = len(df)
        except Exception as e:
            result['issues'].append(f"无法读取validated.tsv: {e}")
    
    # 判断状态
    if result['clips_found'] and result['validated_found']:
        if result['audio_count'] > 0 and result['validated_count'] > 0:
            result['status'] = 'complete'
        else:
            result['status'] = 'empty'
            if result['audio_count'] == 0:
                result['issues'].append("clips目录为空")
            if result['validated_count'] == 0:
                result['issues'].append("validated.tsv为空")
    else:
        if not result['clips_found']:
            result['issues'].append("未找到clips目录")
        if not result['validated_found']:
            result['issues'].append("未找到validated.tsv文件")
    
    return result

def create_symlinks_if_needed():
    """如果需要，创建符号链接到标准位置"""
    from config import RAW_DATA_DIR, LANGUAGES
    
    for lang_code, lang_name in LANGUAGES.items():
        lang_dir = Path(RAW_DATA_DIR) / lang_code
        
        # 检查标准位置
        standard_clips = lang_dir / "clips"
        standard_validated = lang_dir / "validated.tsv"
        
        # 如果标准位置不存在，查找并创建链接
        if not standard_clips.exists():
            # 查找clips目录
            for cv_dir in lang_dir.glob("cv-corpus-*"):
                for subdir in cv_dir.iterdir():
                    if subdir.is_dir():
                        potential_clips = subdir / "clips"
                        if potential_clips.exists():
                            try:
                                # Windows上创建目录连接（需要管理员权限）或复制
                                # 为了兼容性，我们创建实际的目录结构
                                import shutil
                                if not standard_clips.exists():
                                    # 创建符号链接或复制
                                    try:
                                        os.symlink(potential_clips, standard_clips)
                                        print(f"✓ 创建clips符号链接: {lang_code}")
                                    except:
                                        # Windows上可能不支持符号链接，跳过
                                        pass
                            except Exception as e:
                                pass
        
        if not standard_validated.exists():
            # 查找validated.tsv
            for cv_dir in lang_dir.glob("cv-corpus-*"):
                for subdir in cv_dir.iterdir():
                    if subdir.is_dir():
                        potential_validated = subdir / "validated.tsv"
                        if potential_validated.exists():
                            try:
                                import shutil
                                # 复制文件而不是链接（更可靠）
                                shutil.copy2(potential_validated, standard_validated)
                                print(f"✓ 复制validated.tsv: {lang_code}")
                            except Exception as e:
                                pass

def main():
    """主函数"""
    print("=" * 60)
    print("数据完整性检查")
    print("=" * 60)
    
    results = []
    for lang_code, lang_name in LANGUAGES.items():
        result = check_language_data(lang_code, lang_name)
        results.append(result)
    
    # 打印结果
    print("\n检查结果:")
    print("-" * 60)
    
    all_complete = True
    for result in results:
        status_icon = "✓" if result['status'] == 'complete' else "✗"
        print(f"\n{status_icon} {result['lang_name']} ({result['lang_code']}):")
        
        if result['clips_found']:
            print(f"  clips目录: ✓ ({result['audio_count']} 个音频文件)")
            print(f"    路径: {result['clips_path']}")
        else:
            print(f"  clips目录: ✗ 未找到")
        
        if result['validated_found']:
            print(f"  validated.tsv: ✓ ({result['validated_count']} 条记录)")
            print(f"    路径: {result['validated_path']}")
        else:
            print(f"  validated.tsv: ✗ 未找到")
        
        if result['issues']:
            print(f"  问题:")
            for issue in result['issues']:
                print(f"    - {issue}")
        
        if result['status'] != 'complete':
            all_complete = False
    
    # 尝试创建标准位置的链接/复制
    print("\n" + "=" * 60)
    print("尝试创建标准目录结构...")
    print("=" * 60)
    create_symlinks_if_needed()
    
    # 再次检查
    print("\n重新检查标准位置...")
    final_check = True
    for lang_code, lang_name in LANGUAGES.items():
        lang_dir = Path(RAW_DATA_DIR) / lang_code
        standard_clips = lang_dir / "clips"
        standard_validated = lang_dir / "validated.tsv"
        
        if standard_clips.exists() and standard_validated.exists():
            print(f"✓ {lang_name} ({lang_code}): 标准位置完整")
        else:
            print(f"⚠ {lang_name} ({lang_code}): 标准位置不完整")
            if not standard_clips.exists():
                print(f"  需要手动创建clips目录链接")
            if not standard_validated.exists():
                print(f"  需要手动复制validated.tsv文件")
            final_check = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_complete and final_check:
        print("✓ 所有数据完整，可以开始预处理！")
        print("\n下一步: 运行 python scripts/preprocess_data.py")
    elif all_complete:
        print("✓ 数据已下载，但需要调整目录结构")
        print("\n建议: 运行预处理脚本，它会自动查找数据")
    else:
        print("✗ 部分数据不完整，请检查上述结果")
    print("=" * 60)

if __name__ == "__main__":
    main()

