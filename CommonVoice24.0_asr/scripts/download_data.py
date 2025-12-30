"""
数据下载脚本：从Mozilla Data Collective API下载数据
使用两步下载流程：1) 创建下载会话获取token 2) 使用token下载数据
"""
import os
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import json
from config import (
    RAW_DATA_DIR, 
    LANGUAGES, 
    DATASET_IDS,
    MOZILLA_API_KEY,
    MOZILLA_CLIENT_ID,
    DATA_COLLECTIVE_BASE_URL,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_MAX_RETRIES
)

def create_download_session(dataset_id, api_key, client_id=None):
    """
    步骤1：创建下载会话，获取下载令牌
    
    Args:
        dataset_id: 数据集ID
        api_key: API密钥
        client_id: 客户ID（可选）
    
    Returns:
        download_token: 下载令牌
    """
    url = f"{DATA_COLLECTIVE_BASE_URL}/datasets/{dataset_id}/download"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 如果提供了客户ID，添加到请求头
    if client_id:
        headers["X-Client-Id"] = client_id
    
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # API返回的键名是 downloadToken (驼峰命名)
        download_token = result.get("downloadToken") or result.get("download_token") or result.get("token")
        if not download_token:
            raise ValueError(f"无法从响应中获取下载令牌: {result}")
        
        # 返回令牌和下载URL（如果提供）
        download_url = result.get("downloadUrl") or result.get("download_url")
        return download_token, download_url
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                raise Exception(f"创建下载会话失败: {e}\n响应内容: {error_detail}")
            except:
                raise Exception(f"创建下载会话失败: {e}\n响应内容: {e.response.text}")
        raise Exception(f"创建下载会话失败: {e}")

def download_file_with_auth(url, save_path, api_key, client_id=None, chunk_size=None, max_retries=None):
    """
    使用认证下载文件并显示进度条（支持断点续传）
    
    Args:
        url: 下载URL
        save_path: 保存路径
        api_key: API密钥
        client_id: 客户ID（可选）
        chunk_size: 块大小（默认从config读取，64KB，可加速下载）
        max_retries: 最大重试次数（默认从config读取）
    """
    if chunk_size is None:
        chunk_size = DOWNLOAD_CHUNK_SIZE
    if max_retries is None:
        max_retries = DOWNLOAD_MAX_RETRIES
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # 如果提供了客户ID，添加到请求头
    if client_id:
        headers["X-Client-Id"] = client_id
    
    # 检查是否已存在部分下载的文件（断点续传）
    resume_pos = 0
    if os.path.exists(save_path):
        resume_pos = os.path.getsize(save_path)
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"
            print(f"  检测到未完成的下载，从 {resume_pos / (1024*1024):.2f} MB 处继续...")
    
    # 使用Session复用连接以提高性能
    session = requests.Session()
    session.headers.update(headers)
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, stream=True, timeout=(30, 300))
            
            # 处理断点续传响应
            if resume_pos > 0 and response.status_code == 206:
                # 206 Partial Content - 支持断点续传
                total_size = int(response.headers.get('content-range', '').split('/')[-1])
                mode = 'ab'  # 追加模式
            elif resume_pos > 0 and response.status_code == 200:
                # 服务器不支持断点续传，重新下载
                print(f"  服务器不支持断点续传，重新下载...")
                resume_pos = 0
                total_size = int(response.headers.get('content-length', 0))
                mode = 'wb'
            else:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                mode = 'ab' if resume_pos > 0 else 'wb'
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            with open(save_path, mode) as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                initial=resume_pos,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()  # 立即刷新到磁盘
                        bar.update(len(chunk))
            
            session.close()
            return True
            
        except (requests.exceptions.RequestException, IOError) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"  下载出错，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(wait_time)
            else:
                session.close()
                raise Exception(f"下载失败（已重试{max_retries}次）: {e}")
    
    session.close()
    return False

def extract_archive(archive_path, extract_to):
    """解压tar.gz文件"""
    print(f"正在解压 {archive_path} 到 {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        print(f"解压完成: {extract_to}")
    except Exception as e:
        raise Exception(f"解压失败: {e}")

def download_dataset(dataset_id, download_token, api_key, save_path, client_id=None, download_url=None, chunk_size=None):
    """
    步骤2：使用令牌和API密钥下载数据集
    
    Args:
        dataset_id: 数据集ID
        download_token: 下载令牌
        api_key: API密钥
        save_path: 保存路径
        client_id: 客户ID（可选）
        download_url: 下载URL（如果API提供，优先使用）
        chunk_size: 下载块大小（默认从config读取，64KB，可加速下载）
    """
    if chunk_size is None:
        chunk_size = DOWNLOAD_CHUNK_SIZE
    
    # 如果API提供了下载URL，直接使用；否则构建URL
    if download_url:
        url = download_url
    else:
        url = f"{DATA_COLLECTIVE_BASE_URL}/datasets/{dataset_id}/download/{download_token}"
    
    try:
        download_file_with_auth(url, save_path, api_key, client_id, chunk_size=chunk_size)
        return True
    except Exception as e:
        raise Exception(f"下载失败: {e}")

def download_language_data(lang_code, lang_name):
    """
    下载指定语言的数据集
    
    Args:
        lang_code: 语言代码
        lang_name: 语言名称
    """
    if not MOZILLA_API_KEY:
        raise ValueError("未设置MOZILLA_API_KEY环境变量，请先设置API密钥")
    
    if lang_code not in DATASET_IDS:
        print(f"警告: {lang_name} ({lang_code}) 没有配置数据集ID，跳过")
        return False
    
    dataset_id = DATASET_IDS[lang_code]
    lang_dir = os.path.join(RAW_DATA_DIR, lang_code)
    os.makedirs(lang_dir, exist_ok=True)
    
    # 检查是否已存在解压后的数据
    # Common Voice数据集解压后通常包含clips目录和validated.tsv文件
    clips_dir = os.path.join(lang_dir, "clips")
    validated_path = os.path.join(lang_dir, "validated.tsv")
    
    if os.path.exists(clips_dir) and os.path.exists(validated_path):
        print(f"✓ {lang_name} ({lang_code}) 数据已存在，跳过下载")
        return True
    
    # 下载tar.gz文件
    tar_filename = f"Common_Voice_Scripted_Speech_24.0_{lang_name.replace(' ', '_')}.tar.gz"
    tar_path = os.path.join(lang_dir, tar_filename)
    
    if os.path.exists(tar_path):
        print(f"{lang_name} ({lang_code}) 的压缩包已存在，直接解压...")
    else:
        print(f"\n开始下载 {lang_name} ({lang_code})...")
        print(f"数据集ID: {dataset_id}")
        
        try:
            # 步骤1：创建下载会话
            print("步骤1: 创建下载会话...")
            download_token, download_url = create_download_session(dataset_id, MOZILLA_API_KEY, MOZILLA_CLIENT_ID)
            print(f"✓ 获取下载令牌成功: {download_token[:20]}...")
            
            # 步骤2：下载数据集
            print("步骤2: 下载数据集...")
            if download_url:
                print(f"  使用API提供的下载URL")
            print(f"  下载块大小: {DOWNLOAD_CHUNK_SIZE / 1024:.0f}KB (可在config.py中调整)")
            download_dataset(dataset_id, download_token, MOZILLA_API_KEY, tar_path, MOZILLA_CLIENT_ID, download_url)
            print(f"✓ 下载完成: {tar_path}")
        except Exception as e:
            print(f"✗ 下载 {lang_name} ({lang_code}) 失败: {e}")
            return False
    
    # 解压文件
    if os.path.exists(tar_path):
        try:
            # 解压到语言目录
            extract_archive(tar_path, lang_dir)
            # 验证解压结果
            if os.path.exists(clips_dir) or os.path.exists(validated_path):
                print(f"✓ {lang_name} ({lang_code}) 数据处理完成")
                return True
            else:
                print(f"警告: {lang_name} ({lang_code}) 解压后未找到预期文件")
                return False
        except Exception as e:
            print(f"✗ 解压 {lang_name} ({lang_code}) 失败: {e}")
            return False
    
    return False

def verify_downloads():
    """验证所有语言的下载完整性"""
    print("\n" + "=" * 60)
    print("验证下载完整性...")
    print("=" * 60)
    
    all_complete = True
    for lang_code, lang_name in LANGUAGES.items():
        lang_dir = os.path.join(RAW_DATA_DIR, lang_code)
        clips_dir = os.path.join(lang_dir, "clips")
        validated_path = os.path.join(lang_dir, "validated.tsv")
        
        # 检查clips目录或validated.tsv文件
        if os.path.exists(clips_dir) or os.path.exists(validated_path):
            print(f"✓ {lang_name} ({lang_code}) 数据完整")
        else:
            print(f"✗ {lang_name} ({lang_code}) 数据不完整")
            all_complete = False
    
    return all_complete

def main():
    """主函数"""
    print("=" * 60)
    print("Mozilla Common Voice Scripted Speech 24.0 数据下载工具")
    print("使用 Data Collective API")
    print("=" * 60)
    
    if not MOZILLA_API_KEY:
        print("\n错误: 未设置MOZILLA_API_KEY")
        print("API密钥已在config.py中配置，如果未生效，请检查环境变量或.env文件")
        return
    
    print(f"\nAPI配置:")
    print(f"  API密钥: {MOZILLA_API_KEY[:20]}...{MOZILLA_API_KEY[-10:]} (长度: {len(MOZILLA_API_KEY)})")
    if MOZILLA_CLIENT_ID:
        print(f"  客户ID: {MOZILLA_CLIENT_ID}")
    print(f"将下载 {len(LANGUAGES)} 种语言的数据集\n")
    
    # 下载所有语言的数据
    success_count = 0
    for lang_code, lang_name in LANGUAGES.items():
        if download_language_data(lang_code, lang_name):
            success_count += 1
    
    # 验证下载结果
    print("\n" + "=" * 60)
    print(f"下载完成: {success_count}/{len(LANGUAGES)} 种语言成功")
    print("=" * 60)
    
    verify_downloads()

if __name__ == "__main__":
    main()