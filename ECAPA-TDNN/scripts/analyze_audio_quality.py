"""
Audio Quality and Channel Analysis Script
Analyze background noise, channel characteristics, and signal quality
检查藏语数据是否包含较多背景噪声或特殊信道特征
"""

# IMPORTANT: Set environment variables BEFORE importing any audio libraries
import os
os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import librosa
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_audio_features(audio_path, sr_target=16000):
    """Extract comprehensive audio features"""
    try:
        # Load audio
        y, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Resample if needed
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        features = {}

        # 1. Signal-to-Noise Ratio (SNR) estimation
        # Use simple energy-based method
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        # Compute frame energies
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0)

        # Estimate noise level (bottom 10% of frame energies)
        noise_threshold = np.percentile(frame_energies, 10)
        signal_energy = np.mean(frame_energies)

        if noise_threshold > 0:
            snr = 10 * np.log10(signal_energy / noise_threshold)
        else:
            snr = 100  # Very high SNR if no noise detected

        features['snr'] = snr
        features['noise_level'] = float(np.sqrt(noise_threshold))
        features['signal_energy'] = float(np.sqrt(signal_energy))

        # 2. Spectral features
        # Compute spectrogram
        D = np.abs(librosa.stft(y))

        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))

        # Spectral rolloff (frequency below which 85% of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))

        # Zero crossing rate (indicates noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # 3. MFCC statistics (channel characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))

        # 4. Spectral flatness (measures how noise-like vs tone-like)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = float(np.std(spectral_flatness))

        # 5. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        # 6. Spectral contrast (useful for detecting channel effects)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = float(np.mean(spectral_contrast[i]))

        # 7. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))

        # 8. High frequency ratio (indicates channel bandwidth)
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_idx = freqs > 4000  # Above 4kHz
        high_freq_energy = np.sum(D[high_freq_idx, :])
        total_energy = np.sum(D)
        features['high_freq_ratio'] = float(high_freq_energy / total_energy if total_energy > 0 else 0)

        # 9. Statistical properties
        features['energy_kurtosis'] = float(kurtosis(frame_energies))
        features['energy_skewness'] = float(skew(frame_energies))

        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def analyze_dataset(data_folder, sample_limit=None):
    """Analyze all audio files in the dataset"""
    languages = ['mandarin', 'tibetan', 'uyghur']
    all_features = defaultdict(list)
    file_info = []

    for lang in languages:
        print(f"\nAnalyzing {lang} dataset...")
        lang_folder = Path(data_folder) / 'test' / lang
        audio_files = sorted(list(lang_folder.glob('*.wav')))

        if sample_limit:
            audio_files = audio_files[:sample_limit]

        for audio_file in tqdm(audio_files):
            features = extract_audio_features(str(audio_file))
            if features:
                features['language'] = lang
                features['filename'] = audio_file.name

                for key, value in features.items():
                    all_features[key].append(value)

                file_info.append({
                    'language': lang,
                    'filename': audio_file.name,
                    'path': str(audio_file)
                })

    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    return df, file_info


def plot_snr_comparison(df, output_path):
    """Compare SNR across languages"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    languages = ['mandarin', 'tibetan', 'uyghur']
    lang_names = {'mandarin': 'Mandarin', 'tibetan': 'Tibetan', 'uyghur': 'Uyghur'}
    colors = {'mandarin': '#FF6B6B', 'tibetan': '#4ECDC4', 'uyghur': '#FFE66D'}

    # Box plot
    ax = axes[0]
    data_to_plot = [df[df['language'] == lang]['snr'].values for lang in languages]
    bp = ax.boxplot(data_to_plot, labels=[lang_names[l] for l in languages],
                    patch_artist=True, showmeans=True)

    for patch, lang in zip(bp['boxes'], languages):
        patch.set_facecolor(colors[lang])
        patch.set_alpha(0.7)

    ax.set_ylabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    ax.set_title('SNR Comparison by Language', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Violin plot
    ax = axes[1]
    for i, lang in enumerate(languages):
        data = df[df['language'] == lang]['snr'].values
        parts = ax.violinplot([data], positions=[i], widths=0.7, showmeans=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[lang])
            pc.set_alpha(0.7)

    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels([lang_names[l] for l in languages])
    ax.set_ylabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    ax.set_title('SNR Distribution by Language', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"SNR comparison saved to: {output_path}")
    plt.show()


def plot_noise_level_comparison(df, output_path):
    """Compare noise levels across languages"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    languages = ['mandarin', 'tibetan', 'uyghur']
    lang_names = {'mandarin': 'Mandarin', 'tibetan': 'Tibetan', 'uyghur': 'Uyghur'}
    colors = {'mandarin': '#FF6B6B', 'tibetan': '#4ECDC4', 'uyghur': '#FFE66D'}

    # Noise level
    ax = axes[0, 0]
    for lang in languages:
        data = df[df['language'] == lang]['noise_level'].values
        ax.hist(data, bins=30, alpha=0.6, label=lang_names[lang], color=colors[lang])
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Background Noise Level Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Zero crossing rate
    ax = axes[0, 1]
    for lang in languages:
        data = df[df['language'] == lang]['zcr_mean'].values
        ax.hist(data, bins=30, alpha=0.6, label=lang_names[lang], color=colors[lang])
    ax.set_xlabel('Zero Crossing Rate', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Zero Crossing Rate (Noisiness Indicator)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Spectral flatness
    ax = axes[1, 0]
    for lang in languages:
        data = df[df['language'] == lang]['spectral_flatness_mean'].values
        ax.hist(data, bins=30, alpha=0.6, label=lang_names[lang], color=colors[lang])
    ax.set_xlabel('Spectral Flatness', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Spectral Flatness (Noise-like vs Tone-like)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # High frequency ratio
    ax = axes[1, 1]
    for lang in languages:
        data = df[df['language'] == lang]['high_freq_ratio'].values
        ax.hist(data, bins=30, alpha=0.6, label=lang_names[lang], color=colors[lang])
    ax.set_xlabel('High Frequency Ratio (>4kHz)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('High Frequency Content (Channel Bandwidth)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Noise analysis saved to: {output_path}")
    plt.show()


def plot_spectral_features_comparison(df, output_path):
    """Compare spectral features across languages"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    languages = ['mandarin', 'tibetan', 'uyghur']
    lang_names = {'mandarin': 'Mandarin', 'tibetan': 'Tibetan', 'uyghur': 'Uyghur'}
    colors = {'mandarin': '#FF6B6B', 'tibetan': '#4ECDC4', 'uyghur': '#FFE66D'}

    features_to_plot = [
        ('spectral_centroid_mean', 'Spectral Centroid (Brightness)'),
        ('spectral_bandwidth_mean', 'Spectral Bandwidth'),
        ('spectral_rolloff_mean', 'Spectral Rolloff'),
        ('rms_mean', 'RMS Energy')
    ]

    for idx, (feature, title) in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        for lang in languages:
            data = df[df['language'] == lang][feature].values
            ax.hist(data, bins=30, alpha=0.6, label=lang_names[lang], color=colors[lang])
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Spectral features comparison saved to: {output_path}")
    plt.show()


def plot_channel_features_pca(df, output_path):
    """PCA visualization of channel features"""
    # Select channel-related features
    channel_features = [
        'spectral_centroid_mean', 'spectral_bandwidth_mean',
        'spectral_rolloff_mean', 'spectral_flatness_mean',
        'high_freq_ratio', 'spectral_contrast_0_mean',
        'spectral_contrast_1_mean', 'spectral_contrast_2_mean'
    ]

    X = df[channel_features].values
    languages = df['language'].values

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    lang_names = {'mandarin': 'Mandarin', 'tibetan': 'Tibetan', 'uyghur': 'Uyghur'}
    colors = {'mandarin': '#FF6B6B', 'tibetan': '#4ECDC4', 'uyghur': '#FFE66D'}

    for lang in ['mandarin', 'tibetan', 'uyghur']:
        mask = languages == lang
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=colors[lang], label=lang_names[lang],
                  alpha=0.6, s=50, edgecolors='w', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title('PCA of Channel Features\n(Are Tibetan and Uyghur clustered together?)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PCA visualization saved to: {output_path}")
    plt.show()


def plot_channel_features_tsne(df, output_path):
    """t-SNE visualization of channel features"""
    # Select channel-related features
    channel_features = [
        'spectral_centroid_mean', 'spectral_bandwidth_mean',
        'spectral_rolloff_mean', 'spectral_flatness_mean',
        'high_freq_ratio', 'spectral_contrast_0_mean',
        'spectral_contrast_1_mean', 'spectral_contrast_2_mean',
        'mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean'
    ]

    X = df[channel_features].values
    languages = df['language'].values

    # t-SNE
    print("\nPerforming t-SNE on channel features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    lang_names = {'mandarin': 'Mandarin', 'tibetan': 'Tibetan', 'uyghur': 'Uyghur'}
    colors = {'mandarin': '#FF6B6B', 'tibetan': '#4ECDC4', 'uyghur': '#FFE66D'}

    for lang in ['mandarin', 'tibetan', 'uyghur']:
        mask = languages == lang
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=colors[lang], label=lang_names[lang],
                  alpha=0.6, s=50, edgecolors='w', linewidth=0.5)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE of Channel Features\n(Checking if Tibetan/Uyghur share similar channel characteristics)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE visualization saved to: {output_path}")
    plt.show()


def compute_similarity_matrix(df):
    """Compute inter-language feature similarity"""
    languages = ['mandarin', 'tibetan', 'uyghur']

    # Select all numeric features except language
    feature_cols = [col for col in df.columns if col not in ['language', 'filename']]

    # Compute mean feature vectors for each language
    mean_features = {}
    for lang in languages:
        mean_features[lang] = df[df['language'] == lang][feature_cols].mean().values

    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity

    feature_matrix = np.array([mean_features[lang] for lang in languages])
    similarity_matrix = cosine_similarity(feature_matrix)

    return similarity_matrix, languages


def plot_similarity_heatmap(df, output_path):
    """Plot similarity heatmap"""
    similarity_matrix, languages = compute_similarity_matrix(df)

    lang_names = {'mandarin': 'Mandarin', 'tibetan': 'Tibetan', 'uyghur': 'Uyghur'}
    labels = [lang_names[l] for l in languages]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=12)

    ax.set_title('Inter-Language Audio Feature Similarity\n(Higher = More similar channel characteristics)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Similarity heatmap saved to: {output_path}")
    plt.show()


def generate_statistics_report(df, output_path):
    """Generate detailed statistics report"""
    lang_names = {'mandarin': 'Mandarin', 'tibetan': 'Tibetan', 'uyghur': 'Uyghur'}

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Audio Quality and Channel Analysis Report\n")
        f.write("音频质量与信道特征分析报告\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("1. SIGNAL-TO-NOISE RATIO (SNR) ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for lang in ['mandarin', 'tibetan', 'uyghur']:
            snr_values = df[df['language'] == lang]['snr'].values
            f.write(f"\n{lang_names[lang]}:\n")
            f.write(f"  Mean SNR: {np.mean(snr_values):.2f} dB\n")
            f.write(f"  Median SNR: {np.median(snr_values):.2f} dB\n")
            f.write(f"  Std SNR: {np.std(snr_values):.2f} dB\n")
            f.write(f"  Min SNR: {np.min(snr_values):.2f} dB\n")
            f.write(f"  Max SNR: {np.max(snr_values):.2f} dB\n")

            # Count low SNR samples
            low_snr_count = np.sum(snr_values < 10)
            f.write(f"  Samples with SNR < 10dB: {low_snr_count} ({low_snr_count/len(snr_values)*100:.2f}%)\n")

        # Noise level comparison
        f.write("\n\n2. BACKGROUND NOISE LEVEL ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for lang in ['mandarin', 'tibetan', 'uyghur']:
            noise_values = df[df['language'] == lang]['noise_level'].values
            f.write(f"\n{lang_names[lang]}:\n")
            f.write(f"  Mean noise level: {np.mean(noise_values):.6f}\n")
            f.write(f"  Median noise level: {np.median(noise_values):.6f}\n")
            f.write(f"  Std noise level: {np.std(noise_values):.6f}\n")

        # Spectral features
        f.write("\n\n3. SPECTRAL CHARACTERISTICS\n")
        f.write("-" * 80 + "\n")
        spectral_features = [
            ('spectral_centroid_mean', 'Spectral Centroid'),
            ('spectral_bandwidth_mean', 'Spectral Bandwidth'),
            ('spectral_rolloff_mean', 'Spectral Rolloff'),
            ('high_freq_ratio', 'High Frequency Ratio (>4kHz)')
        ]

        for feature, feature_name in spectral_features:
            f.write(f"\n{feature_name}:\n")
            for lang in ['mandarin', 'tibetan', 'uyghur']:
                values = df[df['language'] == lang][feature].values
                f.write(f"  {lang_names[lang]}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")

        # Channel similarity
        f.write("\n\n4. INTER-LANGUAGE CHANNEL SIMILARITY\n")
        f.write("-" * 80 + "\n")
        similarity_matrix, languages = compute_similarity_matrix(df)

        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i < j:
                    f.write(f"{lang_names[lang1]} vs {lang_names[lang2]}: ")
                    f.write(f"Similarity = {similarity_matrix[i, j]:.4f}\n")

        # Key findings
        f.write("\n\n5. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")

        # Check if Tibetan has higher noise
        tibetan_noise = df[df['language'] == 'tibetan']['noise_level'].mean()
        mandarin_noise = df[df['language'] == 'mandarin']['noise_level'].mean()
        uyghur_noise = df[df['language'] == 'uyghur']['noise_level'].mean()

        if tibetan_noise > max(mandarin_noise, uyghur_noise) * 1.2:
            f.write("⚠️ WARNING: Tibetan dataset has significantly higher noise levels!\n")

        # Check Tibetan-Uyghur similarity
        tib_idx = languages.index('tibetan')
        uyg_idx = languages.index('uyghur')
        tib_uyg_sim = similarity_matrix[tib_idx, uyg_idx]

        if tib_uyg_sim > 0.95:
            f.write("⚠️ WARNING: Tibetan and Uyghur show very high channel similarity!\n")
            f.write("   This suggests they may have similar recording conditions/equipment.\n")
            f.write("   The model may be learning channel features instead of language features.\n")

        # Check SNR differences
        tibetan_snr = df[df['language'] == 'tibetan']['snr'].mean()
        mandarin_snr = df[df['language'] == 'mandarin']['snr'].mean()
        uyghur_snr = df[df['language'] == 'uyghur']['snr'].mean()

        if tibetan_snr < min(mandarin_snr, uyghur_snr) * 0.8:
            f.write("⚠️ WARNING: Tibetan dataset has significantly lower SNR!\n")

    print(f"Statistics report saved to: {output_path}")


def find_problematic_samples(df, output_path, n_samples=20):
    """Find samples with extreme feature values"""
    tibetan_df = df[df['language'] == 'tibetan'].copy()

    # Sort by different criteria
    criteria = [
        ('snr', 'Lowest SNR (Noisiest)', False),
        ('noise_level', 'Highest Noise Level', True),
        ('spectral_flatness_mean', 'Highest Spectral Flatness (Most noise-like)', True),
        ('high_freq_ratio', 'Unusual High Frequency Content', True)
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Problematic Tibetan Audio Samples\n")
        f.write("=" * 80 + "\n\n")

        for feature, title, ascending in criteria:
            f.write(f"\n{title}:\n")
            f.write("-" * 80 + "\n")

            sorted_df = tibetan_df.sort_values(by=feature, ascending=ascending)
            top_samples = sorted_df.head(n_samples)

            for idx, (_, row) in enumerate(top_samples.iterrows(), 1):
                f.write(f"{idx}. {row['filename']}\n")
                f.write(f"   {feature}: {row[feature]:.4f}\n")
                if 'snr' in row:
                    f.write(f"   SNR: {row['snr']:.2f} dB\n")

    print(f"Problematic samples list saved to: {output_path}")


def main():
    """Main function"""
    # Configuration
    data_folder = '../data/processed'
    output_dir = Path('../results/audio_quality_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze dataset (use sample_limit for quick testing, None for full analysis)
    print("=" * 80)
    print("Audio Quality and Channel Analysis")
    print("=" * 80)

    sample_limit = 100  # Set to e.g. 100 for quick testing
    df, file_info = analyze_dataset(data_folder, sample_limit)

    # Save raw data
    df.to_csv(output_dir / 'audio_features.csv', index=False)
    print(f"\nRaw features saved to: {output_dir / 'audio_features.csv'}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)

    plot_snr_comparison(df, output_dir / 'snr_comparison.png')
    plot_noise_level_comparison(df, output_dir / 'noise_analysis.png')
    plot_spectral_features_comparison(df, output_dir / 'spectral_features.png')
    plot_channel_features_pca(df, output_dir / 'channel_pca.png')
    plot_channel_features_tsne(df, output_dir / 'channel_tsne.png')
    plot_similarity_heatmap(df, output_dir / 'similarity_heatmap.png')

    # Generate reports
    print("\n" + "=" * 80)
    print("Generating reports...")
    print("=" * 80)

    generate_statistics_report(df, output_dir / 'statistics_report.txt')
    find_problematic_samples(df, output_dir / 'problematic_samples.txt')

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    print("\nKey outputs:")
    print("  - audio_features.csv: Raw feature data")
    print("  - statistics_report.txt: Detailed statistical analysis")
    print("  - problematic_samples.txt: List of low-quality samples")
    print("  - *.png: Various visualization plots")


if __name__ == '__main__':
    main()
