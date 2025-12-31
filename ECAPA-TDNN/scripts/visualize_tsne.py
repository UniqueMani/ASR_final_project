"""
t-SNE Visualization Script
Extract embeddings from test set and plot t-SNE dimensionality reduction visualization
"""

# IMPORTANT: Set environment variables BEFORE importing any audio libraries
import os
os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '1'

import torch
import torchaudio

# Fix torchaudio compatibility issue with SpeechBrain
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = list_audio_backends

# Force set soundfile backend (for older versions)
if hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend("soundfile")

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import speechbrain as sb
from speechbrain.lobes.features import Fbank
from hyperpyyaml import load_hyperpyyaml
import seaborn as sns

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EmbeddingExtractor:
    """Extract audio embeddings"""

    def __init__(self, model_path, config_path):
        """Initialize"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        with open(config_path) as f:
            self.hparams = load_hyperpyyaml(f)

        # Load model
        self.feature_extractor = self.hparams['compute_features']
        self.normalizer = self.hparams['mean_var_norm']
        self.embedding_model = self.hparams['embedding_model']

        # Load checkpoint
        checkpointer = sb.utils.checkpoints.Checkpointer(
            checkpoints_dir=Path(model_path) / 'save',
            recoverables={
                'embedding_model': self.embedding_model,
                'normalizer': self.normalizer,
            }
        )
        checkpointer.recover_if_possible()

        self.embedding_model.eval()
        self.embedding_model.to(self.device)

    def extract_embedding(self, audio_path):
        """Extract embedding from a single audio file"""
        # Load audio using soundfile directly to avoid torchcodec
        try:
            import soundfile as sf
            signal, sr = sf.read(audio_path)
            signal = torch.FloatTensor(signal).unsqueeze(0)
        except ImportError:
            # If soundfile not available, try torchaudio with soundfile backend
            try:
                signal, sr = torchaudio.load(audio_path, backend="soundfile")
            except:
                # Final fallback to SpeechBrain's read_audio
                signal = sb.dataio.dataio.read_audio(audio_path)
                signal = signal.unsqueeze(0)

        # Ensure mono channel
        if signal.dim() > 1 and signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        elif signal.dim() == 1:
            signal = signal.unsqueeze(0)

        signal = signal.to(self.device)

        with torch.no_grad():
            # Extract features
            feats = self.feature_extractor(signal)
            feats = self.normalizer(feats, torch.tensor([1.0]).to(self.device))

            # Extract embedding
            embedding = self.embedding_model(feats, torch.tensor([1.0]).to(self.device))

        return embedding.squeeze().cpu().numpy()


def collect_embeddings(data_folder, extractor):
    """Collect embeddings from all test set samples"""
    embeddings = []
    labels = []
    languages = ['mandarin', 'tibetan', 'uyghur']

    test_folder = Path(data_folder) / 'test'

    for lang in languages:
        lang_folder = test_folder / lang
        audio_files = list(lang_folder.glob('*.wav'))

        print(f"Extracting {lang} embeddings...")
        for audio_file in tqdm(audio_files):
            try:
                emb = extractor.extract_embedding(str(audio_file))
                embeddings.append(emb)
                labels.append(lang)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    return np.array(embeddings), labels


def plot_tsne(embeddings, labels, output_path):
    """Plot t-SNE visualization"""
    print("\nPerforming t-SNE dimensionality reduction...")

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Prepare plot
    plt.figure(figsize=(12, 10))

    # Language to color mapping
    lang_colors = {
        'mandarin': '#FF6B6B',
        'tibetan': '#4ECDC4',
        'uyghur': '#FFE66D'
    }

    lang_names = {
        'mandarin': 'Mandarin',
        'tibetan': 'Tibetan',
        'uyghur': 'Uyghur'
    }

    # Plot scatter points
    for lang in ['mandarin', 'tibetan', 'uyghur']:
        mask = np.array(labels) == lang
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=lang_colors[lang],
            label=lang_names[lang],
            alpha=0.6,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )

    plt.title('Language Identification Embedding t-SNE Visualization', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved to: {output_path}")

    # Show plot
    plt.show()


def plot_tsne_with_density(embeddings, labels, output_path):
    """Plot t-SNE visualization with density"""
    print("\nPerforming t-SNE dimensionality reduction (density version)...")

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Language to color mapping
    lang_colors = {
        'mandarin': '#FF6B6B',
        'tibetan': '#4ECDC4',
        'uyghur': '#FFE66D'
    }

    lang_names = {
        'mandarin': 'Mandarin',
        'tibetan': 'Tibetan',
        'uyghur': 'Uyghur'
    }

    # Left plot: scatter plot
    for lang in ['mandarin', 'tibetan', 'uyghur']:
        mask = np.array(labels) == lang
        ax1.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=lang_colors[lang],
            label=lang_names[lang],
            alpha=0.6,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )

    ax1.set_title('t-SNE Scatter Plot', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Right plot: density plot
    for lang in ['mandarin', 'tibetan', 'uyghur']:
        mask = np.array(labels) == lang
        ax2.hexbin(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            gridsize=30,
            alpha=0.5,
            cmap='YlOrRd' if lang == 'mandarin' else ('GnBu' if lang == 'tibetan' else 'YlGn'),
            label=lang_names[lang]
        )

    ax2.set_title('t-SNE Density Plot', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Language Identification Embedding Visualization Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    density_output = output_path.replace('.png', '_density.png')
    plt.savefig(density_output, dpi=300, bbox_inches='tight')
    print(f"Density plot saved to: {density_output}")

    plt.show()


def main():
    """Main function"""
    # Configuration paths
    model_path = '../models/ecapa_lang_id'
    config_path = './train_config.yaml'
    data_folder = '../data/processed'
    output_path = '../results/tsne_visualization.png'
    cache_path = '../results/embeddings_cache.npz'

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Try to load cached embeddings
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        cache_data = np.load(cache_path, allow_pickle=True)
        embeddings = cache_data['embeddings']
        labels = cache_data['labels'].tolist()
        print("Cached embeddings loaded successfully!")
    else:
        # Initialize extractor
        print("Loading model...")
        extractor = EmbeddingExtractor(model_path, config_path)

        # Extract embeddings
        print("\nCollecting test set embeddings...")
        embeddings, labels = collect_embeddings(data_folder, extractor)

        # Save embeddings to cache
        print(f"\nSaving embeddings to cache: {cache_path}")
        np.savez(cache_path, embeddings=embeddings, labels=np.array(labels))
        print("Cache saved successfully!")

    print(f"\nExtraction complete!")
    print(f"Total samples: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Statistics for each language
    from collections import Counter
    label_counts = Counter(labels)
    print("\nSamples per language:")
    for lang, count in label_counts.items():
        print(f"  {lang}: {count}")

    # Plot t-SNE visualization
    plot_tsne(embeddings, labels, output_path)
    plot_tsne_with_density(embeddings, labels, output_path)

    print("\nVisualization complete!")
    print(f"\nNote: Embeddings are cached at {cache_path}")
    print("To re-extract embeddings, delete this cache file.")


if __name__ == '__main__':
    main()
