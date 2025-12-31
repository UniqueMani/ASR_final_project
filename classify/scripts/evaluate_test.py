"""
Test Set Evaluation and Visualization Script
Evaluate model on all test samples and visualize results
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

import speechbrain as sb
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from collections import defaultdict

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LanguageIdentifier:
    """Language Identifier for batch evaluation"""

    def __init__(self, model_path, config_path):
        """Initialize"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        with open(config_path) as f:
            self.hparams = load_hyperpyyaml(f)

        # Load model components
        self.feature_extractor = self.hparams['compute_features']
        self.normalizer = self.hparams['mean_var_norm']
        self.embedding_model = self.hparams['embedding_model']
        self.classifier = self.hparams['classifier']

        # Load checkpoint
        checkpointer = sb.utils.checkpoints.Checkpointer(
            checkpoints_dir=Path(model_path) / 'save',
            recoverables={
                'embedding_model': self.embedding_model,
                'classifier': self.classifier,
                'normalizer': self.normalizer,
            }
        )
        checkpointer.recover_if_possible()

        # Set to evaluation mode
        self.embedding_model.eval()
        self.classifier.eval()
        self.embedding_model.to(self.device)
        self.classifier.to(self.device)

        # Load label encoder
        label_encoder_path = Path(model_path) / 'save' / 'label_encoder.txt'
        self.label_encoder = sb.dataio.encoder.CategoricalEncoder()
        self.label_encoder.load(label_encoder_path)

        print(f"Model loaded successfully! Device: {self.device}")

    def predict(self, audio_path):
        """Predict language for a single audio file"""
        # Load audio using soundfile directly to avoid torchcodec
        try:
            import soundfile as sf
            signal, sr = sf.read(audio_path)
            # Convert to tensor
            signal = torch.FloatTensor(signal)

            # Handle different audio shapes
            if signal.dim() == 1:
                # Mono audio: (samples,) -> keep as is
                pass
            elif signal.dim() == 2:
                # Stereo/multi-channel: (samples, channels) -> convert to mono
                # Take mean across channels (axis 1)
                signal = torch.mean(signal, dim=1)

        except ImportError:
            # If soundfile not available, try torchaudio with soundfile backend
            try:
                signal, sr = torchaudio.load(audio_path, backend="soundfile")
                # torchaudio returns (channels, samples)
                if signal.shape[0] > 1:
                    signal = torch.mean(signal, dim=0)
                else:
                    signal = signal.squeeze(0)
            except:
                # Final fallback to SpeechBrain's read_audio
                signal = sb.dataio.dataio.read_audio(audio_path)

        # Ensure signal is 1D at this point
        while signal.dim() > 1:
            signal = signal.squeeze()

        # Add batch dimension: (samples,) -> (1, samples)
        signal = signal.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features
            feats = self.feature_extractor(signal)

            # Prepare length tensor
            lens = torch.ones(1, device=self.device)
            feats = self.normalizer(feats, lens)

            # Extract embedding
            embedding = self.embedding_model(feats, lens)

            # Classification
            output = self.classifier(embedding)

            # Squeeze to remove extra dimensions: [1, 1, 3] -> [1, 3]
            output = output.squeeze(1) if output.dim() == 3 else output

            log_probs = torch.nn.functional.log_softmax(output, dim=1)
            probs = torch.exp(log_probs)

            # Get prediction result
            pred_id_tensor = torch.argmax(probs, dim=1)
            pred_id = pred_id_tensor.item()

            # Decode label
            pred_tensor = torch.tensor([pred_id])
            decoded = self.label_encoder.decode_torch(pred_tensor)

            # Handle different return types
            if isinstance(decoded, torch.Tensor):
                if decoded.numel() == 1:
                    predicted_lang = str(decoded.item())
                else:
                    predicted_lang = str(decoded[0])
            elif isinstance(decoded, list):
                predicted_lang = decoded[0]
            else:
                predicted_lang = str(decoded)

            confidence = probs[0, pred_id].item()

            # Get all probabilities
            all_probs = {}
            for i in range(len(self.label_encoder)):
                lang_tensor = torch.tensor([i])
                decoded_lang = self.label_encoder.decode_torch(lang_tensor)
                if isinstance(decoded_lang, torch.Tensor):
                    if decoded_lang.numel() == 1:
                        lang = str(decoded_lang.item())
                    else:
                        lang = str(decoded_lang[0])
                elif isinstance(decoded_lang, list):
                    lang = decoded_lang[0]
                else:
                    lang = str(decoded_lang)
                all_probs[lang] = probs[0, i].item()

        return predicted_lang, confidence, all_probs


def evaluate_test_set(identifier, data_folder):
    """Evaluate on all test samples"""
    results = {
        'true_labels': [],
        'pred_labels': [],
        'confidences': [],
        'file_paths': [],
        'all_probs': []
    }

    languages = ['mandarin', 'tibetan', 'uyghur']
    test_folder = Path(data_folder) / 'test'

    print("\nEvaluating test set...")
    for lang in languages:
        lang_folder = test_folder / lang
        audio_files = sorted(list(lang_folder.glob('*.wav')))

        print(f"\nProcessing {lang} ({len(audio_files)} files)...")
        for audio_file in tqdm(audio_files):
            try:
                pred_lang, confidence, all_probs = identifier.predict(str(audio_file))

                results['true_labels'].append(lang)
                results['pred_labels'].append(pred_lang)
                results['confidences'].append(confidence)
                results['file_paths'].append(str(audio_file))
                results['all_probs'].append(all_probs)

            except Exception as e:
                print(f"\nError processing {audio_file}: {e}")

    return results


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix"""
    # Language names
    lang_names = {
        'mandarin': 'Mandarin',
        'tibetan': 'Tibetan',
        'uyghur': 'Uyghur'
    }

    labels = ['mandarin', 'tibetan', 'uyghur']
    display_labels = [lang_names[l] for l in labels]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_labels, yticklabels=display_labels,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    # Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=display_labels, yticklabels=display_labels,
                ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.show()


def plot_accuracy_by_language(y_true, y_pred, output_path):
    """Plot accuracy by language"""
    lang_names = {
        'mandarin': 'Mandarin',
        'tibetan': 'Tibetan',
        'uyghur': 'Uyghur'
    }

    languages = ['mandarin', 'tibetan', 'uyghur']
    accuracies = []

    for lang in languages:
        # Filter samples for this language
        indices = [i for i, true_label in enumerate(y_true) if true_label == lang]
        if len(indices) > 0:
            lang_true = [y_true[i] for i in indices]
            lang_pred = [y_pred[i] for i in indices]
            acc = accuracy_score(lang_true, lang_pred)
            accuracies.append(acc * 100)
        else:
            accuracies.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    bars = ax.bar([lang_names[l] for l in languages], accuracies, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Language Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Per-language accuracy plot saved to: {output_path}")
    plt.show()


def plot_confidence_distribution(results, output_path):
    """Plot confidence distribution"""
    lang_names = {
        'mandarin': 'Mandarin',
        'tibetan': 'Tibetan',
        'uyghur': 'Uyghur'
    }

    # Separate correct and incorrect predictions
    correct_confidences = defaultdict(list)
    incorrect_confidences = defaultdict(list)

    for true_label, pred_label, confidence in zip(results['true_labels'],
                                                    results['pred_labels'],
                                                    results['confidences']):
        if true_label == pred_label:
            correct_confidences[true_label].append(confidence)
        else:
            incorrect_confidences[true_label].append(confidence)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    languages = ['mandarin', 'tibetan', 'uyghur']
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

    for idx, (lang, color) in enumerate(zip(languages, colors)):
        ax = axes[idx]

        # Plot correct predictions
        if correct_confidences[lang]:
            ax.hist(correct_confidences[lang], bins=20, alpha=0.7,
                   color='green', label='Correct', edgecolor='black')

        # Plot incorrect predictions
        if incorrect_confidences[lang]:
            ax.hist(incorrect_confidences[lang], bins=20, alpha=0.7,
                   color='red', label='Incorrect', edgecolor='black')

        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{lang_names[lang]}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Confidence Distribution by Language', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confidence distribution plot saved to: {output_path}")
    plt.show()


def plot_error_analysis(results, output_path):
    """Plot detailed error analysis"""
    lang_names = {
        'mandarin': 'Mandarin',
        'tibetan': 'Tibetan',
        'uyghur': 'Uyghur'
    }

    # Count misclassification patterns
    error_matrix = defaultdict(lambda: defaultdict(int))

    for true_label, pred_label in zip(results['true_labels'], results['pred_labels']):
        if true_label != pred_label:
            error_matrix[true_label][pred_label] += 1

    # Create error summary
    fig, ax = plt.subplots(figsize=(12, 6))

    languages = ['mandarin', 'tibetan', 'uyghur']
    x = np.arange(len(languages))
    width = 0.25

    # For each true language, show where it was misclassified
    for idx, true_lang in enumerate(languages):
        errors = []
        for pred_lang in languages:
            if true_lang != pred_lang:
                errors.append(error_matrix[true_lang][pred_lang])
            else:
                errors.append(0)

        # Get positions for grouped bars
        positions = x + (idx - 1) * width
        ax.bar(positions, errors, width, label=f'True: {lang_names[true_lang]}', alpha=0.7)

    ax.set_xlabel('Predicted as', fontsize=12)
    ax.set_ylabel('Number of Errors', fontsize=12)
    ax.set_title('Error Analysis: Misclassification Patterns', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([lang_names[l] for l in languages])
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Error analysis plot saved to: {output_path}")
    plt.show()


def save_detailed_report(results, output_path):
    """Save detailed text report"""
    lang_names = {
        'mandarin': 'Mandarin',
        'tibetan': 'Tibetan',
        'uyghur': 'Uyghur'
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Language Identification - Test Set Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        # Overall metrics
        overall_acc = accuracy_score(results['true_labels'], results['pred_labels'])
        f.write(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
        f.write(f"Total Samples: {len(results['true_labels'])}\n\n")

        # Classification report
        f.write("Classification Report:\n")
        f.write("-" * 80 + "\n")
        report = classification_report(
            results['true_labels'],
            results['pred_labels'],
            target_names=[lang_names[l] for l in ['mandarin', 'tibetan', 'uyghur']],
            digits=4
        )
        f.write(report)
        f.write("\n")

        # Per-language statistics
        f.write("\nPer-Language Statistics:\n")
        f.write("-" * 80 + "\n")

        for lang in ['mandarin', 'tibetan', 'uyghur']:
            indices = [i for i, true_label in enumerate(results['true_labels']) if true_label == lang]
            if len(indices) > 0:
                lang_true = [results['true_labels'][i] for i in indices]
                lang_pred = [results['pred_labels'][i] for i in indices]
                lang_conf = [results['confidences'][i] for i in indices]

                correct = sum([1 for t, p in zip(lang_true, lang_pred) if t == p])
                acc = correct / len(lang_true)
                avg_conf = np.mean(lang_conf)

                f.write(f"\n{lang_names[lang]}:\n")
                f.write(f"  Total samples: {len(lang_true)}\n")
                f.write(f"  Correct predictions: {correct}\n")
                f.write(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
                f.write(f"  Average confidence: {avg_conf:.4f} ({avg_conf*100:.2f}%)\n")

        # Error analysis
        f.write("\n\nError Analysis:\n")
        f.write("-" * 80 + "\n")

        errors = [(true, pred, conf, path)
                  for true, pred, conf, path in zip(results['true_labels'],
                                                      results['pred_labels'],
                                                      results['confidences'],
                                                      results['file_paths'])
                  if true != pred]

        f.write(f"Total errors: {len(errors)}\n\n")

        if errors:
            f.write("Error details:\n")
            for idx, (true, pred, conf, path) in enumerate(errors[:20], 1):  # Show first 20 errors
                f.write(f"{idx}. {Path(path).name}\n")
                f.write(f"   True: {lang_names[true]}, Predicted: {lang_names[pred]}, Confidence: {conf:.4f}\n")

            if len(errors) > 20:
                f.write(f"\n... and {len(errors) - 20} more errors\n")

    print(f"Detailed report saved to: {output_path}")


def main():
    """Main function"""
    # Configuration
    model_path = '../models/ecapa_lang_id'
    config_path = './train_config.yaml'
    data_folder = '../data/processed'
    output_dir = Path('../results/evaluation')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize identifier
    print("Loading model...")
    identifier = LanguageIdentifier(model_path, config_path)

    # Evaluate test set
    results = evaluate_test_set(identifier, data_folder)

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    overall_acc = accuracy_score(results['true_labels'], results['pred_labels'])
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Total Samples: {len(results['true_labels'])}")

    # Count per language
    from collections import Counter
    true_counts = Counter(results['true_labels'])
    print("\nSamples per language:")
    for lang, count in sorted(true_counts.items()):
        print(f"  {lang}: {count}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)

    plot_confusion_matrix(
        results['true_labels'],
        results['pred_labels'],
        output_dir / 'confusion_matrix.png'
    )

    plot_accuracy_by_language(
        results['true_labels'],
        results['pred_labels'],
        output_dir / 'accuracy_by_language.png'
    )

    plot_confidence_distribution(
        results,
        output_dir / 'confidence_distribution.png'
    )

    plot_error_analysis(
        results,
        output_dir / 'error_analysis.png'
    )

    # Save detailed report
    save_detailed_report(
        results,
        output_dir / 'evaluation_report.txt'
    )

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    main()
