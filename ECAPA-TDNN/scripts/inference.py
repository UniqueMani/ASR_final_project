"""
Language Identification Inference Script
Predict language for new audio files
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


class LanguageIdentifier:
    """Language Identifier"""

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
            pred_id = pred_id_tensor.item()  # Convert single-element tensor to Python int
            confidence = probs[0, pred_id].item()  # Use item() directly

            # Decode label
            pred_tensor = torch.tensor([pred_id])
            decoded = self.label_encoder.decode_torch(pred_tensor)

            # Handle different return types
            if isinstance(decoded, torch.Tensor):
                if decoded.numel() == 1:
                    predicted_lang = str(decoded.item())
                else:
                    # Multi-element tensor - assume first element is the label
                    predicted_lang = str(decoded[0])
            elif isinstance(decoded, list):
                predicted_lang = decoded[0]
            else:
                predicted_lang = str(decoded)

            # Get probabilities for all languages
            all_probs = {}
            for i in range(len(self.label_encoder)):
                lang_tensor = torch.tensor([i])
                decoded_lang = self.label_encoder.decode_torch(lang_tensor)

                # Handle different return types
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


def main():
    """Main function"""
    import sys

    # Configuration paths
    model_path = '../models/ecapa_lang_id'
    config_path = './train_config.yaml'

    # Initialize identifier
    identifier = LanguageIdentifier(model_path, config_path)

    # Language name mapping
    lang_names = {
        'mandarin': 'Mandarin',
        'tibetan': 'Tibetan',
        'uyghur': 'Uyghur'
    }

    if len(sys.argv) > 1:
        # Command line mode
        audio_path = sys.argv[1]
        print(f"\nAnalyzing audio: {audio_path}")

        predicted_lang, confidence, all_probs = identifier.predict(audio_path)

        print(f"\nPredicted language: {lang_names.get(predicted_lang, predicted_lang)}")
        print(f"Confidence: {confidence:.2%}")
        print("\nAll language probabilities:")
        for lang, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang_names.get(lang, lang)}: {prob:.2%}")
    else:
        # Interactive mode
        print("=" * 60)
        print("Language Identification System")
        print("=" * 60)
        print("Enter audio file path (type 'quit' to exit):")

        while True:
            audio_path = input("\n> ").strip()

            if audio_path.lower() in ['quit', 'exit', 'q']:
                break

            if not Path(audio_path).exists():
                print("Error: File not found!")
                continue

            try:
                predicted_lang, confidence, all_probs = identifier.predict(audio_path)

                print(f"\nPredicted language: {lang_names.get(predicted_lang, predicted_lang)}")
                print(f"Confidence: {confidence:.2%}")
                print("\nAll language probabilities:")
                for lang, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {lang_names.get(lang, lang)}: {prob:.2%}")

            except Exception as e:
                print(f"Error: {e}")

    print("\nThank you for using!")


if __name__ == '__main__':
    main()
