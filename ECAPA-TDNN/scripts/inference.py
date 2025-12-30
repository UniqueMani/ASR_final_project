"""
Language Identification Inference Script
Predict language for new audio files
"""

import torch
import torchaudio

# Fix torchaudio compatibility issue with SpeechBrain
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = list_audio_backends

# Set soundfile as default backend to avoid FFmpeg dependency
import os as _os
_os.environ.setdefault('TORCHAUDIO_BACKEND', 'soundfile')

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
        # Load audio
        signal = sb.dataio.dataio.read_audio(audio_path)
        signal = signal.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features
            feats = self.feature_extractor(signal)
            feats = self.normalizer(feats, torch.tensor([1.0]).to(self.device))

            # Extract embedding
            embedding = self.embedding_model(feats, torch.tensor([1.0]).to(self.device))

            # Classification
            output = self.classifier(embedding)
            log_probs = torch.nn.functional.log_softmax(output, dim=1)
            probs = torch.exp(log_probs)

            # Get prediction result
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_id].item()

            # Decode label
            predicted_lang = self.label_encoder.decode_torch(torch.tensor([pred_id]))[0]

            # Get probabilities for all languages
            all_probs = {}
            for i in range(len(self.label_encoder)):
                lang = self.label_encoder.decode_torch(torch.tensor([i]))[0]
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
