"""
Train Language Identification Model using SpeechBrain ECAPA-TDNN
"""

import os
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
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.lobes.features import Fbank
from speechbrain.nnet.losses import LogSoftmaxWrapper
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm


class LanguageIdBrain(sb.Brain):
    """Language Identification Training Class"""

    def compute_forward(self, batch, stage):
        """Forward propagation"""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Extract features
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, wav_lens)

        # ECAPA-TDNN embedding
        embeddings = self.modules.embedding_model(feats, wav_lens)

        # Classification
        outputs = self.modules.classifier(embeddings)

        # Ensure outputs has correct shape [batch_size, num_classes]
        # Remove extra dimension if present (e.g., [32, 1, 3] -> [32, 3])
        if outputs.dim() == 3 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        elif outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)

        return outputs, embeddings

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss"""
        outputs, embeddings = predictions
        lang_ids, _ = batch.lang_id_encoded

        # Flatten lang_ids to 1D tensor [batch_size] and ensure it's long type
        if lang_ids.dim() > 1:
            lang_ids = lang_ids.view(-1)
        lang_ids = lang_ids.long()

        # Debug: print shapes on first batch
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG - outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
            print(f"DEBUG - lang_ids shape: {lang_ids.shape}, dtype: {lang_ids.dtype}")
            print(f"DEBUG - outputs sample: {outputs[0]}")
            print(f"DEBUG - lang_ids sample: {lang_ids[:5]}")
            self._debug_printed = True

        # Compute classification loss
        # CrossEntropyLoss expects: input [batch_size, num_classes], target [batch_size]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, lang_ids)

        if stage != sb.Stage.TRAIN:
            # Compute accuracy
            self.error_metrics.append(batch.id, outputs, lang_ids)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Operations at the beginning of each stage"""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Operations at the end of each stage"""
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage != sb.Stage.TRAIN:
            stage_stats["error_rate"] = self.error_metrics.summarize("average")

        # Log stats
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save checkpoint
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                min_keys=["error_rate"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def create_datasets(data_folder, hparams):
    """Create datasets"""
    train_data = []
    valid_data = []
    test_data = []

    # Language label mapping
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    languages = ['mandarin', 'tibetan', 'uyghur']

    # Collect training data
    train_folder = Path(data_folder) / 'train'
    for lang in languages:
        lang_folder = train_folder / lang
        for audio_file in lang_folder.glob('*.wav'):
            train_data.append({
                'id': audio_file.stem,
                'wav': str(audio_file),
                'lang_id': lang
            })

    # Collect test data
    test_folder = Path(data_folder) / 'test'
    for lang in languages:
        lang_folder = test_folder / lang
        for audio_file in lang_folder.glob('*.wav'):
            test_data.append({
                'id': audio_file.stem,
                'wav': str(audio_file),
                'lang_id': lang
            })

    # Split validation set from training set (10%)
    np.random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)
    valid_data = train_data[split_idx:]
    train_data = train_data[:split_idx]

    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(valid_data)} samples")
    print(f"Test set: {len(test_data)} samples")

    # Encode labels
    label_encoder.update_from_iterable(languages)

    # Create dynamic datasets
    datasets = {}
    for name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
        # Convert list to dictionary format required by DynamicItemDataset
        # Note: 'id' key is reserved, so we only include other fields
        data_dict = {item['id']: {'wav': item['wav'], 'lang_id': item['lang_id']} for item in data}
        dataset = DynamicItemDataset(data_dict)

        # Add audio pipeline
        @sb.utils.data_pipeline.takes("wav")
        @sb.utils.data_pipeline.provides("sig")
        def audio_pipeline(wav):
            # Use soundfile directly to avoid torchaudio backend issues
            import soundfile as sf
            audio_data, sample_rate = sf.read(wav)
            sig = torch.from_numpy(audio_data).float()
            return sig

        # Add label pipeline
        @sb.utils.data_pipeline.takes("lang_id")
        @sb.utils.data_pipeline.provides("lang_id", "lang_id_encoded")
        def label_pipeline(lang_id):
            yield lang_id
            # Encode label and ensure it's a tensor with shape [1]
            lang_id_encoded = label_encoder.encode_label_torch(lang_id)
            # Ensure it has shape [1] for proper batching
            if lang_id_encoded.dim() == 0:
                lang_id_encoded = lang_id_encoded.unsqueeze(0)
            yield lang_id_encoded

        dataset.add_dynamic_item(audio_pipeline)
        dataset.add_dynamic_item(label_pipeline)
        dataset.set_output_keys(["id", "sig", "lang_id_encoded"])

        datasets[name] = dataset

    return datasets, label_encoder


def dataio_prep(hparams):
    """Prepare data loaders"""
    datasets, label_encoder = create_datasets(
        hparams["data_folder"],
        hparams
    )

    # Save label encoder
    label_encoder_file = Path(hparams["save_folder"]) / "label_encoder.txt"
    label_encoder.save(label_encoder_file)

    return datasets


def main(hparams_file="train_config.yaml"):
    """Main function"""

    # Load hyperparameters
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides={},
    )

    # Prepare data
    datasets = dataio_prep(hparams)

    # Initialize Brain
    lang_id_brain = LanguageIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=hparams["checkpointer"],
    )

    # Train
    lang_id_brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    lang_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
