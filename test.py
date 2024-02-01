import transformers
import datasets
from datasets import load_dataset
import torch
import evaluate
from transformers import (
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    DataCollatorWithPadding
)
from wav2vec2 import Wav2Vec2ForSequenceClassification
from utils import *
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    max_length_seconds: float = field(
        default=1,
        metadata={"help": ""},
    )
    dataset_script_path: str = field(
        default="./scripts/cmdc_load_scripts.py",
        metadata={"help": ""},
    )
    cache_file_path: str = field(
        default="./cache",
        metadata={"help": ""},
    )


@dataclass
class ModelArguments:
    model_path: str = field(
        default="./models/mms-lid-126",
        metadata={"help": " "},
    )


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()
    train_arguments = TrainingArguments(
        output_dir='./checkpoints',
        do_train=True,
        fp16=True,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_args.model_path, ignore_mismatched_sizes=True)
    dataset = load_dataset(data_args.dataset_script_path)

    model_input_name = feature_extractor.model_input_names[0]

    def train_transforms(batch):
        subsampled_wavs = []
        for audio in batch['audio']:
            wav = random_subsample(
                audio['array'], max_length=data_args.max_length_seconds, sample_rate=16000
            )
            subsampled_wavs.append(wav)
        inputs = feature_extractor(
            subsampled_wavs,
            sample_rate=feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt"
        )
        output_batch = {model_input_name: inputs.get(model_input_name), 'labels': list(batch['label'])}
        return output_batch

    def test_transforms(batch):
        wavs = [audio["array"] for audio in batch['audio']]
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate, return_tensors='pt')
        output_batch = {model_input_name: inputs.get(model_input_name), 'labels': list(batch['label'])}
        return output_batch

    if train_arguments.do_train:
        train_dataset = dataset['train'].map(
            train_transforms,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

    if train_arguments.do_eval:
        test_dataset = dataset['test'].map(
            test_transforms,
            batched=True,
            remove_columns=dataset['test'].column_names
        )

    trainer = Trainer(
        model=model,
        args=train_arguments,
        train_dataset=train_dataset if train_arguments.do_train else None,
        eval_dataset=test_dataset if train_arguments.do_eval else None,
        tokenizer=feature_extractor
    )

    trainer.train()


if __name__ == '__main__':
    main()