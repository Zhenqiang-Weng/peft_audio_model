import torch
import transformers
import datasets
from datasets import (
    load_dataset,
)
from transformers import (
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

import numpy as np

from whisper import WhisperForAudioClassification

from utils import (
    random_subsample,
    eval_metric
)

from typing import Any, Dict, List, Union
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    max_length_seconds: float = field(
        default=30,
        metadata={"help": ""},
    )
    dataset_script_path: str = field(
        default="scripts/daic_load_scripts.py",
        metadata={"help": " "},
    )
    cache_file_path: str = field(
        default="./cache",
        metadata={"help": " "},
    )


@dataclass
class ModelArguments:
    model_path: str = field(
        default="models/whisper-tiny",
        metadata={"help": " "},
    )
    resume_from_checkpoint: str = field(
        default=False,
        metadata={"help": " "},
    )


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()

    output_dir = data_args.dataset_script_path.split('/')[-1].replace('.py', '') + '/' + \
                 model_args.model_path.split('/')[-1]

    train_arguments = TrainingArguments(
        output_dir='./checkpoints/BitFit/' + output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        fp16=False,
        # gradient_accumulation_steps=8,
        label_smoothing_factor=0.1,
        learning_rate=2e-5,
        per_device_train_batch_size=4,

        logging_steps=10,
        num_train_epochs=500,
        evaluation_strategy='steps',
        eval_steps=100,
        metric_for_best_model="roc_auc",
        save_steps=50000,
        push_to_hub=False,
        remove_unused_columns=False,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_path)

    # model

    model = WhisperForAudioClassification.from_pretrained(
        model_args.model_path,
        ignore_mismatched_sizes=True,
        cache_dir='./cache/models',
    )


    dataset = load_dataset(
        data_args.dataset_script_path,
        trust_remote_code=True,
        cache_dir='./cache',
    ).shuffle()

    model_input_name = feature_extractor.model_input_names[0]

    print(model)

    # BitFit ft
    total_params = 0
    trainable_params = 0
    for name, parameters in model.named_parameters():
        total_params += parameters.numel()
        if 'bias' in name or 'classifier' in name:
            parameters.requires_grad = True
            trainable_params += parameters.numel()
        else:
            parameters.requires_grad = False
    print(
        f"total parameters:{total_params},trainable parameters:{trainable_params},r:{trainable_params / total_params}"
    )

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
        subsampled_wavs = []
        for audio in batch['audio']:
            wav = random_subsample(
                audio['array'], max_length=data_args.max_length_seconds, sample_rate=16000
            )
            subsampled_wavs.append(wav)

        # wavs = [audio["array"] for audio in batch['audio']]

        inputs = feature_extractor(
            subsampled_wavs,
            sampling_rate=feature_extractor.sampling_rate,
            padding=True,
            return_tensors='pt'
        )
        output_batch = {model_input_name: inputs.get(model_input_name), 'labels': list(batch['label'])}
        return output_batch

    remove_columns = dataset['train'].column_names
    if 'uid' in remove_columns:
        remove_columns.remove('uid')

    if train_arguments.do_train:
        train_dataset = dataset['train'].map(
            train_transforms,
            batched=True,
            remove_columns=remove_columns,
        )

    if train_arguments.do_eval:
        test_dataset = dataset['test'].map(
            test_transforms,
            batched=True,
            remove_columns=remove_columns,
        )


    trainer = Trainer(
        model=model,
        args=train_arguments,
        train_dataset=train_dataset if train_arguments.do_train else None,
        eval_dataset=test_dataset if train_arguments.do_eval else None,
        tokenizer=feature_extractor,
        compute_metrics=eval_metric,

    )

    # print(model)

    trainer.train(
        resume_from_checkpoint=model_args.resume_from_checkpoint,

    )


if __name__ == '__main__':
    main()
