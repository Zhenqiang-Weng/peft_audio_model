import os

import torch
import transformers
import datasets
import yaml
from peft import TaskType, LoraConfig, get_peft_model
from datasets import load_dataset
from utils import *

from transformers import (
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from dataclasses import dataclass, field

from whisper import WhisperForAudioClassification


@dataclass
class DataArguments:
    max_length_seconds: float = field(
        default=30,
        metadata={"help": ""},
    )
    sample_rate: int = field(
        default=16000,
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
    dataInformationPath = data_args.dataset_script_path.split('/')[-1].split('_')[0].upper() + '.yaml'
    with open('config/dataset/' + dataInformationPath) as f:
        information = yaml.load(f.read(), Loader=yaml.FullLoader)
    output_dir += ('/' + str(information['fold_i'])) if 'fold_i' in information.keys() else ''

    train_arguments = TrainingArguments(
        output_dir='./checkpoints/LoRA/' + output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        fp16=False,
        # gradient_accumulation_steps=8,
        label_smoothing_factor=0.1,
        learning_rate=2e-4,
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


    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=['q_proj', 'v_proj'],
    )
    model = get_peft_model(model, lora_config)

    for name, parameters in model.named_parameters():
        if 'classifier' in name:
            parameters.requires_grad = True

    print(model)
    model.print_trainable_parameters()

    # data
    dataset = load_dataset(
        data_args.dataset_script_path,
        trust_remote_code=True,
        cache_dir='./cache'
    ).shuffle()
    # print(dataset)

    model_input_name = feature_extractor.model_input_names[0]

    def train_transforms(batch):
        subsampled_wavs = []
        for audio in batch['audio']:
            wav = random_subsample(
                audio['array'], max_length=data_args.max_length_seconds, sample_rate=data_args.sample_rate
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
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
    # os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'
    main()
