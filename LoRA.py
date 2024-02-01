import torch
import transformers
import datasets
import peft
from datasets import load_dataset
from utils import *
import evaluate

from transformers import (
    Wav2Vec2ForSequenceClassification,
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    DataCollatorWithPadding
)
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
        default="models/chinese-hubert-large",
        metadata={"help": " "},
    )


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_args.model_path, ignore_mismatched_sizes=True)

    print(model)

    for name,_ in model.named_parameters():
        print(name)


if __name__ == '__main__':
    main()






