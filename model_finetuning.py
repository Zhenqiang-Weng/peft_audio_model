import os
import json
import torch
import datasets
import transformers
import yaml

from datasets import (
    load_dataset,
)

from peft import (
    TaskType,
    get_peft_model,
    IA3Config,
    LoraConfig,
)

from transformers import (
    AutoFeatureExtractor,
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    DataCollatorWithPadding
)

from trainer import Trainer

from wav2vec2 import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from hubert import HubertForSequenceClassification, HubertModel
from wavlm import WavLMForSequenceClassification, WavLMModel
from whisper import WhisperForAudioClassification, WhisperModel

from utils import (
    random_subsample,
    StrategyType,
    init_seed,
    EvaluateMetrics,
    SaveBestModelCallback,
)

from dataclasses import dataclass, field

from typing import Any, Dict, List, Union
from enum import Enum


@dataclass
class DataArguments:
    max_length_seconds: float = field(
        default=10,
        metadata={"help": ""},
    )

    sample_rate: int = field(
        default=16000,
        metadata={"help": ""},
    )

    dataset_script_path: str = field(
        default="scripts/eatd3_load_script.py",
        metadata={"help": " "},
    )

    robustness_verification: bool = field(
        default=False,
        metadata={"help": ""},
    )

    dataset_script_path_for_train: str = field(
        default="scripts/edaic_load_scripts2.py",
        metadata={"help": " "},
    )

    dataset_script_path_for_test: str = field(
        default="scripts/eatd_load_script5.py",
        metadata={"help": " "},
    )

    cache_file_path: str = field(
        default="F:/cache/audio_pretained_model",
        metadata={"help": " "},
    )


@dataclass
class ModelArguments:
    model_path: str = field(
        default="models/hubert-base-ls960",
        metadata={"help": " "},
    )
    resume_from_checkpoint: str = field(
        default=False,
        metadata={"help": " "},
    )


@dataclass
class StrategyParameters:
    # target_modules = ['q_proj', 'v_proj'],
    target_modules = ['attention.k_proj', 'attention.v_proj', 'feed_forward.output_dense'],
    feedforward_modules = ['feed_forward.output_dense']


@dataclass
class StrategyArguments:
    strategy: StrategyType = field(
        default=StrategyType.BITFIT,
        metadata={"help": "Fine-tuning methods"},
    )
    strategy_parameters: StrategyParameters = field(
        default=StrategyParameters,
        metadata={"help": "Fine-tuning methods"},
    )
    fp16: bool = field(
        default=False,
        metadata={"help": " "},
    )
    num_train_epochs: int = field(
        default=25,
        metadata={"help": " "},
    )


@dataclass
class HyperparameterArguments:
    label_smoothing_factor: float = field(
        default=0,
        metadata={"help": " "},
    )
    learning_rate: float = field(
        default=0.0001,
        metadata={"help": " "},
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": " "},
    )


@dataclass
class RecordArguments:
    do_train: bool = field(
        default=True,
        metadata={"help": " "},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": " "},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": " "},
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={"help": " "},
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": " "},
    )
    metric_for_best_model: str = field(
        default="eval_f1",
        metadata={"help": " "},
    )
    save_steps: int = field(
        default=100,
        metadata={"help": " "},
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # feature名字需要按basemodel的输入名称修改，待定
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 此处尚未写label以及其他！！！！
        # batch["labels"] = labels

        return batch


def main():
    parser = HfArgumentParser(
        (
            DataArguments,
            ModelArguments,
            HyperparameterArguments,
            StrategyArguments,
            RecordArguments
        )
    )
    data_args, model_args, hyperparameter_args, strategy_args, record_args = parser.parse_args_into_dataclasses()

    output_dir = 'F:/cache/audio_pretained_model/checkpoints/' + strategy_args.strategy.value + '/'

    if data_args.robustness_verification:
        output_dir += "robustness_verification/" + \
                      data_args.dataset_script_path_for_train.split('/')[-1].replace('.py', '') + \
                      '/' + \
                      data_args.dataset_script_path_for_test.split('/')[-1].replace('.py', '') + \
                      '/' + \
                      model_args.model_path.split('/')[-1]

    else:
        output_dir += data_args.dataset_script_path.split('/')[-1].replace('.py', '') + '/' + \
                      model_args.model_path.split('/')[-1]

        dataInformationPath = data_args.dataset_script_path.split('/')[-1].split('_')[0].upper() + '.yaml'

        with open('config/dataset/' + dataInformationPath) as f:
            information = yaml.load(f.read(), Loader=yaml.FullLoader)
        output_dir += ('/' + str(information['fold_i'])) if 'fold_i' in information.keys() else ''

    train_arguments = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        push_to_hub=False,
        remove_unused_columns=False,

        do_train=record_args.do_train,
        do_eval=record_args.do_eval,
        logging_steps=record_args.logging_steps,
        evaluation_strategy=record_args.evaluation_strategy,
        eval_steps=record_args.eval_steps,
        metric_for_best_model=record_args.metric_for_best_model,
        save_steps=record_args.save_steps,

        fp16=strategy_args.fp16,
        num_train_epochs=strategy_args.num_train_epochs,

        # label_smoothing_factor=hyperparameter_args.label_smoothing_factor,
        learning_rate=hyperparameter_args.learning_rate,
        per_device_train_batch_size=hyperparameter_args.per_device_train_batch_size,

        load_best_model_at_end=True,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_path)

    # model
    if 'hubert' in model_args.model_path:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_args.model_path,
            ignore_mismatched_sizes=True,
            cache_dir='F:/cache/audio_pretained_model/cache/models',
        )
    elif 'wav2vec' in model_args.model_path:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_args.model_path,
            ignore_mismatched_sizes=True,
            cache_dir='F:/cache/audio_pretained_model/cache/models',
        )
    elif 'wavlm' in model_args.model_path:
        model = WavLMForSequenceClassification.from_pretrained(
            model_args.model_path,
            ignore_mismatched_sizes=True,
            cache_dir='F:/cache/audio_pretained_model/cache/models',
        )
    elif 'whisper' in model_args.model_path:
        model = WhisperForAudioClassification.from_pretrained(
            model_args.model_path,
            ignore_mismatched_sizes=True,
            cache_dir='F:/cache/audio_pretained_model/cache/models',
        )

    print(model)
    for name, para in model.named_parameters():
        print(name)

    # PEFT config
    if strategy_args.strategy.value == "bitfit":
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
    elif strategy_args.strategy.value in ["ia3", "lora"]:
        if strategy_args.strategy.value == "ia3":
            peft_config = IA3Config(
                task_type=TaskType.SEQ_CLS,
                target_modules=strategy_args.strategy_parameters.target_modules[0],
                feedforward_modules=strategy_args.strategy_parameters.feedforward_modules[0],
            )
        elif strategy_args.strategy.value == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=strategy_args.strategy_parameters.target_modules[0],
            )
        model = get_peft_model(model, peft_config)
        print(model)
        model.print_trainable_parameters()

    elif strategy_args.strategy.value == "adapter":
        print("set add_adapter=True in the configuration")
        # adapter ft
        total_params = 0
        trainable_params = 0
        for name, parameters in model.named_parameters():
            total_params += parameters.numel()
            if 'adapter' in name or 'classifier' in name:
                trainable_params += parameters.numel()
            else:
                parameters.requires_grad = False
        print(
            f"total parameters:{total_params},trainable parameters:{trainable_params},r:{trainable_params / total_params}"
        )
        print(model)

    for name, parameters in model.named_parameters():
        if 'classifier' in name or 'layer_norm' in name:
            parameters.requires_grad = True

    for name, para in model.named_parameters():
        print(name, para.requires_grad)

    # data
    cache_dir = "F:/cache/audio_pretained_model/cache"
    with open('config/dataset/' + dataInformationPath) as f:
        information = yaml.load(f.read(), Loader=yaml.FullLoader)
    cache_dir += ('/' + str(information['fold_i'])) if 'fold_i' in information.keys() else ''

    train_dataset = None
    test_dataset = None
    if data_args.robustness_verification:
        if record_args.do_train:

            train_dataset = load_dataset(

                data_args.dataset_script_path_for_train,

                trust_remote_code=True,
                cache_dir=cache_dir
            ).shuffle()['train']
        if record_args.do_eval:
            test_dataset = load_dataset(
                data_args.dataset_script_path_for_test,
                trust_remote_code=True,
                cache_dir=cache_dir
            ).shuffle()['test']
    else:
        dataset = load_dataset(
            data_args.dataset_script_path,
            trust_remote_code=True,
            cache_dir=cache_dir
        ).shuffle()
        if record_args.do_train:
            train_dataset = dataset['train']
        if record_args.do_eval:
            test_dataset = dataset['test']

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
        # output_batch = {model_input_name: inputs.get(model_input_name), 'labels': list(batch['label']),
        #                 'attention_mask': inputs['attention_mask']}
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
        # output_batch = {model_input_name: inputs.get(model_input_name), 'labels': list(batch['label']),
        #                 'attention_mask': inputs['attention_mask']}
        output_batch = {model_input_name: inputs.get(model_input_name), 'labels': list(batch['label'])}

        return output_batch

    remove_columns = train_dataset.column_names if train_dataset else test_dataset.column_names

    if 'uid' in remove_columns:
        remove_columns.remove('uid')

    if train_arguments.do_train:
        train_dataset = train_dataset.map(
            train_transforms,
            batched=True,
            remove_columns=remove_columns,
        )

    if train_arguments.do_eval:
        test_dataset = test_dataset.map(
            test_transforms,
            batched=True,
            remove_columns=remove_columns,
        )

    evalMetrics = EvaluateMetrics(output_dir)
    save_best_model_callback = SaveBestModelCallback(save_path=output_dir)

    trainer = Trainer(
        model=model,
        args=train_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=feature_extractor,
        compute_metrics=evalMetrics,
        callbacks=[save_best_model_callback],
        # data_collator=
    )

    trainer.train(
        resume_from_checkpoint=model_args.resume_from_checkpoint,
    )

    trainer.evaluate()


if __name__ == '__main__':
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
    # os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'

    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # init_seed(0)

    main()
