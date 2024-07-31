import os
import re
import numpy as np
import yaml
import glob
import datasets
from datasets import DownloadManager
from datasets.tasks import AudioClassification


class EATDDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="speech", version=VERSION, description="Data for speech recognition"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "uid": datasets.Value('int64'),
                "audio": datasets.Audio(sampling_rate=16_000),
                "label": datasets.ClassLabel(num_classes=2, names=['0', '1']),
            }
        )

        return datasets.DatasetInfo(
            description='EATD DataSet',
            features=features,
            supervised_keys=None,
            task_templates=[
                AudioClassification(audio_column='audio', label_column='label'),
            ]
        )

    def _split_generators(self, dl_manager: DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "config_path": './config/dataset/EATD5.yaml',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "config_path": './config/dataset/EATD5.yaml',
                },
            ),
        ]

    def _generate_examples(
            self, split, config_path
    ):
        with open(config_path) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        fold = config['fold_i']
        root = config['root']
        files = glob.glob(os.path.join(root, str(fold), split, '*.wav'))
        pattern = re.compile(r'[_\\]')

        for id, i in enumerate(files):
            parts = pattern.split(i)
            # print(parts)
            if split == 'train':
                label = int(parts[-2])
                uid = int(parts[-4])
                suffix = int(parts[-1].replace('.wav', ''))
            else:
                label = int(parts[-1].replace('.wav', ''))
                uid = int(parts[-3])
                suffix = 0

            uuid = label * 100000 + uid * 100 + suffix
            with open(i, 'rb') as f:
                yield id, {
                    "uid": uuid,
                    "audio": {
                        'bytes': f.read(),
                        'path': i,
                    },
                    "label": label
                }
