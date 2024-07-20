import datasets
import yaml
import os
import numpy as np
from datasets import DownloadManager, DatasetInfo, AudioClassification
class EATDATASET(datasets.GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        features = datasets.Features({
            'audio': datasets.Audio(sampling_rate=16000, mono=True),
            'label': datasets.ClassLabel(num_classes=2, names=['nd', 'd']),
            "uid": datasets.Value('int64'),
        })

        return datasets.DatasetInfo(
            description='EDAIC DataSet',
            features=features,
            task_templates=[
                AudioClassification(audio_column='audio', label_column='label'),
            ]
        )

    def _split_generators(self, dl_manager: DownloadManager):
        train_generator = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                'split': 'train',
                'config_path': './config/dataset/DAIC.yaml',
            }
        )
        test_generator = datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                'split': 'dev',
                'config_path': './config/dataset/DAIC.yaml',
            }
        )
        val_generator = datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={
                'split': 'dev',
                'config_path': './config/dataset/DAIC.yaml',
            }
        )
        return [
            train_generator,
            test_generator,
            val_generator
        ]

    def _generate_examples(self, split, config_path):
        with open(config_path) as f:
            mes = yaml.load(f.read(), Loader=yaml.FullLoader)
        root = mes['root']
        audio_dir = split
        data_path = os.path.join(root, audio_dir)
        audio_list = os.listdir(data_path)
        for id, det in enumerate(audio_list):
            sid = id
            d_id = int(det.split('_')[0])
            label = int(det.split('_')[2].replace('.wav', ''))
            path = os.path.join(root, audio_dir, det)
            uid = label * 100000 + d_id
            with open(path, 'rb') as f:
                yield sid, {
                    'uid': uid,
                    'audio': {
                        'bytes': f.read(),
                        'path': path,
                    },
                    'label': label
                }
