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
        })

        return datasets.DatasetInfo(
            description='EDAIC DataSet',
            features=features,
            task_templates=[
                AudioClassification(audio_column='audio',label_column='label'),
            ]
        )

    def _split_generators(self, dl_manager: DownloadManager):
        train_generator = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                'split': 'train',
                'config_path': './config/dataset/EDAIC.yaml',
            }
        )
        test_generator = datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                'split': 'test',
                'config_path': './config/dataset/EDAIC.yaml',
            }
        )
        val_generator = datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={
                'split': 'dev',
                'config_path': './config/dataset/EDAIC.yaml',
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
        audio_dir = mes['audio_file']
        data_path = os.path.join(root, 'labels', f'{split}_split.csv')
        with open(data_path) as f:
            data_index = np.loadtxt(f, delimiter=',', skiprows=1, usecols=[0, 2, 3]).astype(dtype=int)
        for id, det in enumerate(data_index):
            template = f'{det[0]}_P/{det[0]}_AUDIO.wav'
            sid = f'{det[0]}'
            path = os.path.join(root, audio_dir, template)
            with open(path, 'rb') as f:
                yield sid, {
                    'audio': {
                        'bytes': f.read(),
                        'path': path,
                    },
                    'label': det[1],
                }
