from pathlib import Path
import datasets
import yaml
from datasets import DownloadManager
from datasets.tasks import AutomaticSpeechRecognition, AudioClassification
import os
from sklearn.model_selection import KFold
import numpy as np

class EATDDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="speech", version=VERSION, description="Data for speech recognition"),
    ]

    def _info(self):
        features = datasets.Features(
            {
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
                    "config_path": './config/dataset/EATD.yaml',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "config_path": './config/dataset/EATD.yaml',
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
        kf = KFold(n_splits=3)
        ulist = os.listdir(os.path.join(root,'HC'))
        dlist = os.listdir(os.path.join(root, 'MDD'))

        trainU = []
        trainD = []
        testU = []
        testD = []

        for t, v in kf.split(ulist):
            trainU.append(t)
            testU.append(v)

        for t,v in kf.split(dlist):
            trainD.append(t)
            testD.append(v)

        if split == 'train':
            dataset1 = trainD[fold]
            dataset2 = trainU[fold]
        else:
            dataset1 = testD[fold]
            dataset2 = testU[fold]

        dataset = [dataset1,dataset2]
        prefixes = ['MDD', 'HC']
        for i, prefix in enumerate(prefixes):
            path_1 = os.path.join(root, prefix)
            datas = dataset[i]
            for d_id in datas:
                path_2 = os.path.join(path_1, prefix + str(d_id).rjust(3, '0'))
                file_list = os.listdir(path_2)
                file_list = list(filter(lambda x: x.endswith('out.wav'), file_list))
                for w_name in file_list:
                    path_3 = os.path.join(path_2, w_name)
                    # print(path_3)
                    sid = f"{prefix}_{d_id}_{w_name.replace('.wav', '')}"
                    with open(path_3, 'rb') as f:
                        yield sid, {
                            "audio": {
                                'bytes': f.read(),
                                'path': path_3,
                            },
                            "label": i
                        }
