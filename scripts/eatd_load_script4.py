import os
import re
import numpy as np
import yaml
import glob
import datasets
from datasets import DownloadManager
from datasets.tasks import AudioClassification
from sklearn.model_selection import KFold


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

        ulist = glob.glob(os.path.join(root, 'HC', '*.wav'))
        dlist = glob.glob(os.path.join(root, 'MDD', '*.wav'))

        trainU = []
        trainD = []
        testU = []
        testD = []

        for t, v in kf.split(ulist):
            trainU.append(t)
            testU.append(v)

        for t, v in kf.split(dlist):
            trainD.append(t)
            testD.append(v)

        if split == 'train':
            dataset1 = np.array(dlist)[trainD[fold]]
            dataset2 = np.array(ulist)[trainU[fold]]
        else:
            dataset1 = np.array(dlist)[testD[fold]]
            dataset2 = np.array(ulist)[testU[fold]]

        dataset = np.concatenate([dataset2, dataset1])

        pattern = re.compile(r'[_\\]')

        for id, file in enumerate(dataset):

            parts = pattern.split(str(file))
            i = parts[2]
            label = int(parts[4])

