import datasets
import yaml
import os
from datasets import DownloadManager, DatasetInfo, AudioClassification


class CMDCDATASET(datasets.GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description="CMDC DataSet",
            features=datasets.Features({
                "audio": datasets.Audio(sampling_rate=16_000, mono=True),
                "label": datasets.ClassLabel(num_classes=2, names=['nd', 'd']),
            }),
            task_templates=[
                AudioClassification(audio_column='audio', label_column='label'),
            ]
        )

    def _split_generators(self, dl_manager: DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": 'train',
                    "config_path": './config/dataset/CMDC.yaml'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": 'test',
                    "config_path": './config/dataset/CMDC.yaml'
                }
            )
        ]

    def _generate_examples(self, split, config_path):
        with open(config_path) as f:
            mes = yaml.load(f.read(), Loader=yaml.FullLoader)
        base = 0 if split == 'train' else 2
        root = mes['root']
        fold = mes['fold_i']
        prefixes = ['MDD', 'HC']
        files = mes["Folders"][fold]
        for i, prefix in enumerate(prefixes):
            path_1 = os.path.join(root, prefix)
            datas = files[base + i]
            for d_id in datas:
                path_2 = os.path.join(path_1, prefix + str(d_id).rjust(2, '0'))
                file_list = os.listdir(path_2)
                file_list = list(filter(lambda x: x.endswith('.wav'), file_list))
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
