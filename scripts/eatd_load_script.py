from pathlib import Path
import datasets
from datasets.tasks import AutomaticSpeechRecognition, AudioClassification
import os


class EATDDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="speech", version=VERSION, description="Data for speech recognition"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "audio_raw": datasets.Audio(sampling_rate=16_000),
                "audio": datasets.Audio(sampling_rate=16_000),
                "id": datasets.Value("string"),
                "raw_sds": datasets.Value("uint8"),
                "sds_score": datasets.Value("float"),
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

    def _split_generators(self, dl_manager):
        if hasattr(dl_manager, 'manual_dir') and dl_manager.manual_dir is not None:
            data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "split": "train",
                        "data_dir": data_dir,
                    },
                ),
            ]

    def _generate_examples(
            self, split, data_dir
    ):
        basepath = Path(data_dir)
        prefix = "t"
        for dir in basepath.glob(f"{prefix}_*"):
            base_id = dir.name
            with open(str(dir / "label.txt")) as labelf:
                label = labelf.read().strip()
                if label.endswith(".0"):
                    raw_sds = int(label[:-2])
                else:
                    raw_sds = int(label)
            with open(str(dir / "new_label.txt")) as labelf:
                new_label = labelf.read().strip()
                sds_score = float(new_label)
            for polarity in ["neutral", "negative", "positive"]:
                raw_audio = dir / f"{polarity}.wav"
                proc_audio = dir / f"{polarity}_out.wav"
                with open(raw_audio, "rb") as rawf, open(proc_audio, "rb") as procf:
                    sid = f"{base_id}_{polarity}"
                    yield sid, {
                        "audio_raw": {
                            "bytes": rawf.read(),
                            "path": str(raw_audio),
                        },
                        "audio": {
                            "bytes": procf.read(),
                            "path": str(proc_audio),
                        },
                        "id": sid,
                        "raw_sds": raw_sds,
                        "sds_score": sds_score,
                        "label": sds_score >= 53,
                    }
