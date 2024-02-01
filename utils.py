import evaluate
import numpy as np
from random import randint

from datasets import load_metric


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset: random_offset + sample_length]


acc_metric = load_metric('./metrics/accuracy/accuracy.py', trust_remote_code=True)
recall_metric = load_metric("./metrics/recall/recall.py", trust_remote_code=True)
precision_metric = load_metric("./metrics/precision/precision.py", trust_remote_code=True)
f1_metric = load_metric("./metrics/f1/f1.py", trust_remote_code=True)
roc_auc_score = load_metric("./metrics/roc_auc/roc_auc.py", trust_remote_code=True)


# acc_metric = evaluate.load("accuracy")
# recall_metric = evaluate.load("recall")
# f1_metirc = evaluate.load("f1")
# roc_auc_score = evaluate.load("roc_auc", "multiclass")
# precision_metric = evaluate.load("precision")


def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions_ = predictions.argmax(axis=-1)

    results = acc_metric.compute(references=labels, predictions=predictions_)
    recall = recall_metric.compute(references=labels, predictions=predictions_)
    precision = precision_metric.compute(references=labels, predictions=predictions_)
    f1 = f1_metric.compute(references=labels, predictions=predictions_)
    auc = roc_auc_score.compute(references=labels, prediction_scores=predictions[..., 1])

    results.update(recall)
    results.update(precision)
    results.update(f1)
    results.update(auc)

    return results
