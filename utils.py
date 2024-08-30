import os.path
from enum import Enum
import evaluate
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
import torch.nn.functional as F
from random import randint
from datasets import load_metric
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score as roc_auc_score2

import matplotlib.pyplot as plt
import random
from transformers import TrainerCallback


def init_seed(seed):
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


class StrategyType(Enum):
    ADAPTER = "adapter"
    LORA = "lora"
    IA3 = "ia3"
    BITFIT = "bitfit"
    CLASSFIER = "classfier"


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


def eval_metrics(eval_predict, optimal_threshold=None):
    predictions, labels = eval_predict
    hidden_embeddings = predictions[1]
    predictions = predictions[0]
    predictions = softmax(predictions, axis=1)
    y_scores = predictions[..., 1]
    fpr, tpr, thresholds = roc_curve(labels, y_scores)
    # roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score2(labels, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] if optimal_threshold is None else optimal_threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(labels, y_pred)
    results = {'accuracy': float(accuracy)}
    precision = precision_score(labels, y_pred)
    results.update({'precision': float(precision)})
    recall = recall_score(labels, y_pred)
    results.update({'recall': float(recall)})
    f1 = f1_score(labels, y_pred)
    results.update({'f1': float(f1)})
    results.update({'roc_auc': float(roc_auc)})
    return results, hidden_embeddings, predictions, labels


class EvaluateMetrics:
    def __init__(self, save_path):
        self.max_f1 = 0
        self.save_path = save_path
        self.eval_function = eval_metrics

    def __call__(self, eval_predict):
        results, hidden_embeddings, predictions, labels = self.eval_function(eval_predict, 0.5)
        if results['f1'] >= self.max_f1:
            self.max_f1 = results['f1']
            self.save_best_results(results)
            self.save_max_f1_hidden_embeddings(hidden_embeddings)
            self.save_best_predictions(predictions)
            self.save_output_as_csv(predictions, labels)
        return results

    def save_max_f1_hidden_embeddings(self, hidden_embeddings):
        np.save(os.path.join(self.save_path, 'best_hidden_embeddings'), hidden_embeddings)

    def save_best_results(self, results):
        np.save(os.path.join(self.save_path, 'best_results.npy'), results)

    def save_best_predictions(self, predictions):
        np.save(os.path.join(self.save_path, 'best_predictions'), predictions)

    def save_output_as_csv(self, predictions, labels):
        dataFrame = pd.DataFrame(np.vstack([labels, predictions[..., 1]]).transpose())
        dataFrame.to_csv(os.path.join(self.save_path, 'best.csv'), sep=',', header=None)


def find_best_optimal_threshold(dir, step=1):
    # 假设y_true是真实的标签，y_scores是模型预测为正类的概率
    data = np.loadtxt(dir, delimiter=',', dtype='float')
    y_true = (data[:, 0] > 100000).astype(dtype=np.int32)
    y_scores = data[:, 2]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score2(y_true,y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        f'step-{step} acc-{round(accuracy, 3)} pre-{round(precision, 3)} rec-{round(recall, 3)} f1-{round(f1, 3)} auroc-{round(roc_auc, 3)}'
    )
    plt.legend(loc="lower right")
    plt.show()
    return f1


def find_optimal_threshold(dir, step, optimal_threshold=None, pos_label=1):
    # 假设y_true是真实的标签，y_scores是模型预测为正类的概率
    data = np.loadtxt(dir, delimiter=',', dtype='float')
    y_true = data[:, 1]
    y_scores = data[:, 2]
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] if optimal_threshold is None else optimal_threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        f'step-{step} acc-{round(accuracy, 3)} pre-{round(precision, 3)} rec-{round(recall, 3)} f1-{round(f1, 4)} auroc-{round(roc_auc, 4)}'
    )
    plt.legend(loc="lower right")
    plt.show()
    return f1, optimal_threshold


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = os.path.join(save_path, 'best_model')
        self.best_metric = float(0)  # 初始化为正无穷大，适用于最小化的指标，比如loss

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics")
        if metrics is None:
            return

        # 假设我们以验证集的损失作为评估标准
        current_metric = metrics.get("eval_f1")
        if current_metric is None:
            return

        if current_metric > self.best_metric:
            self.best_metric = current_metric
            # 保存模型到指定路径
            self.save_model(kwargs["model"])

    def save_model(self, model):
        model.save_pretrained(self.save_path)
        print(f"Model saved to {self.save_path}")


def process_classification_results(file_path) -> pd.DataFrame:
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Extract label, identity ID, and suffix using division and modulus
    data['label'] = data.iloc[:, 0] // 100000
    data['identity_id'] = (data.iloc[:, 0] // 100) % 1000
    data['suffix'] = data.iloc[:, 0] % 100

    # Convert the probability columns to numeric
    data['Category 2'] = data.iloc[:, 2].astype(float)

    # Drop duplicate columns if any
    data = data.loc[:, ~data.columns.duplicated()]

    # Group by identity ID and label, then compute the average of the second category's probability values
    grouped_data = data.groupby(['identity_id', 'label'])['Category 2'].mean().reset_index()

    return grouped_data


def find_dataframe_optimal_threshold(data, step, optimal_threshold=None):
    # 假设y_true是真实的标签，y_scores是模型预测为正类的概率
    y_true = data['label'].values
    y_scores = data['Category 2'].values
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] if optimal_threshold == None else optimal_threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        f'step-{step} acc-{round(accuracy, 3)} pre-{round(precision, 3)} rec-{round(recall, 3)} f1-{round(f1, 4)} auroc-{round(roc_auc, 4)}'
    )
    plt.legend(loc="lower right")
    plt.show()

    # 将结果转换为ndarray并返回
    results = np.array([f1, optimal_threshold, accuracy, precision, recall, roc_auc])
    return results
