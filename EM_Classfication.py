import numpy as np
from scipy.stats import multivariate_normal


def initialize_parameters(data, num_components):
    """初始化均值、协方差和混合权重"""
    np.random.seed(0)
    n, d = data.shape
    means = data[np.random.choice(n, num_components, replace=False)]
    covariances = [np.eye(d) * 0.1 for _ in range(num_components)]  # 初始化时添加小的正值
    weights = np.ones(num_components) / num_components
    return means, covariances, weights

def e_step(data, means, covariances, weights):
    """E步：计算每个组件对每个数据点的责任"""
    num_components = len(weights)
    responsibilities = np.zeros((data.shape[0], num_components))

    for k in range(num_components):
        rv = multivariate_normal(means[k], covariances[k], allow_singular=True)  # 允许奇异矩阵
        responsibilities[:, k] = weights[k] * rv.pdf(data)

    responsibilities_sum = responsibilities.sum(axis=1)[:, np.newaxis]
    responsibilities /= responsibilities_sum

    return responsibilities


def m_step(data, responsibilities):
    """M步：更新均值、协方巧矩阵和混合权重"""
    num_components = responsibilities.shape[1]
    n, d = data.shape
    means = np.zeros((num_components, d))
    covariances = np.zeros((num_components, d, d))
    weights = np.zeros(num_components)

    for k in range(num_components):
        Nk = responsibilities[:, k].sum()
        means[k] = (responsibilities[:, k][:, np.newaxis] * data).sum(axis=0) / Nk
        diff = data - means[k]
        cov_k = (responsibilities[:, k][:, np.newaxis, np.newaxis] * np.matmul(diff[:, :, np.newaxis],
                                                                               diff[:, np.newaxis, :])).sum(axis=0) / Nk
        covariances[k] = cov_k + np.eye(d) * 0.01  # 添加对角正则化以避免奇异性
        weights[k] = Nk / n

    return means, covariances, weights


def log_likelihood(data, means, covariances, weights):
    """计算数据的对数似然"""
    ll = 0
    for k, (mean, cov, weight) in enumerate(zip(means, covariances, weights)):
        rv = multivariate_normal(mean, cov)
        ll += weight * rv.pdf(data)
    return np.log(ll).sum()


def em_algorithm(data, num_components, num_iters):
    """执行EM算法"""
    means, covariances, weights = initialize_parameters(data, num_components)

    for i in range(num_iters):
        responsibilities = e_step(data, means, covariances, weights)
        means, covariances, weights = m_step(data, responsibilities)
        if i % 10 == 0:
            print("Iteration", i, "log-likelihood:", log_likelihood(data, means, covariances, weights))

    return means, covariances, weights


# 示例数据：生成一个具有728维特征的数据集
data = np.random.randn(100, 2)  # 100个样本，每个样本728维

# 执行EM算法
num_components = 2  # 二维分类
num_iters = 20000
means, covariances, weights = em_algorithm(data, num_components, num_iters)
