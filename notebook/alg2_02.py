#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
4 维自回归无梯度优化 Demo
变量: (x, y1, y2, y3)
目标: 演示 Algorithm 2 在 4 维函数上的优化过程，
      并画出每个 autoregressive step 后的能量变化。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/home/cxm/NotoSansCJKsc-Regular.otf')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
# font size
plt.rcParams['font.size'] = 24

# ---------- 1. 定义 4 维能量函数 ----------
def energy_full(ys):
    """
    ys: shape (N, 4), 每一行是 [x, y1, y2, y3]
    返回: shape (N,) 的能量
    """
    x  = ys[:, 0]
    y1 = ys[:, 1]
    y2 = ys[:, 2]
    y3 = ys[:, 3]
    return ((x - 1.0) ** 2
            + (y1 + 2.0) ** 2
            + (y2 - 0.5) ** 2
            + (y3 + 1.0) ** 2)


def energy_prefix(ys, j):
    """
    前缀能量 E^j(x, y[:j])
    j = 0: 只看 x
    j = 1: 看 (x, y1)
    j = 2: 看 (x, y1, y2)
    j = 3: 看全部 (x, y1, y2, y3)
    """
    x  = ys[:, 0]
    y1 = ys[:, 1]
    y2 = ys[:, 2]
    y3 = ys[:, 3]

    if j == 0:
        return (x - 1.0) ** 2
    elif j == 1:
        return (x - 1.0) ** 2 + (y1 + 2.0) ** 2
    elif j == 2:
        return (x - 1.0) ** 2 + (y1 + 2.0) ** 2 + (y2 - 0.5) ** 2
    elif j == 3:
        return energy_full(ys)
    else:
        raise ValueError("j must be 0,1,2,3 for 4D example")


# ---------- 2. 自回归无梯度优化 ----------
def autoregressive_opt_4d(
    num_samples=100,
    N_iters=15,
    sigma_init=1.0,
    K=0.5,
    bounds=(-5, 5),
    seed=0,
):
    np.random.seed(seed)
    low, high = bounds
    dim = 4

    # 初始化样本: 均匀分布在区间 [low, high]
    ys = np.random.uniform(low, high, size=(num_samples, dim))
    sigma = sigma_init

    # 记录每个 step 的平均 & 最佳能量（用 full energy 测）
    mean_E_history = []
    best_E_history = []

    # 记录初始状态的能量
    E0 = energy_full(ys)
    mean_E_history.append(E0.mean())
    best_E_history.append(E0.min())

    # 外层: iteration
    for it in range(N_iters):
        for j in range(dim):
            # 1) 用前缀能量 E^j 计算权重
            E_prefix = energy_prefix(ys, j)
            negE = -E_prefix
            negE -= np.max(negE)               # 数值稳定
            w = np.exp(negE)
            p = w / np.sum(w)

            # 2) 按 p 重采样（整体样本级别）
            idx = np.random.choice(len(ys), size=len(ys), replace=True, p=p)
            ys = ys[idx]

            # 3) 只在第 j 维加高斯噪声
            noise = np.zeros_like(ys)
            noise[:, j] = sigma * np.random.randn(len(ys))
            ys = ys + noise
            ys = np.clip(ys, low, high)

            # 4) 在 full energy 下评估当前样本集合
            E_full = energy_full(ys)
            mean_E_history.append(E_full.mean())
            best_E_history.append(E_full.min())

        # 5) 每轮结束后缩小噪声
        sigma *= K
        print(f"The current noise size is {sigma}")

    return np.array(mean_E_history), np.array(best_E_history)


# ---------- 3. 跑一次并画图 ----------
if __name__ == "__main__":
    mean_E, best_E = autoregressive_opt_4d()

    steps = np.arange(len(mean_E))  # 0 是初始，其余每步是一次 (iter, j)
    #
    print(f"the best energy {best_E[-1]}")
    plt.figure(figsize=(27, 24))
    plt.plot(steps, mean_E, marker="o", label="average energies")
    plt.plot(steps, best_E, marker="x", label="best energy")
    plt.xlabel("优化 step (初始 + 5 次迭代 × 4 维度)")
    plt.ylabel("E(x, y1, y2, y3)")
    plt.title("4 维自回归无梯度优化: 能量随 step 变化")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
