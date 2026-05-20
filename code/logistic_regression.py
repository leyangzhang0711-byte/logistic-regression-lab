"""
逻辑回归：从零实现三种优化算法并对比
矩阵计算与优化 · 实验课作业 · 张乐扬

算法实现：
  - 批量梯度下降 (BGD)
  - 随机梯度下降 (SGD, mini-batch)
  - 牛顿法 (Newton / IRLS)

实验：UCI 乳腺癌数据集（569 样本，30 特征，二分类）
所有算法从零手写，不调用 sklearn 的 LogisticRegression。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════
#  全局超参数
# ══════════════════════════════════════════════════════════
LAMBDA = 0.01   # L2 正则化系数


# ══════════════════════════════════════════════════════════
#  1. 核心函数
# ══════════════════════════════════════════════════════════

def sigmoid(z):
    """数值稳定的 Sigmoid：对正负 z 分别处理，避免 exp overflow。"""
    result = np.zeros_like(z, dtype=float)
    pos = z >= 0
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    result[~pos] = ez / (1.0 + ez)
    return result


def cross_entropy_loss(X, y, w, lam=LAMBDA):
    """
    带 L2 正则化的交叉熵损失：
    f(w) = -(1/n)Σ[y_i log σ_i + (1-y_i)log(1-σ_i)] + (λ/2)||w[1:]||²
    """
    y_hat = np.clip(sigmoid(X @ w), 1e-12, 1 - 1e-12)
    ce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return ce + lam / 2.0 * np.dot(w[1:], w[1:])


def gradient(X, y, w, lam=LAMBDA):
    """
    梯度：∇f(w) = (1/n)X^T(ŷ - y) + λw  （偏置项不正则化）
    与线性回归梯度形式完全一致，体现优化框架的统一性。
    """
    grad = (X.T @ (sigmoid(X @ w) - y)) / len(y)
    reg = lam * w.copy();  reg[0] = 0.0
    return grad + reg


def hessian(X, y, w, lam=LAMBDA):
    """
    Hessian：∇²f(w) = (1/n)X^T D X + λI  （偏置项不正则化）
    D = diag(σ_i(1-σ_i)) > 0  →  H 正定  →  f 严格凸  →  全局最优唯一。
    """
    d = sigmoid(X @ w);  d = d * (1 - d)
    H = (X.T * d) @ X / len(y)
    reg = lam * np.eye(H.shape[0]);  reg[0, 0] = 0.0
    return H + reg


# ══════════════════════════════════════════════════════════
#  2. 三种优化算法
# ══════════════════════════════════════════════════════════

def bgd(X, y, lr=0.1, max_iter=500, tol=1e-7):
    """批量梯度下降：w -= α·∇f(w)  线性收敛，可能出现锯齿现象。"""
    w = np.zeros(X.shape[1])
    losses = [cross_entropy_loss(X, y, w)]
    for k in range(max_iter):
        grad = gradient(X, y, w)
        w -= lr * grad
        losses.append(cross_entropy_loss(X, y, w))
        if np.linalg.norm(grad) < tol:
            print(f"  [BGD]    第 {k+1} 步收敛"); break
    else:
        print(f"  [BGD]    达到最大迭代次数 {max_iter}")
    return w, losses


def sgd(X, y, lr=0.05, max_iter=300, batch_size=32, tol=1e-7, seed=42):
    """Mini-batch SGD：每 epoch 随机打乱，分批估计梯度。波动大、每步轻量。"""
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1]);  n = len(y)
    losses = [cross_entropy_loss(X, y, w)]
    for epoch in range(max_iter):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            w -= lr * gradient(X[batch], y[batch], w)
        losses.append(cross_entropy_loss(X, y, w))
        if epoch > 5 and abs(losses[-1] - losses[-6]) < tol:
            print(f"  [SGD]    第 {epoch+1} 个 epoch 收敛"); break
    else:
        print(f"  [SGD]    达到最大迭代次数 {max_iter}")
    return w, losses


def newton(X, y, max_iter=30, tol=1e-10):
    """
    牛顿法：w -= H^{-1} · ∇f(w)
    二次收敛，迭代次数极少。
    不直接求逆，用 np.linalg.solve 更稳定高效。
    """
    w = np.zeros(X.shape[1])
    losses = [cross_entropy_loss(X, y, w)]
    for k in range(max_iter):
        grad = gradient(X, y, w)
        H = hessian(X, y, w)
        delta = np.linalg.solve(H, grad)
        w -= delta
        losses.append(cross_entropy_loss(X, y, w))
        if np.linalg.norm(delta) < tol:
            print(f"  [Newton] 第 {k+1} 步收敛，||Δw||={np.linalg.norm(delta):.2e}"); break
    else:
        print(f"  [Newton] 达到最大迭代次数 {max_iter}")
    return w, losses


# ══════════════════════════════════════════════════════════
#  3. 评估
# ══════════════════════════════════════════════════════════

def accuracy(X, y, w):
    return np.mean((sigmoid(X @ w) >= 0.5) == y)


# ══════════════════════════════════════════════════════════
#  4. 可视化
# ══════════════════════════════════════════════════════════

def plot_convergence(losses_dict, save_path="convergence_curve.png"):
    """收敛曲线对比：左全局，右前 30 步放大（突出牛顿法二次收敛）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    styles = {
        'BGD':    ('tab:blue',   '-',  2.0),
        'SGD':    ('tab:orange', '--', 1.8),
        'Newton': ('tab:red',    '-',  2.5),
    }

    for ax, zoom, title in zip(
        axes,
        [False, True],
        ['Full Convergence Curve', 'Zoom-in: First 30 Iterations']
    ):
        for name, losses in losses_dict.items():
            color, ls, lw = styles[name]
            data = losses[:31] if zoom else losses
            marker = 'o' if (zoom and name == 'Newton') else None
            ax.plot(data, label=name, color=color, linestyle=ls, linewidth=lw,
                    marker=marker, markersize=5)
        ax.set_xlabel('Iterations / Epochs', fontsize=11)
        ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10);  ax.grid(True, alpha=0.3)
        if zoom: ax.set_xlim(0, 30)

    plt.suptitle('Logistic Regression: Optimizer Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  已保存：{save_path}")


def plot_decision_boundary(X_2d, y, weights_dict, save_path="decision_boundary.png"):
    """决策边界：三者几乎重合，验证凸优化的全局唯一最优解。"""
    fig, ax = plt.subplots(figsize=(9, 7))
    kw = dict(alpha=0.6, edgecolors='none', s=30)
    mask = y == 1
    ax.scatter(X_2d[mask, 0],  X_2d[mask, 1],  color='tab:blue',   label='Malignant (y=1)', **kw)
    ax.scatter(X_2d[~mask, 0], X_2d[~mask, 1], color='tab:orange', label='Benign (y=0)',    **kw)

    m = 0.8
    x1 = np.linspace(X_2d[:, 0].min()-m, X_2d[:, 0].max()+m, 400)
    x2 = np.linspace(X_2d[:, 1].min()-m, X_2d[:, 1].max()+m, 400)
    xx1, xx2 = np.meshgrid(x1, x2)
    Xg = np.c_[np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()]

    line_styles = {
        'BGD':    ('tab:blue',   '-',  2.5),
        'SGD':    ('tab:orange', '--', 2.0),
        'Newton': ('tab:red',    ':',  3.0),
    }
    for name, w in weights_dict.items():
        probs = sigmoid(Xg @ w).reshape(xx1.shape)
        color, ls, lw = line_styles[name]
        ax.contour(xx1, xx2, probs, levels=[0.5], colors=[color],
                   linestyles=[ls], linewidths=[lw])
        ax.plot([], [], color=color, linestyle=ls, linewidth=lw,
                label=f'{name} boundary')

    ax.set_xlabel('PCA Component 1', fontsize=11)
    ax.set_ylabel('PCA Component 2', fontsize=11)
    ax.set_title('Decision Boundary Comparison\n(PCA 2D Projection, Breast Cancer)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10);  ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  已保存：{save_path}")


# ══════════════════════════════════════════════════════════
#  5. 主程序
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  逻辑回归：三种优化算法对比实验")
    print(f"  数据集：UCI Breast Cancer  λ={LAMBDA}")
    print("=" * 60)

    # 数据
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    X_train = np.hstack([np.ones((len(y_tr), 1)), X_tr_s])
    X_test  = np.hstack([np.ones((len(y_te), 1)), X_te_s])

    print(f"\n训练 {len(y_tr)} / 测试 {len(y_te)} 样本，{X_train.shape[1]-1} 特征\n")

    # 训练
    print("▶ BGD  lr=0.1")
    w_bgd, L_bgd = bgd(X_train, y_tr, lr=0.1)

    print("\n▶ SGD  lr=0.05, batch=32")
    w_sgd, L_sgd = sgd(X_train, y_tr, lr=0.05, batch_size=32)

    print("\n▶ Newton")
    w_nt,  L_nt  = newton(X_train, y_tr)

    # 结果
    print("\n" + "=" * 52)
    print(f"  {'算法':<8}  {'准确率':>8}  {'迭代次数':>10}  {'最终 Loss':>12}")
    print("-" * 52)
    for name, w, L in [('BGD', w_bgd, L_bgd), ('SGD', w_sgd, L_sgd), ('Newton', w_nt, L_nt)]:
        acc = accuracy(X_test, y_te, w)
        print(f"  {name:<8}  {acc*100:>7.2f}%  {len(L)-1:>10}  {L[-1]:>12.6f}")
    print("=" * 52)

    # 可视化
    plot_convergence({'BGD': L_bgd, 'SGD': L_sgd, 'Newton': L_nt})

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_tr_s)
    Xb = np.hstack([np.ones((len(y_tr), 1)), X_2d])

    print("\n在 PCA 2D 空间训练（决策边界可视化）...")
    w2_bgd, _ = bgd(Xb, y_tr, lr=0.3)
    w2_sgd, _ = sgd(Xb, y_tr, lr=0.1, batch_size=32)
    w2_nt,  _ = newton(Xb, y_tr)

    plot_decision_boundary(X_2d, y_tr,
                           {'BGD': w2_bgd, 'SGD': w2_sgd, 'Newton': w2_nt})
    print("\n✅ 完成！图片已保存至当前目录。")


if __name__ == "__main__":
    main()
