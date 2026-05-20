"""
逻辑回归：从零实现三种优化算法
矩阵计算与优化 · 实验课作业

算法：批量梯度下降(BGD) / 随机梯度下降(SGD) / 牛顿法(Newton)
================================================================
【组员使用说明】
只需修改 main() 函数中"数据准备"部分，替换数据集即可。
其他代码不需要改动。
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

LAMBDA = 0.01  # L2 正则化系数

# ══════════════════════════════════════════════════════════
#  1. 数学核心
# ══════════════════════════════════════════════════════════

def sigmoid(z):
    result = np.zeros_like(z, dtype=float)
    pos = z >= 0
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    result[~pos] = ez / (1.0 + ez)
    return result

def cross_entropy_loss(X, y, w, lam=LAMBDA):
    y_hat = np.clip(sigmoid(X @ w), 1e-12, 1 - 1e-12)
    ce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return ce + lam / 2.0 * np.dot(w[1:], w[1:])

def gradient(X, y, w, lam=LAMBDA):
    grad = (X.T @ (sigmoid(X @ w) - y)) / len(y)
    reg = lam * w.copy(); reg[0] = 0.0
    return grad + reg

def hessian(X, y, w, lam=LAMBDA):
    d = sigmoid(X @ w); d = d * (1 - d)
    H = (X.T * d) @ X / len(y)
    reg = lam * np.eye(H.shape[0]); reg[0, 0] = 0.0
    return H + reg

# ══════════════════════════════════════════════════════════
#  2. 三种优化算法
# ══════════════════════════════════════════════════════════

def bgd(X, y, lr=0.1, max_iter=500, tol=1e-7):
    """批量梯度下降"""
    w = np.zeros(X.shape[1])
    losses = [cross_entropy_loss(X, y, w)]
    for k in range(max_iter):
        grad = gradient(X, y, w)
        w -= lr * grad
        losses.append(cross_entropy_loss(X, y, w))
        if np.linalg.norm(grad) < tol:
            print(f"  [BGD] 第{k+1}步收敛"); break
    return w, losses

def sgd(X, y, lr=0.05, max_iter=300, batch_size=32, seed=42):
    """随机梯度下降（Mini-batch）"""
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1]); n = len(y)
    losses = [cross_entropy_loss(X, y, w)]
    for epoch in range(max_iter):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            w -= lr * gradient(X[batch], y[batch], w)
        losses.append(cross_entropy_loss(X, y, w))
        if epoch > 5 and abs(losses[-1] - losses[-6]) < 1e-7:
            print(f"  [SGD] 第{epoch+1}个epoch收敛"); break
    return w, losses

def newton(X, y, max_iter=30, tol=1e-10):
    """牛顿法"""
    w = np.zeros(X.shape[1])
    losses = [cross_entropy_loss(X, y, w)]
    for k in range(max_iter):
        grad = gradient(X, y, w)
        delta = np.linalg.solve(hessian(X, y, w), grad)
        w -= delta
        losses.append(cross_entropy_loss(X, y, w))
        if np.linalg.norm(delta) < tol:
            print(f"  [Newton] 第{k+1}步收敛"); break
    return w, losses

# ══════════════════════════════════════════════════════════
#  3. 评估 & 可视化
# ══════════════════════════════════════════════════════════

def accuracy(X, y, w):
    return np.mean((sigmoid(X @ w) >= 0.5) == y)

def plot_convergence(losses_dict, title="Convergence Comparison", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    styles = {'BGD': ('tab:blue', '-', 2.0),
              'SGD': ('tab:orange', '--', 1.8),
              'Newton': ('tab:red', '-', 2.5)}
    for ax, zoom, t in zip(axes, [False, True],
                           ['Full Convergence', 'Zoom: First 30 Iters']):
        for name, losses in losses_dict.items():
            c, ls, lw = styles.get(name, ('gray', '-', 1.5))
            data = losses[:31] if zoom else losses
            marker = 'o' if (zoom and name == 'Newton') else None
            ax.plot(data, label=name, color=c, linestyle=ls, linewidth=lw,
                    marker=marker, markersize=5)
        ax.set_xlabel('Iterations'); ax.set_ylabel('Loss')
        ax.set_title(t); ax.legend(); ax.grid(alpha=0.3)
        if zoom: ax.set_xlim(0, 30)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存：{save_path}")
    plt.show()

def plot_decision_boundary(X_2d, y, weights_dict, title="Decision Boundary", save_path=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = y == 1
    ax.scatter(X_2d[mask,0],  X_2d[mask,1],  color='tab:blue',   alpha=0.6, s=30, label='Class 1')
    ax.scatter(X_2d[~mask,0], X_2d[~mask,1], color='tab:orange', alpha=0.6, s=30, label='Class 0')
    m = 0.8
    x1 = np.linspace(X_2d[:,0].min()-m, X_2d[:,0].max()+m, 400)
    x2 = np.linspace(X_2d[:,1].min()-m, X_2d[:,1].max()+m, 400)
    xx1, xx2 = np.meshgrid(x1, x2)
    Xg = np.c_[np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()]
    styles = {'BGD': ('tab:blue','-',2.5), 'SGD': ('tab:orange','--',2.0), 'Newton': ('tab:red',':',3.0)}
    for name, w in weights_dict.items():
        probs = sigmoid(Xg @ w).reshape(xx1.shape)
        c, ls, lw = styles.get(name, ('gray','-',1.5))
        ax.contour(xx1, xx2, probs, levels=[0.5], colors=[c], linestyles=[ls], linewidths=[lw])
        ax.plot([], [], color=c, linestyle=ls, linewidth=lw, label=f'{name} boundary')
    ax.set_xlabel('PCA 1'); ax.set_ylabel('PCA 2')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存：{save_path}")
    plt.show()

# ══════════════════════════════════════════════════════════
#  4. 数据加载函数（组员替换这里）
# ══════════════════════════════════════════════════════════

def load_breast_cancer_data():
    """默认数据集：UCI 乳腺癌"""
    data = load_breast_cancer()
    return data.data, data.target, "Breast Cancer"

def load_titanic_data():
    """
    【组员任务一】泰坦尼克生存预测
    目标：预测乘客是否生还（0=遇难, 1=生还）
    在 main() 里把 load_breast_cancer_data() 替换成这个函数即可
    """
    data = fetch_openml(name='titanic', version=1, as_frame=True)
    df = data.frame.copy()
    # 选取数值型特征，去掉缺失值
    df = df[['pclass', 'age', 'sibsp', 'parch', 'fare', 'survived']].dropna()
    X = df.drop('survived', axis=1).values.astype(float)
    y = df['survived'].astype(int).values
    return X, y, "Titanic Survival"

def load_credit_data():
    """
    【组员任务一】德国信用风险预测
    目标：预测贷款申请人是否违约（1=良好, 0=风险）
    在 main() 里把 load_breast_cancer_data() 替换成这个函数即可
    """
    data = fetch_openml(name='credit-g', version=1, as_frame=False)
    X = data.data
    le = LabelEncoder()
    y = le.fit_transform(data.target)  # 转成 0/1
    # 只保留数值列（简化处理）
    X = X[:, [1, 4, 7, 10, 12]].astype(float)
    return X, y, "German Credit Risk"

# ══════════════════════════════════════════════════════════
#  5. 主程序
# ══════════════════════════════════════════════════════════

def main():
    # ── 【组员】在这里替换数据集函数 ──────────────────────
    X_raw, y, dataset_name = load_breast_cancer_data()
    # X_raw, y, dataset_name = load_titanic_data()
    # X_raw, y, dataset_name = load_credit_data()
    # ──────────────────────────────────────────────────────

    print(f"\n数据集：{dataset_name}  样本数：{len(y)}  特征数：{X_raw.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    X_train = np.hstack([np.ones((len(y_tr),1)), X_tr_s])
    X_test  = np.hstack([np.ones((len(y_te),1)), X_te_s])

    # 训练
    print("\n▶ BGD"); w_bgd, L_bgd = bgd(X_train, y_tr)
    print("▶ SGD"); w_sgd, L_sgd = sgd(X_train, y_tr)
    print("▶ Newton"); w_nt,  L_nt  = newton(X_train, y_tr)

    # 结果
    print(f"\n{'算法':<10} {'准确率':>8} {'迭代次数':>10}")
    print("-" * 32)
    for name, w, L in [('BGD',w_bgd,L_bgd),('SGD',w_sgd,L_sgd),('Newton',w_nt,L_nt)]:
        print(f"{name:<10} {accuracy(X_test,y_te,w)*100:>7.2f}% {len(L)-1:>10}")

    # 可视化
    plot_convergence({'BGD': L_bgd, 'SGD': L_sgd, 'Newton': L_nt},
                     title=f"{dataset_name}: Convergence",
                     save_path=f"convergence_{dataset_name.replace(' ','_')}.png")

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_tr_s)
    Xb = np.hstack([np.ones((len(y_tr),1)), X_2d])
    w2_bgd,_ = bgd(Xb, y_tr, lr=0.3)
    w2_sgd,_ = sgd(Xb, y_tr, lr=0.1)
    w2_nt, _ = newton(Xb, y_tr)
    plot_decision_boundary(X_2d, y_tr,
                           {'BGD':w2_bgd,'SGD':w2_sgd,'Newton':w2_nt},
                           title=f"{dataset_name}: Decision Boundary (PCA 2D)",
                           save_path=f"boundary_{dataset_name.replace(' ','_')}.png")

if __name__ == "__main__":
    main()

