"""
Experiment A: How Learning Rate alpha Affects BGD Convergence
Logistic Regression - Matrix Computation & Optimization Lab

Key finding:
  - alpha too small  → extremely slow convergence
  - alpha just right → stable linear convergence
  - alpha too large  → diverges (loss oscillates and explodes)
  - StandardScaler makes large alpha viable: a preprocessing insight
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════
#  Core functions (no regularization — to expose raw
#  sensitivity of BGD to learning rate)
# ══════════════════════════════════════════════════════════

def sigmoid(z):
    """Numerically stable sigmoid: handle positive/negative z separately."""
    result = np.zeros_like(z, dtype=float)
    pos = z >= 0
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    result[~pos] = ez / (1.0 + ez)
    return result

def cross_entropy_loss(X, y, w):
    yh = np.clip(sigmoid(X @ w), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh))

def gradient(X, y, w):
    """Gradient: (1/n) X^T (y_hat - y)  — identical form to linear regression."""
    return (X.T @ (sigmoid(X @ w) - y)) / len(y)

# ══════════════════════════════════════════════════════════
#  BGD with divergence detection
# ══════════════════════════════════════════════════════════

def bgd(X, y, lr=0.1, max_iter=60):
    """
    Batch Gradient Descent.
    Returns (loss_history, diverged_at_step).
    diverged_at_step is None if training completed normally.
    """
    w = np.zeros(X.shape[1])
    losses = [cross_entropy_loss(X, y, w)]

    for k in range(max_iter):
        w -= lr * gradient(X, y, w)
        v = cross_entropy_loss(X, y, w)

        # ── Key design: detect divergence early ───────────
        # If loss explodes (NaN or > 5), record where it happened
        # and stop — no point running further.
        if np.isnan(v) or v > 5:
            losses.append(float('nan'))
            return losses, k + 1          # return divergence step
        losses.append(v)

    return losses, None                    # None = converged normally

# ══════════════════════════════════════════════════════════
#  Data preparation
# ══════════════════════════════════════════════════════════

data = load_breast_cancer()
X_tr, _, y_tr, _ = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

scaler = StandardScaler()
X_train = np.hstack([np.ones((len(y_tr), 1)), scaler.fit_transform(X_tr)])

# ══════════════════════════════════════════════════════════
#  Run experiment across 5 learning rates
# ══════════════════════════════════════════════════════════

lrs    = [0.001,   0.1,          0.8,           5.0,           50.0      ]
labels = ['a=0.001\ntoo slow', 'a=0.1\njust right', 'a=0.8\noscillates',
          'a=5.0\nstill OK*',  'a=50\nDIVERGES'                          ]
colors = ['tab:green', 'tab:blue', 'tab:orange', 'purple', 'tab:red']

# ══════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ── Left: all 5 learning rates ────────────────────────────
ax = axes[0]
for lr, lbl, col in zip(lrs, labels, colors):
    L, div = bgd(X_train, y_tr, lr=lr, max_iter=60)
    valid = [v for v in L if not np.isnan(v)]
    lw = 2.5 if lr in [0.1, 50.0] else 1.8
    ax.plot(range(len(valid)), valid,
            label=lbl.replace('\n', ' '), color=col, linewidth=lw)
    if div:
        ax.annotate(f'diverges @ step {div}',
                    xy=(div - 1, valid[-1]),
                    xytext=(div + 1, valid[-1] + 0.15),
                    color=col, fontsize=8.5,
                    arrowprops=dict(arrowstyle='->', color=col))

ax.set_xlabel('Iterations', fontsize=11)
ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax.set_title('Learning Rate Effect on BGD', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.5)

# ── Right: stable runs only, with preprocessing insight ───
ax = axes[1]
for lr, lbl, col in zip(lrs[:4], labels[:4], colors[:4]):
    L, _ = bgd(X_train, y_tr, lr=lr, max_iter=60)
    valid = [v for v in L if not np.isnan(v)]
    ax.plot(valid, label=lbl.replace('\n', ' '), color=col, linewidth=2)

ax.set_xlabel('Iterations', fontsize=11)
ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax.set_title('Convergence Speed (stable runs)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── The "extra insight" annotation ────────────────────────
ax.text(33, 0.55,
        '* StandardScaler makes\n  large alpha viable here',
        fontsize=9, color='purple', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.7))

plt.suptitle('Experiment A: How Learning Rate alpha Affects BGD',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_A_learning_rate.png', dpi=150, bbox_inches='tight')
print("Saved: exp_A_learning_rate.png")
