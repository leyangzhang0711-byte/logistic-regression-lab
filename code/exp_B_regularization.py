"""
Experiment B: Effect of L2 Regularization on Logistic Regression
Logistic Regression - Matrix Computation & Optimization Lab

Core question:
  What happens to train/test accuracy as regularization lambda increases?

Key finding:
  - lambda too small  → slight overfitting (train-test gap visible)
  - lambda just right → test accuracy peaks (best generalization)
  - lambda too large  → underfitting (model weights crushed to zero,
                         both train and test accuracy collapse)

Design highlight:
  - Lambda swept on LOG scale (np.logspace) — spans 6 orders of magnitude
  - Loss plotted WITHOUT the regularization term for fair cross-dataset comparison
  - Gray shading visualizes the train-test gap as a proxy for overfitting severity
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════
#  Core functions
# ══════════════════════════════════════════════════════════

def sigmoid(z):
    """Numerically stable sigmoid."""
    result = np.zeros_like(z, dtype=float)
    pos = z >= 0
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    result[~pos] = ez / (1.0 + ez)
    return result

def cross_entropy_loss(X, y, w, lam=0.0):
    """
    Cross-entropy loss with optional L2 regularization.
    When lam=0, gives pure CE for fair comparison across lambda values.
    When lam>0, includes the penalty term used during training.
    """
    yh = np.clip(sigmoid(X @ w), 1e-12, 1 - 1e-12)
    ce = -np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh))
    return ce + lam / 2.0 * np.dot(w[1:], w[1:])

def gradient(X, y, w, lam):
    """Gradient with L2 regularization. Bias term w[0] is not penalized."""
    g = (X.T @ (sigmoid(X @ w) - y)) / len(y)
    reg = lam * w.copy()
    reg[0] = 0.0    # do NOT regularize the bias term
    return g + reg

def bgd(X, y, lam, lr=0.1, max_iter=500, tol=1e-7):
    """Batch Gradient Descent with L2 regularization."""
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        g = gradient(X, y, w, lam)
        w -= lr * g
        if np.linalg.norm(g) < tol:
            break
    return w

def accuracy(X, y, w):
    return np.mean((sigmoid(X @ w) >= 0.5) == y)

# ══════════════════════════════════════════════════════════
#  Data preparation
# ══════════════════════════════════════════════════════════

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)
X_train = np.hstack([np.ones((len(y_tr), 1)), X_tr_s])
X_test  = np.hstack([np.ones((len(y_te), 1)), X_te_s])

# ══════════════════════════════════════════════════════════
#  Lambda sweep — KEY DESIGN: log scale, not linear
#  Linear scale would cluster all interesting behavior near 0
#  and miss the underfitting collapse at large lambda.
# ══════════════════════════════════════════════════════════

lambdas = np.logspace(-4, 2, 40)   # 0.0001 → 100, 40 evenly-spaced points on log scale

train_accs, test_accs = [], []
train_losses, test_losses = [], []

for lam in lambdas:
    w = bgd(X_train, y_tr, lam=lam)
    train_accs.append(accuracy(X_train, y_tr, w))
    test_accs.append(accuracy(X_test,  y_te, w))
    # Use lam=0 for loss: we want PURE cross-entropy, not inflated by penalty
    train_losses.append(cross_entropy_loss(X_train, y_tr, w, lam=0))
    test_losses.append(cross_entropy_loss(X_test,  y_te, w, lam=0))

best_idx = int(np.argmax(test_accs))
best_lam = lambdas[best_idx]

print(f"Best lambda  : {best_lam:.4f}")
print(f"Test accuracy: {test_accs[best_idx]*100:.2f}%")
print(f"lambda=0.0001 → train {train_accs[0]*100:.2f}%, test {test_accs[0]*100:.2f}%")
print(f"lambda=100    → train {train_accs[-1]*100:.2f}%, test {test_accs[-1]*100:.2f}%")

# ══════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ── Left: Accuracy vs Lambda ──────────────────────────────
ax = axes[0]
ax.semilogx(lambdas, [a * 100 for a in train_accs],
            color='tab:blue', linewidth=2.2, label='Train Accuracy')
ax.semilogx(lambdas, [a * 100 for a in test_accs],
            color='tab:orange', linewidth=2.2, label='Test Accuracy')

# Gray shading = train-test gap = visual proxy for overfitting
ax.fill_between(lambdas,
                [a * 100 for a in train_accs],
                [a * 100 for a in test_accs],
                alpha=0.12, color='gray', label='Train-Test gap')

# Mark optimal lambda
ax.axvline(best_lam, color='tab:red', linestyle='--', alpha=0.7, linewidth=1.5)
ax.scatter([best_lam], [test_accs[best_idx] * 100], color='tab:red', zorder=5, s=80)
ax.annotate(f'Best λ={best_lam:.3f}\nTest={test_accs[best_idx]*100:.1f}%',
            xy=(best_lam, test_accs[best_idx] * 100),
            xytext=(best_lam * 4, test_accs[best_idx] * 100 - 2),
            fontsize=9, color='tab:red',
            arrowprops=dict(arrowstyle='->', color='tab:red'))

# Region labels
ax.text(0.00015, 87, 'Overfitting zone\n(train > test)',
        fontsize=8.5, color='gray', style='italic')
ax.text(8, 87, 'Underfitting zone\n(both drop)',
        fontsize=8.5, color='gray', style='italic')

ax.set_xlabel('Regularization λ (log scale)', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Accuracy vs Regularization Strength', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(85, 102)

# ── Right: Loss vs Lambda ─────────────────────────────────
ax = axes[1]
ax.semilogx(lambdas, train_losses, color='tab:blue',   linewidth=2.2, label='Train Loss (CE only)')
ax.semilogx(lambdas, test_losses,  color='tab:orange', linewidth=2.2, label='Test Loss (CE only)')
ax.axvline(best_lam, color='tab:red', linestyle='--', alpha=0.7, linewidth=1.5,
           label=f'Best λ={best_lam:.3f}')

ax.set_xlabel('Regularization λ (log scale)', fontsize=11)
ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax.set_title('Loss vs Regularization Strength', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.suptitle('Experiment B: Effect of L2 Regularization (λ sweep)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_B_regularization.png', dpi=150, bbox_inches='tight')
print("Saved: exp_B_regularization.png")
