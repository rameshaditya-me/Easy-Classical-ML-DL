<a href="https://colab.research.google.com/github/rameshaditya-me/Easy-Classical-ML-DL/blob/main/src/notebooks/logistic-regression/logistic_regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Logistic Regression
---

Logistic Regression is both a regression and classification method. In linear regression and linear models for classification we model the hypothesis function using a parameterized model.

In logistic regression however, we model the class conditional probability for a binary classification problem using a parametrized model which is defined as:

$$
p(C_1~\mid~x,w) = σ(w^Tx) = \frac{1}{1+\exp(-w^Tx)} \in [0,1]
$$

$$
p(C_0~\mid~w,x) = 1 - p(C_1~\mid~w,x)
$$

For a given input $x$ the class is classified as:
$$
x\in\begin{cases}
      C_0 & p(C_0~\mid~w,x) > p(C_1~\mid~w,x) \\
      C_1 & o.w
   \end{cases}
$$

The logistic regression model is a linear model due to the fact that the decision boundary between classes is linear. We can see this by:

$$
\begin{align*}
p(C_0~\mid~w,x) = p(C_1~\mid~w,x) \rightarrow p(C_1~\mid~w,x) = 0.5 \\
\frac{1}{1+\exp(-w^Tx)} = 0.5 \\
w^Tx = 0
\end{align*}
$$

#### Fitting a Logistic Regression model to data
---

Given a dataset $\mathcal{D}=\{x_1,C_{x_1},\dots,x_n,C_{x_n}\}$ we want to fit a classifiction model to this data using a logistic regression model. We can model this using the log-likelihood function as follows:
$$
\begin{align*}
p(w|x_1,x_2,\dots,x_n) \propto p(x_1,\dots,x_n|w) \\
-log~p(x_1,\dots,x_n|w) = -\sum_{i=1}^n~log~p(x_i|w)\\ = - \sum_{i=1}^n log~p_{x_i}^{C_{x_i}}(1-p_{x_i})^{1-C_{x_i}} \\
= - \sum_{i=1}^n C_{x_i}log~p_{x_i} + ({1-C_{x_i}})log(1-p_{x_i})
\end{align*}
$$

where $p_{x_i}=\sigma(w^Tx_i)$. Now if we substitute $p_{x_i}$ in the above equation we will get a differentiable objective when we try to maximize the liklihood. We can leverage this to our advantage and use this to compute the ML estimate of teh weight $w$. The objective does not have a closed form solution.

#### Gradient function of the objective
---

The objective $\mathcal{L}$ is defined as:
$$
\begin{align*}
\mathcal{L} = \underset{w}{argmax} \sum_{i=1}^n C_{x_i}log~p_{x_i} + ({1-C_{x_i}})log(1-p_{x_i}) \\
\nabla_{w} \mathcal{L} = \sum_{i=1}^n C_{x_i}\frac{1}{p_{x_i}}\nabla_{w}p_{x_i} + \frac{1}{1-p_{x_i}}\nabla_w(1-p_{x_i}) \\
\end{align*}
$$

Here $\nabla_wp_{x_i}=\sigma(w^Tx)(1-\sigma(w^Tx))$ and $\nabla_w~(1-p_{x_i})=-\nabla_w~p_{x_i}=-\sigma(w^Tx)(1-\sigma(w^Tx))$. Substituting this we get:

$$
\begin{align*}
\nabla_{w} \mathcal{L} = \sum_{i=1}^n C_{x_i}\frac{1}{p_{x_i}}p_{x_i}(1-p_{x_i}) - (1-C_{x_i})\frac{1}{1-p_{x_i}}p_{x_i}(1-p_{x_i}) \\
\nabla_{w} \mathcal{L} = \sum_{i=1}^n C_{x_i}(1-p_{x_i})-(1-C_{xi})p_{x_i}
\end{align*}
$$

This gradient has a closed form expression and can be easily computed.

### Building intuition

We fix **two labelled points** in $\mathbb{R}^2$ and a **random initial decision boundary** (a line where $w^\top x = 0$). At each gradient-ascent step on $\mathcal{L}(w)$, the update direction is

$$
\nabla_w \mathcal{L} = \sum_{i=1}^2 (y_i - p_i)\, x_i.
$$

Each point contributes an arrow scaled by the **residual** $y_i - p_i$. When the model assigns the correct label with high probability, $p_i \approx y_i$ and that contribution vanishes; misclassified or uncertain points produce large residuals and pull $w$ more strongly.


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

rng = np.random.default_rng(7)

# --- two labelled points on the SAME side of the initial boundary ---
# (opposite labels, so at least one is misclassified at step 0)
X = np.array([[1.4, 1.6], [0.7, 0.9]])
y = np.array([1, 0])
X_aug = np.hstack([X, np.ones((2, 1))])


def random_boundary_same_side(X_aug, rng, margin=0.25):
    """Sample w so every point satisfies w^T x > 0 (same side of the line)."""
    while True:
        w = rng.normal(size=3)
        logits = X_aug @ w
        if np.all(logits > margin) or np.all(logits < -margin):
            return w / np.linalg.norm(w)


w = random_boundary_same_side(X_aug, rng)
assert np.all(np.sign(X_aug @ w) == np.sign(X_aug @ w)[0]), "both points must start on the same side"

lr = 0.45
n_steps = 100


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def probs_and_grad(w):
    logits = X_aug @ w
    p = sigmoid(logits)
    residuals = y - p
    grad = (residuals[:, None] * X_aug).sum(axis=0)
    return p, residuals, grad


def boundary_segment(w, xlim=(-3, 3), ylim=(-3, 3)):
    w1, w2, b = w
    if abs(w2) > 1e-6:
        xs = np.array(xlim)
        ys = -(b + w1 * xs) / w2
        return xs, ys
    x = -b / (w1 + 1e-8)
    return np.array([x, x]), np.array(ylim)


def boundary_center(w, xlim, ylim):
    xs, ys = boundary_segment(w, xlim, ylim)
    return np.array([0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1])])


history = []
w_curr = w.copy()
for _ in range(n_steps + 1):
    p, residuals, grad = probs_and_grad(w_curr)
    history.append({
        "w": w_curr.copy(),
        "grad": grad.copy(),
        "p": p.copy(),
        "residuals": residuals.copy(),
        "grad_norm": np.linalg.norm(grad),
    })
    w_curr = w_curr + lr * grad

# common scale for weight-space vectors drawn in the feature plane
max_vec = max(
    max(np.linalg.norm(h["w"][:2]) for h in history),
    max(np.linalg.norm(lr * h["grad"][:2]) for h in history),
    1e-6,
)
vec_scale = 0.55 / max_vec

colors = ["#e74c3c", "#3498db"]

fig, ax = plt.subplots(figsize=(9, 7))
xlim, ylim = (-0.5, 2.5), (-0.5, 2.5)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal")
ax.grid(True, alpha=0.25)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("$w$ (normal to boundary), update $lr\\nabla_w\\mathcal{L}$, and $w + lr\\nabla_w\\mathcal{L}$")

boundary_line, = ax.plot([], [], "k-", lw=2.5, label="decision boundary ($w^\\top x = 0$)")
ax.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in y], s=120, zorder=4, edgecolors="white", linewidths=1.5)
anchor_dot = ax.scatter([], [], c="k", s=30, zorder=5)

residual_arrows = []
for i in range(2):
    arr = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", lw=2, color=colors[i], alpha=0.55),
    )
    residual_arrows.append(arr)

w_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=3, color="#8e44ad"),
)
step_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=3, color="#27ae60"),
)
sum_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=3, color="#e67e22", linestyle="--"),
)
parallelogram_h, = ax.plot([], [], ":", color="gray", lw=1.2, alpha=0.8)
parallelogram_v, = ax.plot([], [], ":", color="gray", lw=1.2, alpha=0.8)

text_box = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)
ax.legend(
    handles=[
        boundary_line,
        plt.Line2D([0], [0], color="#8e44ad", lw=3, label=r"$w_{1:2}$ (normal to line)"),
        plt.Line2D([0], [0], color="#27ae60", lw=3, label=r"$lr\,\nabla_{w_{1:2}}\mathcal{L}$"),
        plt.Line2D([0], [0], color="#e67e22", lw=3, ls="--", label=r"$w_{1:2}+lr\,\nabla_{w_{1:2}}\mathcal{L}$"),
    ],
    loc="lower right",
    fontsize=9,
)


def _set_arrow(ann, origin, vec):
    tip = origin + vec
    ann.xy = (tip[0], tip[1])
    ann.set_position((origin[0], origin[1]))


def init():
    boundary_line.set_data([], [])
    parallelogram_h.set_data([], [])
    parallelogram_v.set_data([], [])
    anchor_dot.set_offsets(np.empty((0, 2)))
    text_box.set_text("")
    return [boundary_line, text_box, parallelogram_h, parallelogram_v, anchor_dot, w_arrow, step_arrow, sum_arrow, *residual_arrows]


def update(frame):
    state = history[frame]
    xs, ys = boundary_segment(state["w"], xlim=xlim, ylim=ylim)
    boundary_line.set_data(xs, ys)

    anchor = boundary_center(state["w"], xlim, ylim)
    anchor_dot.set_offsets(anchor.reshape(1, -1))

    w_xy = state["w"][:2] * vec_scale
    step_xy = lr * state["grad"][:2] * vec_scale
    sum_xy = w_xy + step_xy

    _set_arrow(w_arrow, anchor, w_xy)
    _set_arrow(step_arrow, anchor, step_xy)
    _set_arrow(sum_arrow, anchor, sum_xy)

    # parallelogram for vector addition
    parallelogram_h.set_data([anchor[0] + w_xy[0], anchor[0] + sum_xy[0]], [anchor[1] + w_xy[1], anchor[1] + sum_xy[1]])
    parallelogram_v.set_data([anchor[0] + step_xy[0], anchor[0] + sum_xy[0]], [anchor[1] + step_xy[1], anchor[1] + sum_xy[1]])

    arrow_scale = 1.8
    for i, arr in enumerate(residual_arrows):
        r = state["residuals"][i]
        arr.set_position((X[i, 0], X[i, 1]))
        arr.xy = (X[i, 0] + r * X[i, 0] * arrow_scale, X[i, 1] + r * X[i, 1] * arrow_scale)
        arr.set_alpha(min(1.0, 0.25 + abs(r)))

    text_box.set_text(
        f"step {frame}/{n_steps}\n"
        f"$\\|\\nabla_w \\mathcal{{L}}\\| = {state['grad_norm']:.3f}$\n"
        f"point 1: $p={state['p'][0]:.3f}$, $y-p={state['residuals'][0]:+.3f}$\n"
        f"point 2: $p={state['p'][1]:.3f}$, $y-p={state['residuals'][1]:+.3f}$"
    )
    return [boundary_line, text_box, parallelogram_h, parallelogram_v, anchor_dot, w_arrow, step_arrow, sum_arrow, *residual_arrows]


anim = animation.FuncAnimation(
    fig, update, frames=len(history), init_func=init, blit=False, interval=350,
)
plt.close(fig)
HTML(anim.to_jshtml())
```

### Why not use linear regression for binary targets?

Ordinary least squares is built for **continuous** responses with **Gaussian** noise of constant variance. Binary labels violate that model in several ways:

1. **Wrong conditional distribution.** For classification, $p(y~\mid~w,x)$ is **Bernoulli**, not normal. Squared error is not the log-likelihood of the data; logistic regression minimizes **cross-entropy**, which matches a Bernoulli model.

2. **Unbounded predictions.** Linear regression can output any real value, so predicted \"probabilities\" often fall **outside $[0,1]$**, even though valid probabilities must stay in that interval.


```python
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
list(data.target_names)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(data.data))
ev = pca.explained_variance_ratio_

df = pd.DataFrame(
    {
        "PC 1": X_pca[:, 0],
        "PC 2": X_pca[:, 1],
        "Diagnosis": pd.Categorical(
            data.target_names[data.target],
            categories=data.target_names,
        ),
    }
)

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.85,
    rc={
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)

palette = {"malignant": "#e74c3c", "benign": "#3498db"}

g = sns.jointplot(
    data=df,
    x="PC 1",
    y="PC 2",
    hue="Diagnosis",
    palette=palette,
    kind="scatter",
    height=7,
    ratio=4,
    space=0.08,
    alpha=0.8,
    s=60,
    edgecolor="white",
    linewidth=0.5,
    marginal_kws={"fill": True, "alpha": 0.45, "linewidth": 1.4},
)

g.ax_joint.set_facecolor("#fafafa")
g.ax_joint.set_xlabel(f"PC 1 ({ev[0]:.1%} explained variance)", labelpad=10)
g.ax_joint.set_ylabel(f"PC 2 ({ev[1]:.1%} explained variance)", labelpad=10)
g.fig.suptitle(
    "Wisconsin Breast Cancer — 2D PCA projection",
    y=1.1,
    fontsize=16,
    weight="semibold",
)
g.fig.text(
    0.5,
    1.005,
    f"{len(df):,} tumors · {ev.sum():.1%} variance captured in 2 components",
    ha="center",
    fontsize=11,
    color="#555555",
)
sns.move_legend(
    g.ax_joint,
    "upper right",
    title="Diagnosis",
    bbox_to_anchor=(1.0, 1.0),
    frameon=True,
    framealpha=0.92,
)
plt.show()

```

In the above plot we have visualized the "breast cancer Wisconsin dataset" by using the $1^{\text{st}}$ and $2^{\text{nd}}$ Principal Component for each datapoint. There are 2 classes "malignant" and "benign" represented by "red" and "blue" colors. It is further visible that the clases seems to be somewhat linearly separable and thus logistic regression might be a good model. We will now test this model on the above dataset.


```python
y = data.target.astype(float)
X_aug = np.hstack([X_pca, np.ones((len(X_pca), 1))])
rng = np.random.default_rng(7)

lr = 0.3
n_steps = 70

# --- display size for animation ---
ANIM_FIGSIZE = (6, 6)
ANIM_DPI = 80

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def grad(w, X_aug, y):
    """Gradient of the log-likelihood w.r.t. w (batch or single-sample)."""
    p = sigmoid(X_aug @ w)
    residuals = y - p
    return (residuals[:, None] * X_aug).sum(axis=0)

def boundary_segment(w, xlim, ylim):
    w1, w2, b = w
    if abs(w2) > 1e-8:
        xs = np.array(xlim)
        ys = -(b + w1 * xs) / w2
        return xs, ys
    x = -b / (w1 + 1e-8)
    return np.array([x, x]), np.array(ylim)

def boundary_center(w, xlim, ylim):
    xs, ys = boundary_segment(w, xlim, ylim)
    return np.array([0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1])])

# --- SGD fit in PCA space ---
w = rng.normal(scale=0.5, size=3)
history = []
w_curr = w.copy()

for step in range(n_steps + 1):
    p = sigmoid(X_aug @ w_curr)
    residuals = y - p
    if step < n_steps:
        i = int(rng.integers(len(y)))
        g_step = grad(w_curr, X_aug[i : i + 1], y[i : i + 1])
    else:
        i = None
        g_step = grad(w_curr, X_aug, y)
    history.append(
        {
            "w": w_curr.copy(),
            "grad": g_step.copy(),
            "p": p.copy(),
            "residuals": residuals.copy(),
            "grad_norm": np.linalg.norm(g_step),
            "sample_idx": i,
        }
    )
    if step < n_steps:
        w_curr = w_curr + lr * g_step

w_fit = history[-1]["w"]
xlim = (df["PC 1"].min() - 0.5, df["PC 1"].max() + 0.5)
ylim = (df["PC 2"].min() - 0.5, df["PC 2"].max() + 0.5)

# --- overlay fitted decision boundary on the PCA plot ---
bx, by = boundary_segment(w_fit, xlim=xlim, ylim=ylim)
g.ax_joint.plot(
    bx,
    by,
    color="black",
    lw=2.8,
    ls="-",
    zorder=6,
)
g.ax_joint.set_xlim(*xlim)
g.ax_joint.set_ylim(*ylim)
g.fig.suptitle(
    "Wisconsin Breast Cancer — PCA + logistic regression boundary",
    y=1.03,
    fontsize=16,
    weight="semibold",
)
g.fig.texts[0].set_text(
    f"{len(df):,} tumors · SGD ({n_steps} steps, lr={lr}) · {ev.sum():.1%} variance in 2 PCs"
)
handles, labels = g.ax_joint.get_legend_handles_labels()
g.ax_joint.legend(
    handles=handles + [plt.Line2D([0], [0], color="black", lw=2.8, label=r"$w^\top x = 0$")],
    title="Diagnosis",
    loc="upper right",
    frameon=True,
    framealpha=0.92,
)
plt.show()

# --- animated fitting (same vector-addition view as the 2-point toy example) ---
max_vec = max(
    max(np.linalg.norm(h["w"][:2]) for h in history),
    max(np.linalg.norm(lr * h["grad"][:2]) for h in history),
    1e-6,
)
vec_scale = 0.55 / max_vec

mal_idx = np.where(y == 1)[0]
ben_idx = np.where(y == 0)[0]

fig, ax = plt.subplots(figsize=ANIM_FIGSIZE, dpi=ANIM_DPI)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal")
ax.grid(True, alpha=0.25)
ax.set_xlabel(f"PC 1 ({ev[0]:.1%} variance)", fontsize=9)
ax.set_ylabel(f"PC 2 ({ev[1]:.1%} variance)", fontsize=9)
ax.set_title(
    r"$w$ (normal to boundary), update $lr\,\nabla_w\mathcal{L}$, and $w + lr\,\nabla_w\mathcal{L}$"
    "\n(SGD on PCA features)",
    fontsize=10,
)

for label, name in enumerate(data.target_names):
    mask = y == label
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=palette[name],
        s=20,
        alpha=0.45,
        edgecolors="white",
        linewidths=0.4,
        label=name,
        zorder=2,
    )

boundary_line, = ax.plot([], [], "k-", lw=2.0, label=r"decision boundary ($w^\top x = 0$)")
anchor_dot = ax.scatter([], [], c="k", s=24, zorder=5)

residual_arrows = []
for _ in range(2):
    arr = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#f39c12", alpha=0.85),
    )
    residual_arrows.append(arr)

w_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2.4, color="#8e44ad"),
)
step_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2.4, color="#27ae60"),
)
sum_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2.4, color="#e67e22", linestyle="--"),
)
parallelogram_h, = ax.plot([], [], ":", color="gray", lw=1.0, alpha=0.8)
parallelogram_v, = ax.plot([], [], ":", color="gray", lw=1.0, alpha=0.8)

text_box = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)
ax.legend(
    handles=[
        boundary_line,
        plt.Line2D([0], [0], color="#8e44ad", lw=3, label=r"$w_{1:2}$ (normal to line)"),
        plt.Line2D([0], [0], color="#27ae60", lw=3, label=r"$lr\,\nabla_{w_{1:2}}\mathcal{L}$ (SGD)"),
        plt.Line2D([0], [0], color="#e67e22", lw=3, ls="--", label=r"$w_{1:2}+lr\,\nabla_{w_{1:2}}\mathcal{L}$"),
    ],
    loc="lower right",
    fontsize=9,
)

def _set_arrow(ann, origin, vec):
    tip = origin + vec
    ann.xy = (tip[0], tip[1])
    ann.set_position((origin[0], origin[1]))

def _top_residual_indices(residuals):
    i_mal = mal_idx[np.argmax(np.abs(residuals[mal_idx]))]
    i_ben = ben_idx[np.argmax(np.abs(residuals[ben_idx]))]
    return i_mal, i_ben

def init():
    boundary_line.set_data([], [])
    parallelogram_h.set_data([], [])
    parallelogram_v.set_data([], [])
    anchor_dot.set_offsets(np.empty((0, 2)))
    text_box.set_text("")
    return [
        boundary_line, text_box, parallelogram_h, parallelogram_v,
        anchor_dot, w_arrow, step_arrow, sum_arrow, *residual_arrows,
    ]

def update(frame):
    state = history[frame]
    bx, by = boundary_segment(state["w"], xlim=xlim, ylim=ylim)
    boundary_line.set_data(bx, by)

    anchor = boundary_center(state["w"], xlim, ylim)
    anchor_dot.set_offsets(anchor.reshape(1, -1))

    w_xy = state["w"][:2] * vec_scale
    step_xy = lr * state["grad"][:2] * vec_scale
    sum_xy = w_xy + step_xy

    _set_arrow(w_arrow, anchor, w_xy)
    _set_arrow(step_arrow, anchor, step_xy)
    _set_arrow(sum_arrow, anchor, sum_xy)

    parallelogram_h.set_data(
        [anchor[0] + w_xy[0], anchor[0] + sum_xy[0]],
        [anchor[1] + w_xy[1], anchor[1] + sum_xy[1]],
    )
    parallelogram_v.set_data(
        [anchor[0] + step_xy[0], anchor[0] + sum_xy[0]],
        [anchor[1] + step_xy[1], anchor[1] + sum_xy[1]],
    )

    i_mal, i_ben = _top_residual_indices(state["residuals"])

    arrow_scale = 0.35
    for arr, idx in zip(residual_arrows, [i_mal, i_ben]):
        r = state["residuals"][idx]
        px, py = X_pca[idx]
        arr.set_position((px, py))
        arr.xy = (px + r * px * arrow_scale, py + r * py * arrow_scale)
        arr.set_alpha(min(1.0, 0.25 + abs(r)))

    sample_line = (
        f"SGD sample: #{state['sample_idx']}"
        if state["sample_idx"] is not None
        else "final step (batch grad)"
    )
    acc = np.mean((sigmoid(X_aug @ state["w"]) >= 0.5) == y)
    text_box.set_text(
        f"step {frame}/{n_steps}\n"
        f"{sample_line}\n"
        f"$\\|\\nabla_w \\mathcal{{L}}\\| = {state['grad_norm']:.3f}$\n"
        f"train acc = {acc:.1%}\n"
        f"malignant: $p={state['p'][i_mal]:.3f}$, $y-p={state['residuals'][i_mal]:+.3f}$\n"
        f"benign: $p={state['p'][i_ben]:.3f}$, $y-p={state['residuals'][i_ben]:+.3f}$"
    )
    return [
        boundary_line, text_box, parallelogram_h, parallelogram_v,
        anchor_dot, w_arrow, step_arrow, sum_arrow, *residual_arrows,
    ]

anim = animation.FuncAnimation(
    fig, update, frames=len(history), init_func=init, blit=False, interval=240,
)
plt.close(fig)

HTML(f'<div style="max-width:520px">{anim.to_jshtml()}</div>')
```

As can be seen from the fit, the logistic regression model acheives an accuracy of $93\%$ for this dataset. Further, we have used only the first 2 PC as features but the model can easily be extended to using all the features for training.

### Multiclass Logistic Regression
---

Logistic regression in its basic form targets **binary** classification. Many real problems have **more than two classes**, so we extend the model in two common ways:

a) **Softmax (multinomial) regression** — one joint model over all classes, or  
b) **One-vs-Rest (OvR)** — train $K$ separate binary classifiers, one per class.

This section focuses on softmax regression. OvR can leave **ambiguous regions** where several classifiers assign high scores and there is no single clear winner.

In softmax regression we assign one weight vector $w_k \in \mathbb{R}^d$ to each class $k \in \{1,\ldots,K\}$ and model the class-conditional probability as

$$
p(C_k \mid x, w_1, \ldots, w_K) = \frac{\exp(w_k^\top x)}{\sum_{j=1}^{K} \exp(w_j^\top x)}
$$

This choice corresponds to a **multinomial likelihood**. Given a dataset $\mathcal{D} = \{x_1, y_1, \ldots, x_n, y_n\}$, where $y_i \in \{1,\ldots,K\}$, we write the likelihood of the labels given the inputs and weights as

$$
p(y_1, \ldots, y_n \mid x_1, \ldots, x_n, w_1, \ldots, w_K) = \prod_{i=1}^{n} p(y_i \mid x_i, w_1, \ldots, w_K)
$$

For each example, letting $p_{k,x_i} = p(C_k \mid x_i, w_1, \ldots, w_K)$ and using a one-hot indicator $\mathbb{1}[y_i = k]$,

$$
p(y_i \mid x_i, w_1, \ldots, w_K) = \prod_{k=1}^{K} p_{k,x_i}^{\mathbb{1}[y_i = k]}
$$

Equivalently, only the probability of the true class appears in the product: if $y_i = c$, then $p(y_i \mid x_i, \ldots) = p_{c,x_i}$.

You can replace that closing sentence with something like this, mirroring the binary $\mathcal{L}$ block:

As in the binary case, we fit $w_1, \ldots, w_K$ by maximizing the log-likelihood. The objective $\mathcal{L}$ is

$$
\mathcal{L} = \underset{w_1, \ldots, w_K}{\mathrm{argmax}} \sum_{i=1}^{n} \log p(y_i \mid x_i, w_1, \ldots, w_K)
$$

Substituting the per-example likelihood from above,

$$
\mathcal{L} = \underset{w_1, \ldots, w_K}{\mathrm{argmax}} \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}[y_i = k] \log p_{k,x_i}
$$

where $p_{k,x_i} = \dfrac{\exp(w_k^\top x_i)}{\sum_{j=1}^{K} \exp(w_j^\top x_i)}$. If $y_i = c$, each term reduces to $\log p_{c,x_i}$, so only the true class contributes.

Equivalently, minimizing cross-entropy is the same problem:

$$
\underset{w_1, \ldots, w_K}{\mathrm{argmin}} \; -\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}[y_i = k] \log p_{k,x_i}
$$

This objective is differentiable but has no closed-form solution, so we optimize it iteratively (e.g. gradient ascent on $\mathcal{L}$).


```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_iris()
list(data.target_names)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(data.data))
ev = pca.explained_variance_ratio_

df = pd.DataFrame(
    {
        "PC 1": X_pca[:, 0],
        "PC 2": X_pca[:, 1],
        "Species": pd.Categorical(
            data.target_names[data.target],
            categories=data.target_names,
        ),
    }
)

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.85,
    rc={
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)

palette = {
    "setosa": "#e74c3c",
    "versicolor": "#3498db",
    "virginica": "#2ecc71",
}

g = sns.jointplot(
    data=df,
    x="PC 1",
    y="PC 2",
    hue="Species",
    palette=palette,
    kind="scatter",
    height=7,
    ratio=4,
    space=0.08,
    alpha=0.8,
    s=60,
    edgecolor="white",
    linewidth=0.5,
    marginal_kws={"fill": True, "alpha": 0.45, "linewidth": 1.4},
)

g.ax_joint.set_facecolor("#fafafa")
g.ax_joint.set_xlabel(f"PC 1 ({ev[0]:.1%} explained variance)", labelpad=10)
g.ax_joint.set_ylabel(f"PC 2 ({ev[1]:.1%} explained variance)", labelpad=10)
g.fig.suptitle(
    "Iris — 2D PCA projection",
    y=1.1,
    fontsize=16,
    weight="semibold",
)
g.fig.text(
    0.5,
    1.005,
    f"{len(df):,} flowers · {ev.sum():.1%} variance captured in 2 components",
    ha="center",
    fontsize=11,
    color="#555555",
)
sns.move_legend(
    g.ax_joint,
    "upper right",
    title="Species",
    bbox_to_anchor=(1.0, 1.0),
    frameon=True,
    framealpha=0.92,
)
plt.show()
```


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

K = len(data.target_names)
y = data.target.astype(int)
X_aug = np.hstack([X_pca, np.ones((len(X_pca), 1))])
rng = np.random.default_rng(7)

lr = 0.5
n_steps = 120

ANIM_FIGSIZE = (6, 6)
ANIM_DPI = 80

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    Z = np.clip(Z, -500, 500)
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)

def predict_proba(W, X_aug):
    return softmax(X_aug @ W.T)

def grad(W, X_aug, y):
    P = predict_proba(W, X_aug)
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return (Y - P).T @ X_aug

def boundary_segment(w_diff, xlim, ylim):
    """Line where w_diff^T x = 0, with x = [pc1, pc2, 1]."""
    w1, w2, b = w_diff
    if abs(w2) > 1e-8:
        xs = np.array(xlim)
        ys = -(b + w1 * xs) / w2
        return xs, ys
    x = -b / (w1 + 1e-8)
    return np.array([x, x]), np.array(ylim)

def plot_decision_regions(ax, W, xlim, ylim, alpha=0.14):
    xx, yy = np.meshgrid(
        np.linspace(*xlim, 250),
        np.linspace(*ylim, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
    pred = (grid @ W.T).argmax(axis=1).reshape(xx.shape)
    region_colors = [palette[name] for name in data.target_names]
    return ax.contourf(
        xx, yy, pred,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=region_colors,
        alpha=alpha,
        zorder=1,
    )

def plot_pairwise_boundaries(ax, W, xlim, ylim, lw=2.0):
    lines = []
    for i in range(K):
        for j in range(i + 1, K):
            bx, by = boundary_segment(W[i] - W[j], xlim, ylim)
            (ln,) = ax.plot(bx, by, color="black", lw=lw, ls="-", zorder=6)
            lines.append(ln)
    return lines

def boundary_center(W, xlim, ylim):
    # centroid of the three pairwise boundaries
    pts = []
    for i in range(K):
        for j in range(i + 1, K):
            xs, ys = boundary_segment(W[i] - W[j], xlim, ylim)
            pts.append([0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1])])
    return np.mean(pts, axis=0)

# --- SGD fit in PCA space ---
W = rng.normal(scale=0.05, size=(K, 3))
history = []
W_curr = W.copy()

for step in range(n_steps + 1):
    P = predict_proba(W_curr, X_aug)
    residuals = np.zeros_like(P)
    residuals[np.arange(len(y)), y] = 1.0
    residuals -= P

    if step < n_steps:
        i = int(rng.integers(len(y)))
        g_step = grad(W_curr, X_aug[i : i + 1], y[i : i + 1])
    else:
        i = None
        g_step = grad(W_curr, X_aug, y)

    history.append(
        {
            "W": W_curr.copy(),
            "grad": g_step.copy(),
            "P": P.copy(),
            "residuals": residuals.copy(),
            "grad_norm": np.linalg.norm(g_step),
            "sample_idx": i,
            "sample_class": None if i is None else int(y[i]),
        }
    )
    if step < n_steps:
        W_curr = W_curr + lr * g_step

W_fit = history[-1]["W"]
xlim = (df["PC 1"].min() - 0.5, df["PC 1"].max() + 0.5)
ylim = (df["PC 2"].min() - 0.5, df["PC 2"].max() + 0.5)

# --- overlay fitted decision regions + boundaries on PCA plot ---
plot_decision_regions(g.ax_joint, W_fit, xlim, ylim, alpha=0.16)
plot_pairwise_boundaries(g.ax_joint, W_fit, xlim, ylim, lw=2.4)
g.ax_joint.set_xlim(*xlim)
g.ax_joint.set_ylim(*ylim)
g.fig.suptitle(
    "Iris — PCA + softmax decision boundaries",
    y=1.03,
    fontsize=16,
    weight="semibold",
)
g.fig.texts[0].set_text(
    f"{len(df):,} flowers · SGD ({n_steps} steps, lr={lr}) · {ev.sum():.1%} variance in 2 PCs"
)
handles, labels = g.ax_joint.get_legend_handles_labels()
g.ax_joint.legend(
    handles=handles + [
        plt.Line2D([0], [0], color="black", lw=2.4, label=r"$w_i^\top x = w_j^\top x$")
    ],
    title="Species",
    loc="upper right",
    frameon=True,
    framealpha=0.92,
)
plt.show()

# --- animated fitting ---
max_vec = max(
    max(np.linalg.norm(h["W"][:, :2]) for h in history),
    max(np.linalg.norm(lr * h["grad"][:, :2]) for h in history),
    1e-6,
)
vec_scale = 0.55 / max_vec

class_idx = {k: np.where(y == k)[0] for k in range(K)}

fig, ax = plt.subplots(figsize=ANIM_FIGSIZE, dpi=ANIM_DPI)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal")
ax.grid(True, alpha=0.25)
ax.set_xlabel(f"PC 1 ({ev[0]:.1%} variance)", fontsize=9)
ax.set_ylabel(f"PC 2 ({ev[1]:.1%} variance)", fontsize=9)
ax.set_title(
    r"Decision regions, $w_k$ (sample class), and $lr\,\nabla_W \mathcal{L}$"
    "\n(SGD softmax on PCA features)",
    fontsize=10,
)

for label, name in enumerate(data.target_names):
    mask = y == label
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=palette[name],
        s=20,
        alpha=0.45,
        edgecolors="white",
        linewidths=0.4,
        label=name,
        zorder=3,
    )

region_art = {"obj": None}
boundary_lines = [ax.plot([], [], "k-", lw=1.8)[0] for _ in range(3)]
anchor_dot = ax.scatter([], [], c="k", s=24, zorder=5)

residual_arrows = []
for _ in range(K):
    arr = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#f39c12", alpha=0.85),
    )
    residual_arrows.append(arr)

w_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2.4, color="#8e44ad"),
)
step_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2.4, color="#27ae60"),
)
sum_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2.4, color="#e67e22", linestyle="--"),
)
parallelogram_h, = ax.plot([], [], ":", color="gray", lw=1.0, alpha=0.8)
parallelogram_v, = ax.plot([], [], ":", color="gray", lw=1.0, alpha=0.8)

text_box = ax.text(
    0.02, 0.98, "",
    transform=ax.transAxes, va="top", fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)

ax.legend(
    handles=[
        plt.Line2D([0], [0], color="#8e44ad", lw=3, label=r"$w_{k,1:2}$ (sample class)"),
        plt.Line2D([0], [0], color="#27ae60", lw=3, label=r"$lr\,\nabla_{w_k}\mathcal{L}$"),
        plt.Line2D([0], [0], color="#e67e22", lw=3, ls="--", label=r"$w_k + lr\,\nabla_{w_k}\mathcal{L}$"),
    ],
    loc="lower right",
    fontsize=9,
)

def _set_arrow(ann, origin, vec):
    tip = origin + vec
    ann.xy = (tip[0], tip[1])
    ann.set_position((origin[0], origin[1]))

def _top_residual_indices(P, residuals, y):
    idxs = []
    for k in range(K):
        idx_k = class_idx[k]
        r_k = residuals[idx_k, k]
        idxs.append(idx_k[np.argmax(np.abs(r_k))])
    return idxs

def init():
    if region_art["obj"] is not None:
        region_art["obj"].remove()
        region_art["obj"] = None
    for ln in boundary_lines:
        ln.set_data([], [])
    parallelogram_h.set_data([], [])
    parallelogram_v.set_data([], [])
    anchor_dot.set_offsets(np.empty((0, 2)))
    text_box.set_text("")
    return [
        text_box, parallelogram_h, parallelogram_v, anchor_dot,
        w_arrow, step_arrow, sum_arrow, *boundary_lines, *residual_arrows,
    ]

def update(frame):
    state = history[frame]
    Wf = state["W"]

    if region_art["obj"] is not None:
        region_art["obj"].remove()
    region_art["obj"] = plot_decision_regions(ax, Wf, xlim, ylim, alpha=0.18)

    pair = [(0, 1), (0, 2), (1, 2)]
    for ln, (i, j) in zip(boundary_lines, pair):
        bx, by = boundary_segment(Wf[i] - Wf[j], xlim, ylim)
        ln.set_data(bx, by)

    anchor = boundary_center(Wf, xlim, ylim)
    anchor_dot.set_offsets(anchor.reshape(1, -1))

    k = state["sample_class"] if state["sample_class"] is not None else int(np.argmax(np.linalg.norm(state["grad"], axis=1)))
    w_xy = Wf[k, :2] * vec_scale
    step_xy = lr * state["grad"][k, :2] * vec_scale
    sum_xy = w_xy + step_xy

    _set_arrow(w_arrow, anchor, w_xy)
    _set_arrow(step_arrow, anchor, step_xy)
    _set_arrow(sum_arrow, anchor, sum_xy)

    parallelogram_h.set_data(
        [anchor[0] + w_xy[0], anchor[0] + sum_xy[0]],
        [anchor[1] + w_xy[1], anchor[1] + sum_xy[1]],
    )
    parallelogram_v.set_data(
        [anchor[0] + step_xy[0], anchor[0] + sum_xy[0]],
        [anchor[1] + step_xy[1], anchor[1] + sum_xy[1]],
    )

    idxs = _top_residual_indices(state["P"], state["residuals"], y)
    arrow_scale = 0.35
    for arr, idx in zip(residual_arrows, idxs):
        true_k = y[idx]
        r = state["residuals"][idx, true_k]
        px, py = X_pca[idx]
        arr.set_position((px, py))
        arr.xy = (px + r * px * arrow_scale, py + r * py * arrow_scale)
        arr.set_alpha(min(1.0, 0.25 + abs(r)))

    sample_line = (
        f"SGD sample: #{state['sample_idx']} ({data.target_names[state['sample_class']]})"
        if state["sample_idx"] is not None
        else "final step (batch grad)"
    )
    pred = predict_proba(Wf, X_aug).argmax(axis=1)
    acc = np.mean(pred == y)

    prob_lines = []
    for idx in idxs:
        k_true = y[idx]
        prob_lines.append(
            f"{data.target_names[k_true]}: "
            f"$p={state['P'][idx, k_true]:.3f}$, "
            f"$1-p={state['residuals'][idx, k_true]:+.3f}$"
        )

    text_box.set_text(
        f"step {frame}/{n_steps}\n"
        f"{sample_line}\n"
        f"$\\|\\nabla_W \\mathcal{{L}}\\| = {state['grad_norm']:.3f}$\n"
        f"train acc = {acc:.1%}\n"
        + "\n".join(prob_lines)
    )

    return [
        region_art["obj"], text_box, parallelogram_h, parallelogram_v,
        anchor_dot, w_arrow, step_arrow, sum_arrow,
        *boundary_lines, *residual_arrows,
    ]

anim = animation.FuncAnimation(
    fig, update, frames=len(history), init_func=init, blit=False, interval=240,
)
plt.close(fig)

HTML(f'<div style="max-width:520px">{anim.to_jshtml()}</div>')
```

### Conclusion
---

This notebook walked through logistic regression from definition to fitting on real data.

**Binary classification.** We model $p(y=1 \mid x, w) = \sigma(w^\top x)$ instead of predicting unbounded real values. That matches a Bernoulli likelihood and yields **linear decision boundaries** ($w^\top x = 0$). Ordinary least squares is a poor fit for binary labels because it assumes Gaussian noise and can produce predictions outside $[0,1]$. We fit $w$ by maximizing log-likelihood (equivalently minimizing cross-entropy). The objective is smooth and concave, but has **no closed-form solution**, so we use iterative methods such as SGD. The toy animation showed the core mechanism: each point pulls $w$ in proportion to the residual $y_i - p_i$; confident correct predictions contribute little, while misclassified or uncertain points drive larger updates.

**Multiclass extension.** For $K$ classes, softmax regression assigns one weight vector $w_k$ per class and models
$p(C_k \mid x) \propto \exp(w_k^\top x)$. Training still maximizes log-likelihood over all classes jointly. Decision boundaries arise where two classes tie ($w_i^\top x = w_j^\top x$), so the classifier remains **linear in the input features**. One-vs-rest is an alternative, but can produce ambiguous regions where multiple binary classifiers disagree; softmax avoids that by learning a single coherent probability distribution over classes.

**Experiments.** On the Wisconsin breast cancer dataset projected to two principal components, the classes are largely separable and logistic regression reached about **93% training accuracy** with a simple SGD fit — a reasonable baseline before using all original features. On Iris, softmax regression in 2D PCA space separates setosa cleanly but versicolor and virginica overlap somewhat, reflecting the limit of **linear boundaries in a low-dimensional projection**. Both demos used PCA for visualization; the same models apply directly in the full feature space.

**Takeaways.** Logistic regression is a strong, interpretable baseline for classification: probabilities are calibrated, boundaries are linear, and optimization is straightforward. Its main limitation is that decision surfaces are hyperplanes so when classes are not linearly separable,generalized linear models can be used to transform the input into a higer dimensional space where they might be linearlly separable or project it down to a lower dimensional space.

Similar to linear regression, logistic regression can also be regularized by adding an **$L_2$ norm penalty** on the weight vectors.


