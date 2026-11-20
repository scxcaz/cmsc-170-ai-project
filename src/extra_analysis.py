import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import prepare_data
from src import evaluation
from src.models.linear_regression import (
    train, predict_batch, budgets_to_millions, millions_to_pesos,
)
from src.models.decision_tree import DecisionTreeRegressor
from src.models.random_forest import RandomForestRegressor

OUT = Path("results")
COLORS = {"Linear Regression": "#1f6f8b", "Decision Tree": "#c8553d", "Random Forest": "#3d8b5a"}


def metrics(y, p):
    return {
        "MAE": evaluation.mean_absolute_error(y, p),
        "RMSE": evaluation.root_mean_squared_error(y, p),
        "R2": evaluation.r_squared(y, p),
    }


def main():
    X_tr, y_tr, X_te, y_te = (np.asarray(a, dtype=float) for a in prepare_data())
    OUT.mkdir(exist_ok=True)
    extra = {}

    # same settings as src/comparison.py
    w, b = train(X_tr, budgets_to_millions(y_tr), learning_rate=0.05, iterations=200)
    preds = {"Linear Regression": millions_to_pesos(predict_batch(X_te, w, b))}
    dt = DecisionTreeRegressor(max_depth=16, min_samples_split=20, min_samples_leaf=10)
    dt.fit(X_tr, y_tr)
    preds["Decision Tree"] = dt.predict(X_te)
    rf = RandomForestRegressor(n_trees=5, max_depth=8, min_samples_split=20,
                               min_samples_leaf=10, max_features=5, random_state=42)
    rf.fit(X_tr, y_tr)
    preds["Random Forest"] = rf.predict(X_te)
    preds = {k: np.asarray(v, dtype=float) for k, v in preds.items()}

    base = np.full_like(y_te, np.median(y_tr))
    extra["Median baseline"] = metrics(y_te, base)
    extra["Mean baseline"] = metrics(y_te, np.full_like(y_te, y_tr.mean()))
    extra["budget_stats"] = {
        "mean": float(y_tr.mean()), "median": float(np.median(y_tr)),
        "p90": float(np.percentile(y_tr, 90)), "p99": float(np.percentile(y_tr, 99)),
        "max": float(y_tr.max()),
    }

    # 1. budget distribution (log scale)
    plt.figure(figsize=(8, 5))
    plt.hist(np.log10(y_tr), bins=60, color="#1f6f8b")
    plt.axvline(np.log10(np.median(y_tr)), color="k", ls="--", label="median")
    plt.axvline(np.log10(y_tr.mean()), color="#c8553d", ls="--", label="mean")
    plt.xlabel("log10(budget in PHP)"); plt.ylabel("Projects")
    plt.title("Budget distribution (training set)"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "budget_distribution.png", dpi=150); plt.close()

    # 2. predicted vs actual
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_te), 4000, replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, (n, p) in zip(axes, preds.items()):
        ax.scatter(y_te[idx] / 1e6, np.clip(p[idx], 1, None) / 1e6, s=5, alpha=.35, color=COLORS[n])
        ax.plot([0.01, 5000], [0.01, 5000], "k--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(n)
        ax.set_xlabel("Actual (M PHP)")
    axes[0].set_ylabel("Predicted (M PHP)")
    plt.tight_layout(); plt.savefig(OUT / "predicted_vs_actual.png", dpi=150); plt.close()

    # 3. absolute-error distributions
    plt.figure(figsize=(8, 5))
    data = [np.abs(y_te - p) / 1e6 for p in preds.values()]
    plt.boxplot(data, tick_labels=list(preds), showfliers=False)
    plt.ylabel("Absolute error (M PHP), outliers hidden")
    plt.title("Typical vs. extreme error"); plt.tight_layout()
    plt.savefig(OUT / "abs_error_boxplot.png", dpi=150); plt.close()
    extra["median_abs_error"] = {n: float(np.median(np.abs(y_te - p))) for n, p in preds.items()}
    extra["pct_within_10M"] = {n: float(np.mean(np.abs(y_te - p) < 1e7)) for n, p in preds.items()}

    # 4. tree depth sweep
    depths = [2, 4, 8, 12, 16, 20]
    sweep = []
    for d in depths:
        t = DecisionTreeRegressor(max_depth=d, min_samples_split=20, min_samples_leaf=10)
        t.fit(X_tr, y_tr)
        sweep.append({"depth": d, "test": metrics(y_te, np.asarray(t.predict(X_te), float)),
                      "train": metrics(y_tr, np.asarray(t.predict(X_tr), float))})
    extra["depth_sweep"] = sweep
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(depths, [s["train"]["R2"] for s in sweep], "o-", label="train")
    ax[0].plot(depths, [s["test"]["R2"] for s in sweep], "o-", label="test")
    ax[0].set_xlabel("max depth"); ax[0].set_ylabel("R²"); ax[0].legend(); ax[0].set_title("R² vs depth")
    ax[1].plot(depths, [s["test"]["MAE"] / 1e6 for s in sweep], "o-", color="#c8553d")
    ax[1].set_xlabel("max depth"); ax[1].set_ylabel("Test MAE (M PHP)"); ax[1].set_title("MAE vs depth")
    plt.tight_layout(); plt.savefig(OUT / "depth_sweep.png", dpi=150); plt.close()

    # 5. forest size sweep
    counts = [1, 5, 10, 20]
    fs = []
    for n in counts:
        f = RandomForestRegressor(n_trees=n, max_depth=8, min_samples_split=20,
                                  min_samples_leaf=10, max_features=5, random_state=42)
        f.fit(X_tr, y_tr)
        fs.append({"trees": n, **metrics(y_te, np.asarray(f.predict(X_te), float))})
    extra["forest_sweep"] = fs
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(counts, [s["R2"] for s in fs], "o-", color="#3d8b5a"); ax[0].set_xlabel("trees"); ax[0].set_ylabel("Test R²")
    ax[1].plot(counts, [s["MAE"] / 1e6 for s in fs], "o-", color="#3d8b5a"); ax[1].set_xlabel("trees"); ax[1].set_ylabel("Test MAE (M PHP)")
    plt.tight_layout(); plt.savefig(OUT / "forest_sweep.png", dpi=150); plt.close()

    with open(OUT / "extra_metrics.json", "w") as f:
        json.dump(extra, f, indent=2)
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main()
