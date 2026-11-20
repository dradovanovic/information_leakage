"""Aggregation over seeds and simple plots."""
import numpy as np
import pandas as pd


def summarise(df, group_cols, metric_cols):
    """Mean, std and an approximate 95% CI half-width (1.96 * std / sqrt(n)) per group.

    Note: Cloud et al. (2025, Fig. 10) draw 95% CIs of the mean over 100 runs. A std error bar
    over 100 runs is 10/1.96 = ~5x wider, so the two must not be compared directly.
    """
    g = df.groupby(group_cols)[metric_cols]
    mean, std, n = g.mean(), g.std(ddof=1), g.count()
    out = []
    for m in metric_cols:
        part = pd.DataFrame({f"{m}_mean": mean[m], f"{m}_std": std[m],
                             f"{m}_ci95": 1.96 * std[m] / np.sqrt(n[m]), "n_runs": n[m]})
        out.append(part)
    res = pd.concat(out, axis=1)
    return res.loc[:, ~res.columns.duplicated()].reset_index()


def bar_plot(summary, metric, order, path, title="", chance=None, ylabel=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary.set_index("model").reindex([o for o in order if o in set(summary["model"])])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, err, lab in [(axes[0], "ci95", "95% CI of the mean"), (axes[1], "std", "± 1 std (run-to-run)")]:
        ax.bar(range(len(s)), s[f"{metric}_mean"], yerr=s[f"{metric}_{err}"], capsize=5, color="#4C72B0")
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s.index, rotation=30, ha="right")
        ax.set_title(f"Error bars: {lab}")
        ax.grid(axis="y", alpha=0.3)
        if chance is not None:
            ax.axhline(chance, color="red", ls="--", lw=1.5, label="chance")
            ax.legend()
    axes[0].set_ylabel(ylabel or metric)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
