import numpy as np
import pandas as pd
from math import ceil
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------- CENTRAL CONFIG ----------
@dataclass
class PlotConfig:
    base_fontsize: int = 14     # master knob — change this most of the time
    title_delta: int = 4        # title size = base + title_delta
    axis_label_delta: int = 2   # axis labels size = base + axis_label_delta
    tick_label_delta: int = 0   # tick labels size = base + tick_label_delta
    legend_delta: int = 0       # legend size = base + legend_delta
    bar_label_delta: int = 0    # numbers printed on/near bars = base + bar_label_delta
    suptitle_delta: int = 6     # suptitle size (non-binary grid)

    @property
    def title_size(self): return self.base_fontsize + self.title_delta
    @property
    def axis_label_size(self): return self.base_fontsize + self.axis_label_delta
    @property
    def tick_label_size(self): return self.base_fontsize + self.tick_label_delta
    @property
    def legend_size(self): return self.base_fontsize + self.legend_delta
    @property
    def bar_label_size(self): return self.base_fontsize + self.bar_label_delta
    @property
    def sup_title_size(self): return self.base_fontsize + self.suptitle_delta

CONFIG = PlotConfig(base_fontsize=14)

# ---------- BINARY DETECTION / HELPERS ----------
YES_NO_MAP = {"yes":1,"no":0,"y":1,"n":0,"true":1,"false":0,"t":1,"f":0,"1":1,"0":0}

def is_binary_series(s: pd.Series) -> bool:
    v = s.dropna()
    if v.empty:
        return False
    if pd.api.types.is_bool_dtype(v):
        return True
    if pd.api.types.is_numeric_dtype(v):
        u = pd.unique(v[np.isfinite(v)])
        return set(np.unique(u)).issubset({0, 1})
    if pd.api.types.is_object_dtype(v) or pd.api.types.is_string_dtype(v):
        norm = v.astype(str).str.strip().str.lower()
        u = set(norm.unique())
        if u.issubset({"0","1"}):
            return True
        if len(u) == 2 and all(x in YES_NO_MAP for x in u):
            mapped = {YES_NO_MAP[x] for x in u}
            return mapped == {0,1}
    return False

def to_bool_series(s: pd.Series) -> pd.Series:
    v = s.copy()
    if pd.api.types.is_bool_dtype(v):
        return v
    if pd.api.types.is_numeric_dtype(v):
        v = pd.to_numeric(v, errors="coerce")
        v = v.where(v.isin([0, 1]))
        return v.map({0: False, 1: True})
    v = v.astype(str).str.strip().str.lower()
    v = v.map(YES_NO_MAP).map({0: False, 1: True})
    return v

def get_non_binary_columns(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    exclude = set(exclude or [])
    return [c for c in df.columns if c not in exclude and not is_binary_series(df[c])]

# ---------- PLOTS ----------
def plot_binary_columns_with_counts(
    df: pd.DataFrame,
    binary_cols: list[str],
    title: str = "Binary Feature Distribution",
    cfg: PlotConfig = CONFIG,
):
    # Build counts
    rows = []
    for col in binary_cols:
        b = to_bool_series(df[col])
        counts = b.value_counts(dropna=True)
        f_count = int(counts.get(False, 0))
        t_count = int(counts.get(True, 0))
        total = f_count + t_count
        true_rate = (t_count / total) if total else np.nan
        rows.append((col, f_count, t_count, total, true_rate))

    counts_df = pd.DataFrame(rows, columns=["feature", "false_count", "true_count", "total", "true_rate"])
    counts_df = counts_df.sort_values("true_rate", ascending=False)

    features  = counts_df["feature"].to_list()
    false_vals = counts_df["false_count"].to_numpy()
    true_vals  = counts_df["true_count"].to_numpy()
    totals     = false_vals + true_vals

    height = max(6, 0.4 * len(features) + 2)
    fig, ax = plt.subplots(figsize=(16, height))

    bars_false = ax.barh(features, false_vals, label="False")
    bars_true  = ax.barh(features, true_vals, left=false_vals, label="True")

    ax.tick_params(axis='y', labelsize=cfg.tick_label_size)
    ax.tick_params(axis='x', labelsize=cfg.tick_label_size)
    ax.set_title(title, fontsize=cfg.title_size, fontweight="bold")
    ax.set_xlabel("Count", fontsize=cfg.axis_label_size)
    ax.legend(loc="best", fontsize=cfg.legend_size)
    ax.invert_yaxis()

    # Annotator with larger labels
    def annotate_segments_inside(bars, values, totals,
                                 min_abs_pad=1.5, rel_pad=0.03, fontsize=None):
        fontsize = fontsize or cfg.bar_label_size
        for i, (rect, val) in enumerate(zip(bars, values)):
            if val <= 0:
                continue
            x = rect.get_x()
            w = rect.get_width()
            y = rect.get_y()
            h = rect.get_height()
            cy = y + h / 2.0

            pad = max(totals[i] * rel_pad, min_abs_pad)
            narrow = (w < max(totals[i] * 0.05, 10))
            if narrow:
                x_text = x + min(pad, max(w * 0.1, 0.0))
                ha = "left"
            else:
                x_text = x + w / 2.0
                ha = "center"

            rect.axes.text(
                x_text, cy, f"{int(val)}",
                va="center", ha=ha,
                fontsize=fontsize,
                fontweight="bold",
                color="black",
                clip_on=True
            )

    annotate_segments_inside(bars_false, false_vals, totals, fontsize=cfg.bar_label_size)
    annotate_segments_inside(bars_true,  true_vals,  totals, fontsize=cfg.bar_label_size)

    # Total at far right
    for i, total in enumerate(totals):
        ax.text(
            false_vals[i] + true_vals[i] + max(total * 0.005, 1.0),
            i, f"{int(total)}",
            va="center", ha="left",
            fontsize=cfg.bar_label_size,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.show()

def plot_non_binary_features(
    df: pd.DataFrame,
    exclude: list[str] = None,
    n_features: int = 12,
    max_cats: int = 20,
    bins: int = 30,
    title: str = "Non-binary Features Overview",
    cfg: PlotConfig = CONFIG,
):
    non_binary = get_non_binary_columns(df, exclude=exclude)
    if not non_binary:
        raise ValueError("No non-binary columns found after applying exclusions.")

    selected = non_binary[:n_features]

    n = len(selected)
    cols = 4
    rows = ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.8*cols, 3.6*rows), squeeze=False)
    fig.suptitle(title, fontsize=cfg.sup_title_size, fontweight="bold", y=0.98)

    for i, col in enumerate(selected):
        r, c = divmod(i, cols)
        ax = axes[r][c]

        s = df[col]
        v = s.dropna()

        if v.empty:
            ax.text(0.5, 0.5, "All NA", ha="center", va="center", fontsize=cfg.bar_label_size)
            ax.set_title(col, fontsize=cfg.axis_label_size)
            ax.axis("off")
            continue

        if pd.api.types.is_numeric_dtype(v):
            v = pd.to_numeric(v, errors="coerce").dropna()
            ax.hist(v, bins=bins)
            ax.set_xlabel("Value", fontsize=cfg.axis_label_size)
            ax.set_ylabel("Frequency", fontsize=cfg.axis_label_size)
        else:
            vc = v.astype(str).value_counts()
            if len(vc) > max_cats:
                top = vc.iloc[:max_cats]
                other = int(vc.iloc[max_cats:].sum())
                vc = pd.concat([top, pd.Series({"__OTHER__": other})])
            vc.plot(kind="bar", ax=ax)
            ax.set_xlabel("Category", fontsize=cfg.axis_label_size)
            ax.set_ylabel("Count", fontsize=cfg.axis_label_size)
            # annotate categorical bar counts
            for p in ax.patches:
                h = p.get_height()
                if np.isnan(h) or h <= 0:
                    continue
                x = p.get_x() + p.get_width() / 2.0
                y = p.get_y() + h
                ax.text(x, y, f"{int(h)}", ha="center", va="bottom", fontsize=cfg.bar_label_size)

        ax.set_title(col, fontsize=cfg.axis_label_size, pad=8)
        ax.tick_params(axis='x', labelrotation=45, labelsize=cfg.tick_label_size)
        ax.tick_params(axis='y', labelsize=cfg.tick_label_size)

    # Hide unused subplots
    total_axes = rows * cols
    for j in range(n, total_axes):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

@dataclass
class DatasetSpec:
    name: str                 # e.g., "2018"
    categorical_path: Path    # full path to the categorical HDF

def build_specs_from_years(
    data_folder: Path,
    years: list[str],
    categorical_template: str = "filtered_categorical_{year}.h5",
) -> list[DatasetSpec]:
    specs = []
    for y in years:
        cat = data_folder / categorical_template.format(year=y)
        specs.append(DatasetSpec(name=y, categorical_path=cat))
    return specs

def build_specs_explicit(pairs: list[tuple[str, str]], data_folder: Path) -> list[DatasetSpec]:
    """
    pairs: list of (name, categorical_filename)
    """
    specs = []
    for name, cat_fname in pairs:
        specs.append(DatasetSpec(name=name, categorical_path=(data_folder / cat_fname)))
    return specs

# ---------- Processing one dataset ----------
def process_dataset(spec: DatasetSpec, exclude_cols: list[str]):
    if not spec.categorical_path.exists():
        raise FileNotFoundError(f"Categorical file not found: {spec.categorical_path}")

    df = pd.read_hdf(spec.categorical_path)

    # Binary plot
    binary_cols = [c for c in df.columns if is_binary_series(df[c])]
    print(f"[{spec.name}] Binary columns ({len(binary_cols)}): {binary_cols}")
    plot_binary_columns_with_counts(
        df,
        binary_cols,
        title=f"{spec.name} — Binary Feature Distribution",
        cfg=CONFIG,
    )

    # Non-binary plot
    plot_non_binary_features(
        df,
        exclude=exclude_cols,
        n_features=12,
        title=f"{spec.name} — Non-binary Feature Distribution",
        cfg=CONFIG,
    )

if __name__ == '__main__':
    data_folder = Path(__file__).resolve().parent / "data"

    specs = build_specs_explicit(
        pairs=[("2018", "filtered_categorical_2018.h5"),
               ("2011", "filtered_categorical_2011.h5")],
        data_folder=data_folder
    )

    exclude_cols = ["Household", "HouseholdID", "MeterID", 'Mergeid', 'Plz', 'Einzgdat', 'Wohnort', 'Meter1', 'Meter2']

    # Iterate all datasets
    for spec in specs:
        process_dataset(spec, exclude_cols=exclude_cols)