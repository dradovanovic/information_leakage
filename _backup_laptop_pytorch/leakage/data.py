"""Smart meter data: weekly snippets, household-level splits, synthetic data for smoke tests.

Input format (output of prepare_data.py):
  load h5        : DataFrame, MultiIndex (Household, Year_Week), columns = 7 * slots values
  categorical h5 : DataFrame with a `Household` column and one column per attribute
                   (-1 = missing / unknown, filtered out here)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    hh_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    hh_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    hh_test: np.ndarray
    n_classes: int


def load_weekly(load_path, cat_path, label):
    """Return (X [n_weeks, 7, slots], y [n_weeks], households [n_weeks]) for one attribute."""
    load = pd.read_hdf(load_path)
    cat = pd.read_hdf(cat_path)
    if label not in cat.columns:
        raise KeyError(f"Label '{label}' not in categorical data. Available: {sorted(c for c in cat.columns if c != 'Household')}")
    cat = cat.drop_duplicates("Household").set_index("Household")[label]
    cat = pd.to_numeric(cat, errors="coerce").fillna(-1).astype(int)
    cat = cat[cat >= 0]

    hh = load.index.get_level_values("Household").astype(int)
    keep = np.isin(hh, cat.index.values)
    load = load[keep]
    hh = hh[keep]
    values = load.to_numpy(dtype="float32")
    slots = values.shape[1] // 7
    x = values.reshape(len(values), 7, slots)
    # labels must be 0..K-1
    classes = np.sort(cat.unique())
    remap = {c: i for i, c in enumerate(classes)}
    y = cat.loc[hh].map(remap).to_numpy()
    return x, y, np.asarray(hh)


def to_hourly(x):
    """Aggregate (n, 7, slots) to (n, 7, 24) by summing kWh within each hour."""
    n, d, s = x.shape
    if s == 24:
        return x
    per_hour = s // 24
    return x.reshape(n, d, 24, per_hour).sum(axis=-1)


def household_split(x, y, hh, seed, test_size=0.2, val_size=0.1, max_weeks_per_household=None):
    """Stratified split on HOUSEHOLDS, so no household appears in two parts.

    Households are stratified by their label; each household keeps all its weeks
    (optionally subsampled to `max_weeks_per_household` for fast CPU runs).
    """
    rng = np.random.default_rng(seed)
    if max_weeks_per_household:
        idx = []
        for h in np.unique(hh):
            rows = np.flatnonzero(hh == h)
            if len(rows) > max_weeks_per_household:
                rows = rng.choice(rows, max_weeks_per_household, replace=False)
            idx.append(rows)
        idx = np.sort(np.concatenate(idx))
        x, y, hh = x[idx], y[idx], hh[idx]

    hh_labels = pd.Series(y, index=hh).groupby(level=0).first()
    h_ids, h_y = hh_labels.index.values, hh_labels.values
    h_trval, h_test = train_test_split(h_ids, test_size=test_size, random_state=seed, stratify=h_y)
    y_trval = hh_labels.loc[h_trval].values
    h_train, h_val = train_test_split(h_trval, test_size=val_size / (1 - test_size), random_state=seed,
                                      stratify=y_trval)

    def take(ids):
        m = np.isin(hh, ids)
        return x[m], y[m], hh[m]

    xtr, ytr, htr = take(h_train)
    xva, yva, hva = take(h_val)
    xte, yte, hte = take(h_test)
    assert not (set(htr) & set(hte)) and not (set(htr) & set(hva)), "household leakage between splits"
    return Split(xtr, ytr, htr, xva, yva, hva, xte, yte, hte, int(len(np.unique(y))))


def standardise(split, mode="global"):
    """Fit on training weeks only. 'global' = one mean/std, 'log' = log1p then global."""
    if mode == "none":
        return split
    tr = split.x_train
    if mode == "log":
        f = np.log1p
        tr = f(np.clip(tr, 0, None))
    else:
        f = lambda a: a  # noqa: E731
    mu, sd = float(tr.mean()), float(tr.std()) + 1e-8

    def tf_(a):
        a = f(np.clip(a, 0, None)) if mode == "log" else a
        return ((a - mu) / sd).astype("float32")

    split.x_train, split.x_val, split.x_test = tf_(split.x_train), tf_(split.x_val), tf_(split.x_test)
    return split


def synthetic_weekly(n_households=300, weeks=20, slots=24, positive_rate=0.3, effect=0.25, seed=0):
    """Fake hourly weeks with a planted binary attribute (larger evening peak if y = 1).

    Only for smoke tests on machines without the real data; numbers mean nothing.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(slots) * 24 / slots
    base = 0.3 + 0.4 * np.exp(-((hours - 19) ** 2) / 6) + 0.2 * np.exp(-((hours - 8) ** 2) / 3)
    y_h = (rng.random(n_households) < positive_rate).astype(int)
    scale_h = rng.lognormal(0, 0.4, n_households)
    xs, ys, hs = [], [], []
    for h in range(n_households):
        prof = base * scale_h[h] * (1 + effect * y_h[h] * np.exp(-((hours - 20) ** 2) / 4))
        wk = prof[None, None, :] * rng.lognormal(0, 0.3, (weeks, 7, slots))
        xs.append(wk.astype("float32"))
        ys.append(np.full(weeks, y_h[h]))
        hs.append(np.full(weeks, 10_000 + h))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(hs)
