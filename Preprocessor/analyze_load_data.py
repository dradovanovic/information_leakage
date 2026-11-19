import re
import h5py
import numpy as np
import pandas as pd
from math import ceil
from pathlib import Path
from datetime import date
from typing import Tuple, Dict
import matplotlib.pyplot as plt

def parse_id(id_str: str):
    s = str(id_str)
    m = re.search(r"_Y(?P<year>\d{4})_W(?P<week>\d{1,2})", s, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse ISO year/week from id: {s}")

    iso_year = int(m.group("year"))
    iso_week = int(m.group("week"))
    # Monday (1) start of that ISO week
    week_start = pd.Timestamp(date.fromisocalendar(iso_year, iso_week, 1))

    # Household = everything before "_Y..."
    household = s.split("_Y", 1)[0]
    return household, week_start, iso_year, iso_week

def compare_consumption_totals(
    load_input_profiles: pd.DataFrame,
    weekly_load_profiles: np.ndarray,   # shape: (n_weeks, 7, 96)
    ids_load_profiles: list[str],
    *,
    freq: str = "15min",
    week_freq: str = "W-SUN",           # weeks start Monday, end Sunday
    atol_kwh: float = 1e-3,             # absolute tolerance in kWh for equality
) -> Dict[str, pd.DataFrame]:

    # Ensure types
    df = load_input_profiles.copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])
    df["Consumption (kWh)"] = pd.to_numeric(df["Consumption (kWh)"], errors="coerce")

    # --- Build snippet per-(Household, week) sums ---
    parsed = [parse_id(x) for x in ids_load_profiles]
    hh_from_ids = [p[0] for p in parsed]
    week_starts = [p[1] for p in parsed]
    iso_years = [p[2] for p in parsed]
    iso_weeks = [p[3] for p in parsed]

    # Sum each snippet block (7x96) to weekly kWh
    weekly_sums = weekly_load_profiles.reshape(weekly_load_profiles.shape[0], -1).sum(axis=1)

    snippet_df = pd.DataFrame({
        "Household": hh_from_ids,
        "WeekStart": week_starts,  # e.g., 2017-10-02 00:00:00
        "ISOYear": iso_years,
        "ISOWeek": iso_weeks,
        "snippet_kwh": weekly_sums
    })

    # If WeekStart is missing in IDs, derive from an artificial Period index
    if snippet_df["WeekStart"].isna().any():
        # Fallback: create a weekly Period index in the same order, then use .start_time
        # (This assumes the ids are already sorted by week per household.)
        # We’ll compute per-household running week numbers and create monotonic week periods.
        # Safer way is to parse actual dates in IDs; prefer including dates when generating IDs.
        snippet_df["__ord"] = snippet_df.groupby("Household").cumcount()
        # Use an arbitrary anchor (1970-01-05 is a Monday); add 7 days per step
        anchor = pd.Timestamp("1970-01-05")  # Monday
        snippet_df["WeekStart"] = anchor + pd.to_timedelta(7 * snippet_df["__ord"], unit="D")
        snippet_df.drop(columns="__ord", inplace=True)

    # --- (1) WHOLE-RANGE / YEAR TOTALS ---
    raw_total = df.groupby("Household")["Consumption (kWh)"].sum()
    raw_total.index = raw_total.index.astype(int)

    snippet_total = (snippet_df
        .groupby("Household", as_index=True)["snippet_kwh"]
        .sum()
        .rename("snippet_total_kwh"))
    snippet_total.index = snippet_total.index.astype(int)

    # Align households seen in either source
    all_households = raw_total[raw_total.index.isin(snippet_total.index)].index
    year_compare = pd.DataFrame(index=all_households)
    year_compare["raw_total_kwh"] = raw_total.reindex(all_households)
    year_compare["snippet_total_kwh"] = snippet_total.reindex(all_households)
    year_compare["abs_diff_kwh"] = (year_compare["raw_total_kwh"] - year_compare["snippet_total_kwh"]).abs()
    year_compare["rel_diff_pct"] = (year_compare["abs_diff_kwh"] / year_compare["raw_total_kwh"]).replace([np.inf, np.nan], np.nan) * 100.0
    year_compare["match"] = year_compare["abs_diff_kwh"] <= atol_kwh

    # --- (2) WEEK-ALIGNED TOTALS (apples-to-apples) ---
    # For raw data: compute weekly sums on the same week alignment as snippets
    df["WeekPeriod"] = df["Time"].dt.to_period(week_freq)
    df["WeekStart"] = df["WeekPeriod"].dt.start_time   # Monday 00:00 for W-SUN
    raw_weekly = (df
        .groupby(["Household", "WeekStart"], as_index=False)["Consumption (kWh)"]
        .sum()
        .rename(columns={"Consumption (kWh)": "raw_week_kwh"}))

    # Now keep only the (Household, WeekStart) pairs that actually appear in snippets:
    key_snip = snippet_df[["Household", "WeekStart", "snippet_kwh"]]
    key_snip['Household'] = key_snip['Household'].astype(int)
    week_aligned = key_snip.merge(raw_weekly, on=["Household", "WeekStart"], how="left")

    # Compute diffs per-week and also per-household (sum over matched weeks)
    week_aligned["abs_diff_kwh"] = (week_aligned["raw_week_kwh"] - week_aligned["snippet_kwh"]).abs()
    week_aligned["match"] = week_aligned["abs_diff_kwh"] <= atol_kwh

    # Per-household summary over matched weeks
    per_hh_week_aligned = (week_aligned
        .groupby("Household", as_index=True)[["snippet_kwh", "raw_week_kwh", "abs_diff_kwh"]]
        .sum()
        .rename(columns={"snippet_kwh": "snippet_sum_matched", "raw_week_kwh": "raw_sum_matched"}))
    per_hh_week_aligned["rel_diff_pct"] = (
        (per_hh_week_aligned["abs_diff_kwh"] / per_hh_week_aligned["raw_sum_matched"])
        .replace([np.inf, np.nan], np.nan) * 100.0
    )
    per_hh_week_aligned["match"] = (per_hh_week_aligned["abs_diff_kwh"] <= atol_kwh)

    return {
        "total_year": year_compare.sort_index(),
        "week_aligned_pairs": week_aligned.sort_values(["Household", "WeekStart"]),
        "week_aligned_per_household": per_hh_week_aligned.sort_index(),
    }

if __name__ == '__main__':
    data_folder = Path(__file__).resolve().parent / "data"
    load_name_weekly = 'weekly_structure_2018.h5'
    load_name = 'input_load_profiles_2018.h5'
    load_path = data_folder / load_name_weekly
    load_input = data_folder / load_name

    with h5py.File(load_path, "r") as f:
        weekly_load_profiles = f["X"][:]
        ids_load_profiles = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                             for x in f["ids"][:]]

    load_input_profiles = pd.read_hdf(load_input)
    load_input_profiles = load_input_profiles.drop(
        columns=['Electricity Tariff (Euros per kWh)', 'Electricity Tariff (Euros per Year)',
                 'Electricity Tariff (Name)'])

    results = compare_consumption_totals(
        load_input_profiles=load_input_profiles,  # your raw DF with Time/Household/Consumption (kWh)
        weekly_load_profiles=weekly_load_profiles,  # shape (n_weeks, 7, 96)
        ids_load_profiles=ids_load_profiles,  # list[str], same length as n_weeks
        freq="15min",
        week_freq="W-SUN",  # Monday-start weeks
        atol_kwh=1e-3  # adjust tolerance if you expect tiny float noise
    )
    pass