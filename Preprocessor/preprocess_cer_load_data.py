import numpy as np
import pandas as pd
from pathlib import Path

def decode_timecode(
        time_col: pd.Series,
        base_date: str = "2009-01-01",
        tz: str = None
) -> pd.Series:
    """
    Convert integer time codes DDDTT (DDD=day index, TT=half-hour slot) to pandas datetime.
      - Day code: 1 -> base_date, 2 -> base_date + 1 day, ...
      - Time code: 1..48, where 1 = 00:00–00:29:59 (start at 00:00), 2 = 00:30, ..., 48 = 23:30

    Parameters
    ----------
    time_col : pandas Series / numpy array / list / int
        Integer codes (e.g., 19503 means day=195, slot=3).
    base_date : str
        Date string for day code 1 (default "2009-01-01").
    tz : Optional[str]
        If provided, localize timestamps to this timezone.

    Returns
    -------
    pd.Series (datetime64[ns] (timezone-aware if tz given))
        Start timestamps of the 30-minute intervals.
    """
    s = pd.Series(time_col, copy=False)

    # to numeric, keep NaNs as NaT later
    vals = pd.to_numeric(s, errors="coerce").astype(int)

    # Extract day and slot (DDDTT => day = // 100, slot = % 100)
    day_code = (vals // 100).astype(int)
    slot_code = (vals % 100).astype(int)

    # Validate (optional—keeps invalid as NaT)
    valid = day_code.ge(1) & slot_code.between(1, 48)

    # Compute start timestamps
    base = pd.Timestamp(base_date)
    # days offset: (day_code - 1) days
    day_delta = (day_code - 1).astype(float)  # to allow NaN masking
    # minutes offset: (slot_code - 1) * 30 minutes
    min_delta = (slot_code - 1).astype(float) * 30.0

    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    dt.loc[valid] = (
        base
        + pd.to_timedelta(day_delta[valid], unit="D")
        + pd.to_timedelta(min_delta[valid], unit="m")
    )

    if tz is not None:
        # dt is a tz-naive datetime Series
        dt_utc = dt.dt.tz_localize("Europe/Vienna",
                                   ambiguous="infer",  # auto-pick first/second 02:xx
                                   nonexistent="shift_forward"  # for spring jumps
                                   ).dt.tz_convert("UTC")
    return pd.to_datetime(dt)

def missing_counts_per_day(
    df: pd.DataFrame,
    household_col: str = "Household",
    time_col: str = "Time",
    value_col: str = "Consumption",
    slot_freq: str = "30min",
) -> pd.DataFrame:
    """
    For each household and each calendar day, count missing 30-min measurements.
    Assumes a complete day should have 48 slots (30 min).

    Returns a DataFrame with:
      [household_col, 'Day', 'present_slots', 'missing_slots', 'expected_slots']
    """
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not in DataFrame")
    if household_col not in df.columns:
        raise KeyError(f"Column '{household_col}' not in DataFrame")
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not in DataFrame")

    d = df[[household_col, time_col, value_col]].copy()
    # ensure datetime
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])  # drop rows with invalid time

    # normalize to 30-min "slots" and to calendar day
    d["slot_time"] = d[time_col].dt.floor(slot_freq)
    d["Day"] = d["slot_time"].dt.normalize()  # midnight of that day

    # keep only slots that actually have a value (non-null)
    d_valid = d[d[value_col].notna()]

    # count unique slots per (Household, Day)
    present = (
        d_valid.groupby([household_col, "Day"])["slot_time"]
        .nunique()
        .rename("present_slots")
        .astype(int)
    )

    # expected slots per day
    expected_slots = int(pd.Timedelta("1D") / pd.Timedelta(slot_freq))  # 48 for 30min

    out = present.reset_index()
    out["expected_slots"] = expected_slots
    # clamp to [0, expected] then compute missing
    out["present_slots"] = out["present_slots"].clip(lower=0, upper=expected_slots)
    out["missing_slots"] = out["expected_slots"] - out["present_slots"]
    return out.sort_values([household_col, "Day"]).reset_index(drop=True)

def keep_only_complete_days(
    df: pd.DataFrame,
    *,
    household_col: str = "Household",
    time_col: str = "Time",
    value_col: str = "Consumption",
    slot_freq: str = "30min",
) -> pd.DataFrame:
    """
    Return a copy of `df` containing only (Household, Day) groups
    that have exactly 48 unique half-hour slots with non-null values.
    """
    d = df[[household_col, time_col, value_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])  # invalid times out

    # Normalize to slot grid and day
    d["slot_time"] = d[time_col].dt.floor(slot_freq)
    d["Day"] = d["slot_time"].dt.normalize()  # midnight

    # Count unique slots with a value per Household-Day
    present = (
        d[d[value_col].notna()]
        .groupby([household_col, "Day"])["slot_time"]
        .nunique()
        .rename("present_slots")
        .reset_index()
    )

    # Keep only complete days (48 slots)
    COMPLETE = int(pd.Timedelta("1D") / pd.Timedelta(slot_freq))  # 48
    complete_days = present.loc[present["present_slots"] == COMPLETE, [household_col, "Day"]]

    # Filter original rows to those Household-Day keys
    out = d.merge(complete_days, on=[household_col, "Day"], how="inner")

    # Drop helper cols and return in original column order
    out = out.drop(columns=["slot_time", "Day"])
    # Reorder to original order if needed
    cols = [household_col, time_col, value_col] + [c for c in out.columns if c not in (household_col, time_col, value_col)]
    return out[cols].sort_values([household_col, time_col]).reset_index(drop=True)

def process_file(path: Path) -> pd.DataFrame:
    # Read as strings (fast & tolerant), then vectorized coercion
    df = pd.read_csv(
        path,
        header=None,
        names=["Household", "Time", "Consumption"],
        usecols=[0, 1, 2],
        na_values=["", "NA", "NaN", "nan"],
        keep_default_na=True,
        engine="c",
        sep=r"\s+",
        dtype=str,
    )

    # Coerce dtypes
    df["Household"]   = pd.to_numeric(df["Household"], errors="coerce").astype(int)
    df["Consumption"] = pd.to_numeric(df["Consumption"], errors="coerce").astype(float)
    df["Time"]        = pd.to_numeric(df["Time"], errors="coerce").astype(int)

    # Decode timecode to datetime (returns start timestamp of 30-min interval)
    df["Time"] = decode_timecode(df["Time"], tz=None)

    # Keep only complete days (48 half-hour slots per household-day)
    df_complete = keep_only_complete_days(df, household_col="Household", time_col="Time", value_col="Consumption")

    print(f"[{path.name}] raw: {len(df):,}  →  complete: {len(df_complete):,}  (dropped {len(df)-len(df_complete):,})")
    return df_complete

# -----------------------
# Main
# -----------------------
def main():
    # Base folder for resolving relative paths
    data_folder = Path(__file__).resolve().parent / "data"

    data_files = [data_folder / f for f in
                  ['File1.txt', 'File2.txt', 'File3.txt', 'File4.txt', 'File5.txt', 'File6.txt']]

    all_load_data = pd.concat((process_file(p) for p in data_files), ignore_index=True)
    print(f"✅ Final concatenated shape: {all_load_data.shape}")

    output_path = data_folder / "cer_load_data_2011.h5"
    all_load_data.to_hdf(output_path, key="load_data")
    print(f"💾 Saved combined data to: {output_path}")

if __name__ == "__main__":
    main()