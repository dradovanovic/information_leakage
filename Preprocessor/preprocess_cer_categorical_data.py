import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil, sqrt
import matplotlib.pyplot as plt

def _clean_columns(cols) -> list[str]:
    cleaned = []
    for c in cols:
        s = str(c)

        # normalize unicode (e.g., fancy quotes → normal forms)
        s = unicodedata.normalize("NFKC", s)

        # common mojibake fixes
        s = s.replace("�", "'")          # replacement char → apostrophe (best-effort)
        s = s.replace("’", "'").replace("‘", "'")
        s = s.replace("“", '"').replace("”", '"')
        s = s.replace("\xa0", " ")       # non-breaking space

        # collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()

        cleaned.append(s)
    return cleaned

def read_survey_csv(path: Path) -> pd.DataFrame:
    # Common fallback encodings for survey CSVs
    for enc in ("utf-8-sig", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(
                path,
                encoding=enc,                 # tolerant encodings
                encoding_errors="replace",    # replace bad bytes instead of failing
                engine="python",              # more robust with weird quoting
                sep=",",                      # comma-delimited
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",          # skip malformed rows
            )
            df.columns = _clean_columns(df.columns)
            print(f"Loaded with encoding={enc}, rows={len(df)}")
            return df
        except UnicodeDecodeError:
            continue
    # Final fallback: read as binary-compatible bytes to inspect or salvage
    raise UnicodeDecodeError("Could not decode CSV with utf-8-sig/cp1252/iso-8859-1")


def select_and_rename_question_columns(df: pd.DataFrame, target_columns: dict) -> pd.DataFrame:
    selected_original_cols = []
    rename_map = {}

    for code, new_name in target_columns.items():
        code = str(code)
        # find all columns whose name contains the code
        matches = [col for col in df.columns if code in str(col)]
        if not matches:
            # nothing found for this code; skip or raise if you prefer
            # print(f"Warning: no column found containing '{code}'")
            continue

        # take the first match deterministically
        original_col = matches[0]
        selected_original_cols.append(original_col)
        rename_map[original_col] = new_name

    # subset and rename
    result = df[selected_original_cols].copy().rename(columns=rename_map)
    return result

def normalize_socio_data(df: pd.DataFrame) -> pd.DataFrame:
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    # Missing answers must stay -1 so that they are filtered out later (NaN == x is always False)
    df['gender'] = df['gender'].apply(lambda x: -1 if pd.isna(x) else 0 if x == 1 else 1)

    df["age_class"] = df["age_class"].apply(lambda x: 0 if x in (1, 2) else 1 if x in (3, 4, 5) else 2 if x == 6 else -1)

    df['employment_category'] = pd.to_numeric(df['employment_category'], errors='coerce')
    df['retired'] = df['employment_category'].apply(lambda x: -1 if pd.isna(x) else 1 if x == 6 else 0)
    df['employment_status'] = df["employment_category"].apply(lambda x: -1 if pd.isna(x) else 1 if x in (1, 2, 3) else 0)

    df['social_class'] = pd.to_numeric(df['social_class'], errors='coerce')
    df['social_class'] = df["social_class"].apply(lambda x: 0 if x == 1 else 1 if x in (2, 3) else 2 if x == 4 else -1)

    df['income'] = pd.to_numeric(df['income'], errors='coerce')
    df['income'] = df['income'].fillna(-1).astype(int)
    df["income"] = df["income"].apply(lambda x: 0 if x in (1, 2, 3) else 1 if x in (4, 5) else -1)

    df['family_status'] = pd.to_numeric(df['family_status'], errors='coerce')
    df['single'] = df['family_status'].apply(lambda x: -1 if pd.isna(x) else 1 if x == 1 else 0)

    # number of residents
    df["num_residents"] = np.nan
    df['residents_over_15'] = pd.to_numeric(df['residents_over_15'], errors='coerce')
    df['residents_over_15'] = df['residents_over_15'].fillna(0).astype(int)
    df['residents_under_15'] = pd.to_numeric(df['residents_under_15'], errors='coerce')
    df['residents_under_15'] = df['residents_under_15'].fillna(0).astype(int)

    # Q410 == 1 → lives alone
    df.loc[df["family_status"] == 1, "num_residents"] = 1

    # Q410 == 2 → only over 15
    mask_2 = df["family_status"] == 2
    df.loc[mask_2, "num_residents"] = df.loc[mask_2, "residents_over_15"].astype(int)

    # Q410 == 3 → over 15 + under 15
    mask_3 = df["family_status"] == 3
    df.loc[mask_3, "num_residents"] = (df.loc[mask_3, "residents_over_15"].astype(int) +
                                       df.loc[mask_3, "residents_under_15"].astype(int))

    df['num_residents'] = df['num_residents'].fillna(-1).astype(int)
    df["family_home"] = df["num_residents"].apply(lambda x: -1 if x < 1 else 0 if x <= 2 else 1)

    df['house_age'] = pd.to_numeric(df['house_age'], errors='coerce')
    df['house_age'] = df["house_age"].apply(lambda x: -1 if pd.isna(x) or x == 9999 else 0 if x <= 1980 else 1)

    df['home_owned'] = pd.to_numeric(df['home_owned'], errors='coerce')
    df['home_owned'] = df['home_owned'].fillna(-1).astype(int)
    df['home_owned'] = df['home_owned'].apply(lambda x: 0 if x in (1, 2) else 1 if x in (3, 4) else -1)

    df['floor_area'] = pd.to_numeric(df['floor_area'], errors='coerce')
    df['floor_area_unit'] = pd.to_numeric(df['floor_area_unit'], errors='coerce')
    df['floor_area_unit'] = df['floor_area_unit'].fillna(-1).astype(int)

    FT2_TO_M2 = 0.09290304

    # Create new column in square meters
    df["floor_area_m2"] = np.nan  # default for invalid units

    # unit == 1: already m²
    m2_mask = df["floor_area_unit"] == 1
    df.loc[m2_mask, "floor_area_m2"] = df.loc[m2_mask, "floor_area"]

    # unit == 2: ft² -> m²
    ft2_mask = df["floor_area_unit"] == 2
    df.loc[ft2_mask, "floor_area_m2"] = np.round(df.loc[ft2_mask, "floor_area"] * FT2_TO_M2)
    df['floor_area_m2'] = df['floor_area_m2'].fillna(-1).astype(int)

    df['floor_area'] = df["floor_area_m2"].apply(
        lambda x: -1 if x == -1 else (0 if x < 100 else 1 if x < 200 else 2)).astype(int)

    df['house_type'] = pd.to_numeric(df['house_type'], errors='coerce')
    df['apartment'] = df['house_type'].fillna(-1).astype(int)
    df['apartment'] = df['apartment'].apply(lambda x: -1 if x == -1 else 1 if x == 1 else 0)
    df['split_house'] = df['house_type'].apply(lambda x: -1 if pd.isna(x) else 1 if x == 2 else 0)

    df['electric_cooking'] = pd.to_numeric(df['electric_cooking'], errors='coerce')
    df['electric_cooking'] = df['electric_cooking'].apply(lambda x: -1 if pd.isna(x) else 1 if x == 1 else 0)

    df['unoccupied_under_15'] = pd.to_numeric(df['unoccupied_under_15'], errors='coerce')
    df['unoccupied_under_15'] = df['unoccupied_under_15'].fillna(-1).astype(int)
    df['unoccupied_over_15'] = pd.to_numeric(df['unoccupied_over_15'], errors='coerce')
    df['unoccupied_over_15'] = df['unoccupied_over_15'].fillna(-1).astype(int)

    # 1) Map each column: 8 -> 1, 1..7 -> 0 (and keep other/invalid like -1 as 0 as well)
    df["unoccupied_under_15"] = df["unoccupied_under_15"].apply(lambda x: 1 if x == 8 else 0).astype(int)
    df["unoccupied_over_15"] = df["unoccupied_over_15"].apply(lambda x: 1 if x == 8 else 0).astype(int)

    # 2) Combined flag: if either is 1 -> 1 else 0
    df["unoccupied"] = ((df["unoccupied_under_15"] == 1) | (df["unoccupied_over_15"] == 1)).astype(int)

    appliance_map = {1: 0, 2: 1, 3: 2, 4: 3}
    df['freezer'] = pd.to_numeric(df['freezer'], errors='coerce')
    df['freezer'] = (df['freezer'].map(appliance_map).fillna(-1).astype(int))
    df['washing_machine'] = pd.to_numeric(df['washing_machine'], errors='coerce')
    df['washing_machine'] = (df['washing_machine'].map(appliance_map).fillna(-1).astype(int))
    df['dryer'] = pd.to_numeric(df['dryer'], errors='coerce')
    df['dryer'] = (df['dryer'].map(appliance_map).fillna(-1).astype(int))
    df['dishwasher'] = pd.to_numeric(df['dishwasher'], errors='coerce')
    df['dishwasher'] = (df['dishwasher'].map(appliance_map).fillna(-1).astype(int))
    df['electric_shower'] = pd.to_numeric(df['electric_shower'], errors='coerce')
    df['electric_shower'] = (df['electric_shower'].map(appliance_map).fillna(-1).astype(int))
    df['electric_pump_shower'] = pd.to_numeric(df['electric_pump_shower'], errors='coerce')
    df['electric_pump_shower'] = (df['electric_pump_shower'].map(appliance_map).fillna(-1).astype(int))
    df['electric_cooker'] = pd.to_numeric(df['electric_cooker'], errors='coerce')
    df['electric_cooker'] = (df['electric_cooker'].map(appliance_map).fillna(-1).astype(int))
    df['electric_heater'] = pd.to_numeric(df['electric_heater'], errors='coerce')
    df['electric_heater'] = (df['electric_heater'].map(appliance_map).fillna(-1).astype(int))
    df['water_pump'] = pd.to_numeric(df['water_pump'], errors='coerce')
    df['water_pump'] = (df['water_pump'].map(appliance_map).fillna(-1).astype(int))
    df['immersion'] = pd.to_numeric(df['immersion'], errors='coerce')
    df['immersion'] = (df['immersion'].map(appliance_map).fillna(-1).astype(int))
    df['tv_small'] = pd.to_numeric(df['tv_small'], errors='coerce')
    df['tv_small'] = (df['tv_small'].map(appliance_map).fillna(-1).astype(int))
    df['tv_large'] = pd.to_numeric(df['tv_large'], errors='coerce')
    df['tv_large'] = (df['tv_large'].map(appliance_map).fillna(-1).astype(int))
    df['desktop_computer'] = pd.to_numeric(df['desktop_computer'], errors='coerce')
    df['desktop_computer'] = (df['desktop_computer'].map(appliance_map).fillna(-1).astype(int))
    df['laptop_computer'] = pd.to_numeric(df['laptop_computer'], errors='coerce')
    df['laptop_computer'] = (df['laptop_computer'].map(appliance_map).fillna(-1).astype(int))
    df['console'] = pd.to_numeric(df['console'], errors='coerce')
    df['console'] = (df['console'].map(appliance_map).fillna(-1).astype(int))
    appliance_cols = [
        "freezer", "washing_machine", "dryer", "dishwasher", "electric_shower", "electric_pump_shower",
        "electric_cooker", "electric_heater", "water_pump", "immersion", "tv_small", "tv_large","desktop_computer",
        "laptop_computer", "console"]

    df["appliances_total"] = df[appliance_cols].clip(lower=0).sum(axis=1).astype(int)

    df["appliances_total"] = df["appliances_total"].apply(
        lambda x: 0 if x <= 8 else 1 if x < 11 else 2).astype(int)
    return df

def plot_all_column_distributions(
    df: pd.DataFrame,
    max_categories: int = 20,
    figsize_per_col: tuple = (12, 8),
    title: str = "Column-wise Value Distributions",
):

    n_cols = df.shape[1]
    if n_cols == 0:
        raise ValueError("DataFrame has no columns.")

    # Arrange subplots roughly in a square grid
    n_rows = ceil(sqrt(n_cols))
    n_cols_grid = ceil(n_cols / n_rows)

    fig_width = figsize_per_col[0] * n_cols_grid
    fig_height = figsize_per_col[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(fig_width, fig_height))
    if n_rows == 1 and n_cols_grid == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols_grid == 1:
        axes = np.atleast_2d(axes)

    fig.suptitle(title, fontsize=20, y=0.98)

    col_names = df.columns.tolist()

    for idx, col in enumerate(col_names):
        r, c = divmod(idx, n_cols_grid)
        ax = axes[r, c]

        s = df[col]
        # value_counts including NaN as a category
        vc = s.value_counts(dropna=False)

        # group tail into "__OTHER__" if too many categories
        if len(vc) > max_categories:
            top = vc.iloc[:max_categories]
            other_sum = vc.iloc[max_categories:].sum()
            vc = top.append(pd.Series({ "__OTHER__": other_sum }))

        categories = [str(x) if not pd.isna(x) else "NaN" for x in vc.index]
        counts = vc.values

        x_idx = np.arange(len(categories))
        bars = ax.bar(x_idx, counts)

        ax.set_title(col, fontsize=16)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

        # Annotate each bar with its exact value
        for p, cnt in zip(bars, counts):
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height(),
                f"{int(cnt)}",
                ha="center",
                va="bottom",
                fontsize=16,
            )

    # Turn off any unused subplots
    total_axes = n_rows * n_cols_grid
    for j in range(len(col_names), total_axes):
        r, c = divmod(j, n_cols_grid)
        axes[r, c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# -----------------------
# Main
# -----------------------
def main():
    # Base folder for resolving relative paths
    data_folder = Path(__file__).resolve().parent.parent / "data"

    target_columns= {
        '200': 'gender',
        '300': 'age_class',
        '310': 'employment_category',
        '401': 'social_class',
        '402': 'income',
        '410': 'family_status',
        '420': 'residents_over_15',
        '430': 'unoccupied_over_15',
        '43111': 'residents_under_15',
        '4312': 'unoccupied_under_15',
        '450': 'house_type',
        '452': 'home_owned',
        '453': 'house_age',
        '4704': 'electric_cooking',
        '49002: Washing machine': 'washing_machine',
        '49002: Tumble dryer': 'dryer',
        '49002: Dishwasher': 'dishwasher',
        '49002: Electric shower (instant)': 'electric_shower',
        '49002: Electric shower (electric pumped from hot tank)': 'electric_pump_shower',
        '49002: Electric cooker': 'electric_cooker',
        '49002: Electric heater (plug-in convector heaters)': 'electric_heater',
        '49002: Stand alone freezer': 'freezer',
        '49002: A water pump or electric well pump or pressurised water system': 'water_pump',
        '49002: Immersion': 'immersion',
        '490002: TV\'s less than 21 inch': 'tv_small',
        '490002: TV\'s greater than 21 inch': 'tv_large',
        '490002: Desk-top computers': 'desktop_computer',
        '490002: Lap-top computers': 'laptop_computer',
        '490002: Games consoles, such as xbox, playstation or Wii': 'console',
        '6103': 'floor_area',
        '61031: Is that': 'floor_area_unit',
    }

    data_file = data_folder / "00_raw_data/Smart meters Residential pre-trial survey data.csv"
    cer_socio = read_survey_csv(data_file)
    filtered_socio = select_and_rename_question_columns(cer_socio, target_columns)
    filtered_socio.index = cer_socio['ID'].astype(int)
    normalized_socio = normalize_socio_data(filtered_socio)

    normalized_socio = normalized_socio.reset_index()

    print(f"✅ Finished with data normalization, normalized data shape: {normalized_socio.shape}")

    output_path = data_folder / "cer_socio_data_2011.h5"
    normalized_socio.to_hdf(output_path, key="socio_data")
    print(f"💾 Saved preprocessed data to: {output_path}")

if __name__ == "__main__":
    main()