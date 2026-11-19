import numpy as np
import pandas as pd
from pathlib import Path

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

def normalize_raw_socio_categorical_data(df:pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'number_of_residents_original',
        'number_of_residents_standardized',
        'housing_type_standardized',
        'floor_area_standardized',
        'merge_id',
        'job_type',
        'building_age_structure',
        "work_day_at_home_8_to_12",
        "work_day_at_home_12_to_17",
        "work_day_at_home_17_to_8",
        "weekend_at_home_8_to_17",
        "weekend_at_home_17_to_8",
    ]

    df['numberofresidents'] = ((df["number_of_residents_standardized"].astype(str).
                                str.strip().str.replace(",", ".", regex=False))
                               .pipe(pd.to_numeric, errors="coerce")).astype(int)

    df['number_of_meters'] = pd.to_numeric(df['number_of_meters'], errors='coerce').fillna(-1).astype(int)
    df['square_meter'] = ((df["square_meter"].astype(str).
                           str.strip().str.replace(",", ".", regex=False))
                          .pipe(pd.to_numeric, errors="coerce")).astype(int)

    df['aquarium'] = pd.to_numeric(df["aquarium"], errors='coerce').fillna(-1).astype(int)
    df['gender'] = pd.to_numeric(df["gender"], errors='coerce').fillna(-1).astype(int)
    df['home_cinema'] = pd.to_numeric(df["home_cinema"], errors='coerce').fillna(-1).astype(int)
    df['heating_gas_biomass'] = pd.to_numeric(df["heating_gas_biomass"], errors='coerce').fillna(-1).astype(int)
    df['heating_gas_natural'] = pd.to_numeric(df["heating_gas_natural"], errors='coerce').fillna(-1).astype(int)
    df['heating_fern_heat'] = pd.to_numeric(df["heating_fern_heat"], errors='coerce').fillna(-1).astype(int)
    df['heating_liquid'] = pd.to_numeric(df["heating_liquid"], errors='coerce').fillna(-1).astype(int)
    df['heating_wood'] = pd.to_numeric(df["heating_wood"], errors='coerce').fillna(-1).astype(int)
    df['heating_coal'] = pd.to_numeric(df["heating_coal"], errors='coerce').fillna(-1).astype(int)
    df['heating_oil'] = pd.to_numeric(df["heating_oil"], errors='coerce').fillna(-1).astype(int)
    df['heating_pellets'] = pd.to_numeric(df["heating_pellets"], errors='coerce').fillna(-1).astype(int)
    df['heating_solar'] = pd.to_numeric(df["heating_solar"], errors='coerce').fillna(-1).astype(int)
    df['heating_electricity'] = pd.to_numeric(df["heating_electricity"], errors='coerce').fillna(-1).astype(int)
    df['heating_heat_pump'] = pd.to_numeric(df["heating_heat_pump"], errors='coerce').fillna(-1).astype(int)
    df['small_town'] = pd.to_numeric(df["small_town"], errors='coerce').fillna(-1).astype(int)
    df['postal_code'] = pd.to_numeric(df["postal_code"], errors='coerce').fillna(-1).astype(int)
    df['sauna'] = pd.to_numeric(df["sauna"], errors='coerce').fillna(-1).astype(int)
    df['solarium'] = pd.to_numeric(df["solarium"], errors='coerce').fillna(-1).astype(int)
    df['city'] = pd.to_numeric(df["city"], errors='coerce').fillna(-1).astype(int)
    df['swimming_pool'] = pd.to_numeric(df["swimming_pool"], errors='coerce').fillna(-1).astype(int)
    df['water_bed'] = pd.to_numeric(df["water_bed"], errors='coerce').fillna(-1).astype(int)
    df['housing_type'] = pd.to_numeric(df["housing_type"], errors='coerce').fillna(-1).astype(int)
    df['housing_building'] = pd.to_numeric(df["housing_building"], errors='coerce').fillna(-1).astype(int)
    df['housing_apartment'] = pd.to_numeric(df["housing_apartment"], errors='coerce').fillna(-1).astype(int)
    df['water_biomass'] = pd.to_numeric(df["water_biomass"], errors='coerce').fillna(-1).astype(int)
    df['water_natural_gas'] = pd.to_numeric(df["water_natural_gas"], errors='coerce').fillna(-1).astype(int)
    df['water_fern_heat'] = pd.to_numeric(df["water_fern_heat"], errors='coerce').fillna(-1).astype(int)
    df['water_liquid_gas'] = pd.to_numeric(df["water_liquid_gas"], errors='coerce').fillna(-1).astype(int)
    df['water_wood'] = pd.to_numeric(df["water_wood"], errors='coerce').fillna(-1).astype(int)
    df['water_coal'] = pd.to_numeric(df["water_coal"], errors='coerce').fillna(-1).astype(int)
    df['water_oil'] = pd.to_numeric(df["water_oil"], errors='coerce').fillna(-1).astype(int)
    df['water_pellets'] = pd.to_numeric(df["water_pellets"], errors='coerce').fillna(-1).astype(int)
    df['water_solar'] = pd.to_numeric(df["water_solar"], errors='coerce').fillna(-1).astype(int)
    df['water_electricity'] = pd.to_numeric(df["water_electricity"], errors='coerce').fillna(-1).astype(int)
    df['water_heat_pump'] = pd.to_numeric(df["water_heat_pump"], errors='coerce').fillna(-1).astype(int)
    df['yearly_electricity_consumption'] = ((df["yearly_electricity_consumption"].astype(str).
                                            str.strip().str.replace(",", ".", regex=False))
                                            .pipe(pd.to_numeric, errors="coerce")).astype(float)

    df['sex'] = pd.to_numeric(df["sex"], errors='coerce').fillna(-1).astype(int)
    df['education_level'] = df['education_level'].astype(str)
    df['job_type'] = df['job_type'].astype(str)
    df['works_from_home'] = pd.to_numeric(df["works_from_home"], errors='coerce').fillna(-1).astype(int)
    df['medical_equipment'] = pd.to_numeric(df["medical_equipment"], errors='coerce').fillna(-1).astype(int)
    df['work_day_at_home_8_to_12'] = pd.to_numeric(df["work_day_at_home_8_to_12"], errors='coerce').fillna(-1).astype(int)
    df['work_day_at_home_12_to_17'] = pd.to_numeric(df["work_day_at_home_12_to_17"], errors='coerce').fillna(-1).astype(int)
    df['work_day_at_home_17_to_8'] = pd.to_numeric(df["work_day_at_home_17_to_8"], errors='coerce').fillna(-1).astype(int)
    df['weekend_at_home_8_to_17'] = pd.to_numeric(df["weekend_at_home_8_to_17"], errors='coerce').fillna(-1).astype(int)
    df['weekend_at_home_17_to_8'] = pd.to_numeric(df["weekend_at_home_17_to_8"], errors='coerce').fillna(-1).astype(int)

    df['single'] = df["single"].astype(str)
    df['building_age_structure'] = df['building_age_structure'].astype(str)
    df['income_category'] = df['income_category'].astype(str)
    df['age'] = df['age'].astype(str)
    df["age"] = df["age"].apply(
        lambda x: -1 if pd.isna(x) or str(x).strip() == "" else
        0 if str(x).strip().lower() in ("a", "b", "c") else
        1 if str(x).strip().lower() in ("d", "e", "f", "g") else
        2 if str(x).strip().lower() in ("h", "i") else
        -1
    ).astype(int)

    df["income_category"] = df["income_category"].apply(
        lambda x: -1 if pd.isna(x) or str(x).strip() == "" else
        1 if str(x).strip().upper() in ("X", "L") else
        0 if str(x).strip().upper() in ("O", "V", "F", "R", "T", "K", "A", "M", "Z", "P") else
        -1
    ).astype(int)

    df["retired"] = df["job_type"].apply(
        lambda x: -1 if pd.isna(x) or str(x).strip() == "" else
        1 if str(x).strip().lower() == "k" else
        0 if ("a" <= str(x).strip().lower() <= "j") or ("l" <= str(x).strip().lower() <= "m") else
        -1
    ).astype(int)

    df["employed"] = df["job_type"].apply(
        lambda x: -1 if pd.isna(x) or str(x).strip() == "" else
        1 if "a" <= str(x).strip().lower() <= "j" else
        0 if "k" <= str(x).strip().lower() <= "m" else
        -1
    ).astype(int)

    df["education_level"] = df["education_level"].apply(
        lambda x: -1 if pd.isna(x) or str(x).strip() == "" else
        0 if (("a" <= str(x).strip().lower() <= "b") or str(x).strip().lower() == "e") else
        1 if ("c" <= str(x).strip().lower() <= "d") else
        -1
    ).astype(int)

    df["single"] = df["single"].apply(
        lambda x: -1 if pd.isna(x) or str(x).strip() == "" else
        1 if str(x).strip().lower() == "a" else
        0 if "b" <= str(x).strip().lower() <= "i" else
        -1
    ).astype(int)

    df["house_age"] = df["building_age_structure"].apply(
        lambda x: -1 if pd.isna(x) or str(x).strip() == "" else
        1 if "d" <= str(x).strip().lower() <= "h" else
        0 if "a" <= str(x).strip().lower() <= "c" else
        -1
    ).astype(int)

    df["split_house"] = df["housing_type"].apply(
        lambda x: 1 if x == 3 else 0 if x in (1, 2) else -1
    ).astype(int)

    unoccupied_cols = [
        "work_day_at_home_8_to_12",
        "work_day_at_home_12_to_17",
        "work_day_at_home_17_to_8",
        "weekend_at_home_8_to_17",
        "weekend_at_home_17_to_8",
    ]

    any_minus1 = df[unoccupied_cols].eq(-1).any(axis=1)
    any_zero = df[unoccupied_cols].eq(0).any(axis=1)

    df["unoccupied"] = np.select(
        [any_minus1, any_zero],
        [-1, 1],
        default=0
    ).astype(int)

    df['family_home'] = df['numberofresidents'].apply(lambda x: 0 if x <= 2 else 1).astype(int)
    df['floor_area'] = df['square_meter'].apply(
        lambda x: -1 if x == -1 else (0 if x < 100 else 1 if x < 200 else 2)).astype(int)

    appliance_cols = [
        "aquarium", "home_cinema", "sauna", "solarium", "swimming_pool", "water_bed",
        "water_electricity", "water_heat_pump", "heating_electricity", "heating_heat_pump"]

    df["appliances_total"] = df[appliance_cols].clip(lower=0).sum(axis=1).astype(int)
    df["appliances_total"] = df["appliances_total"].apply(
        lambda x: 0 if x <= 1 else 1 if x < 3 else 2).astype(int)

    df = df.drop(columns=columns_to_drop)
    return df

def set_household_ids(df:pd.DataFrame, household_id_col: str = "household_id",) -> pd.DataFrame:
    df = df[~df['merge_id'].isna()]
    df['serial_number_basis_1'] = pd.to_numeric(df['serial_number_basis_1'], errors='coerce').fillna(-1).astype(int)
    df['serial_number_basis_2'] = pd.to_numeric(df['serial_number_basis_2'], errors='coerce').fillna(-1).astype(int)
    df['serial_addon_1'] = pd.to_numeric(df['serial_addon_1'], errors='coerce').fillna(-1).astype(int)
    df['serial_addon_2'] = pd.to_numeric(df['serial_addon_2'], errors='coerce').fillna(-1).astype(int)

    cols = [ "serial_number_basis_1",
            "serial_number_basis_2",
            "serial_addon_1",
            "serial_addon_2"]

    # all other info columns to carry along
    other_cols = [c for c in df.columns if c not in cols]

    # Long format: one row per (original row, serial column)
    long = (
        df[cols + other_cols]
        .melt(
            id_vars=other_cols,
            value_vars=cols,
            var_name="serial_source",
            value_name=household_id_col,
        )
    )
    # keep only valid ids
    long[household_id_col] = pd.to_numeric(long[household_id_col], errors="coerce")
    long = long[(long[household_id_col].notna()) & (long[household_id_col] != -1)].copy()

    # Optional: make household_id int (nullable int if needed)
    long[household_id_col] = long[household_id_col].astype(int)

    # If a household_id appears multiple times (across rows/serial columns),
    # keep one record per household_id
    out = (
        long.sort_values([household_id_col, "serial_source"])
        .drop_duplicates(subset=[household_id_col], keep="first")
        .drop(columns=["serial_source"])
        .reset_index(drop=True)
    )

    cols = [household_id_col] + [c for c in out.columns if c != household_id_col]
    out = out[cols]

    return out

def normalize_socio_categorical_2018_data(df:pd.DataFrame) -> pd.DataFrame:
    """ Normalize socio-demographic categorical data values for 2018 data set."""
    df['family_home'] = df['numberofresidents'].apply(lambda x: 0 if x <= 2 else 1).astype(int)
    df['floor_area'] = df['square_meter'].apply(
        lambda x: -1 if x == -1 else (0 if x < 100 else 1 if x < 200 else 2)).astype(int)

    appliance_cols = [
        "dryer", "swimming_pool", "aquarium", "water_bed", "sauna", "air_condition",
        "deepFreezers", "computers", "pev", "emotorbike", "ebike"]

    df["appliances_total"] = df[appliance_cols].clip(lower=0).sum(axis=1).astype(int)
    df["appliances_total"] = df["appliances_total"].apply(
        lambda x: 0 if x <= 4 else 1 if x < 7 else 2).astype(int)
    return df

def normalize_socio_categorical_2011_data(df:pd.DataFrame) -> pd.DataFrame:
    """ Normalize socio-demographic categorical data values for 2011 data set."""
    df['family_home'] = df['numberofresidents'].apply(lambda x: 0 if x <= 2 else 1).astype(int)
    df['floor_area'] = df['square_meter'].apply(
        lambda x: -1 if x == -1 else (0 if x < 100 else 1 if x < 200 else 2)).astype(int)

    appliance_cols = [
        "aquarium", "home_cinema", "sauna", "solarium", "swimming_pool", "water_bed",
        "electric_water", "heat_pump_water", "electric_heating", "heat_pump_heating"]

    df["appliances_total"] = df[appliance_cols].clip(lower=0).sum(axis=1).astype(int)
    df["appliances_total"] = df["appliances_total"].apply(
        lambda x: 0 if x <= 1 else 1 if x < 3 else 2).astype(int)
    return df

def main():
    # Base folder for resolving relative paths
    data_folder = Path(__file__).resolve().parent.parent / "data"

    # Specify input data paths
    input_categorical = '00_raw_data/Datenverschraenkung_old_Energy_institute_EMot.csv'
    # input_categorical = '00_raw_data/input_categorical_data_2011_original.h5'
    path_categorical = data_folder / input_categorical

    if input_categorical == '00_raw_data/Datenverschraenkung_old_Energy_institute_EMot.csv':
        print(f" You started the script in the raw mode. So you are reading the raw csv file and processing it from scratch.")
        target_columns = {
            'MergeID': 'merge_id',
            'AnzahlPHH_imputiert': 'number_of_residents_standardized',
            'AnzPHH_original': 'number_of_residents_original',
            'AnzZaehler': 'number_of_meters',
            'Aquarium': 'aquarium',
            'Duflaeche': 'floor_area_standardized',
            'DUSwimmingPool': 'swimming_pool_standardized',
            'Duwohnform': 'housing_type_standardized',
            'EinzDat': 'individual_date',
            'Flaeche': 'square_meter',
            'Gender': 'gender',
            'Heimkino': 'home_cinema',
            'HGBiomasse': 'heating_gas_biomass',
            'HGErdgas': 'heating_gas_natural',
            'HGFernwaerme': 'heating_fern_heat',
            'HGFluessiggas': 'heating_liquid',
            'HGHolz': 'heating_wood',
            'HGKohle': 'heating_coal',
            'HGOel': 'heating_oil',
            'HGPellets': 'heating_pellets',
            'HGSolar': 'heating_solar',
            'HGStrom': 'heating_electricity',
            'HGWaermepumpe': 'heating_heat_pump',
            'kleingemeinde': 'small_town',
            'PLZ': 'postal_code',
            'Sauna': 'sauna',
            'Solarium': 'solarium',
            'stadt': 'city',
            'Swimmingpool': 'swimming_pool',
            'Wasserbett': 'water_bed',
            'Wohnform': 'housing_type',
            'wohnhaus': 'housing_building',
            'wohnort': 'residence_location',
            'wohnung': 'housing_apartment',
            'WWBiomasse': 'water_biomass',
            'WWBrennsoffe': 'water_fuels',
            'WWErdgas': 'water_natural_gas',
            'WWFernwaerme': 'water_fern_heat',
            'WWFluessiggas': 'water_liquid_gas',
            'WWHolz': 'water_wood',
            'WWKohle': 'water_coal',
            'WWOel': 'water_oil',
            'WWPellets': 'water_pellets',
            'WWSolar': 'water_solar',
            'WWStrom': 'water_electricity',
            'WWWaermepumpe': 'water_heat_pump',
            'YearlyElectricityDemand': 'yearly_electricity_consumption',
            'Sex': 'sex',
            'f11Education': 'education_level',
            'f12Job': 'job_type',
            'f13workathome': 'works_from_home',
            'fr14NrChildren': 'number_of_children',
            'f16Med': 'medical_equipment',
            'f17a': 'work_day_at_home_8_to_12',
            'f17b': 'work_day_at_home_12_to_17',
            'f17c': 'work_day_at_home_17_to_8',
            'f17d': 'weekend_at_home_8_to_17',
            'f17e': 'weekend_at_home_17_to_8',
            'f18': 'single',
            'f20AgeStructure': 'building_age_structure',
            'f26IncomeCategory': 'income_category',
            'f27Age': 'age',
            'SerialnummerBasis1': 'serial_number_basis_1',
            'SerialnummerBasis2': 'serial_number_basis_2',
            'SerialnummerZusatz1': 'serial_addon_1',
            'SerialnummerZusatz2': 'serial_addon_2',
        }
        df_categorical = pd.read_csv(path_categorical, sep=';')
        df_categorical = select_and_rename_question_columns(df_categorical, target_columns)
        df_categorical = set_household_ids(df_categorical)
        normalized_socio = normalize_raw_socio_categorical_data(df_categorical)
        output_path = data_folder / "socio_data_2011.h5"
    else:
        print(
            f" You started the script in the preprocessed mode. So you are reading the preprocessed hdf5 file and normalizing it.")
        df_categorical = pd.read_hdf(path_categorical)
        year = input_categorical.split("_")[-2].split(".")[0]

        if year == '2011':
            normalized_socio = normalize_socio_categorical_2011_data(df_categorical)
            output_path = data_folder / "socio_data_2011.h5"
        elif year == '2018':
            normalized_socio = normalize_socio_categorical_2018_data(df_categorical)
            output_path = data_folder / "socio_data_2018.h5"
        else:
            print(f"⚠️ Warning: The input categorical data is from {year}, no normalization function found for this year.")
            normalized_socio = None
            output_path = None

    print(f"✅ Finished with data normalization, normalized data shape: {normalized_socio.shape}")

    normalized_socio.to_hdf(output_path, key="socio_data")
    print(f"💾 Saved preprocessed data to: {output_path}")

if __name__ == "__main__":
    main()
