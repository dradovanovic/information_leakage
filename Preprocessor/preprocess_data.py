import numpy as np
import pandas as pd
from typing import Dict, Iterable

class PreProcessor:
    def __init__(self, load_data_path, categorical_data_path):
        self.load_data_path = load_data_path
        self.categorical_data_path = categorical_data_path
        self.load_data = pd.DataFrame()
        self.categorical_data = pd.DataFrame()

    def read_data(self) -> pd.DataFrame:
        print(f" \n##################################### 1.Step - Reading Data ########################################## ")
        print(f" Started reading preprocessed load data from file: {self.load_data_path}")
        if not self.load_data_path.exists():
            return False, None, f"File not found: {self.load_data_path}"
        try:
            self.load_data = pd.read_hdf(self.load_data_path)
            print(f' \tShape of load data is: {self.load_data.shape}')
        except (FileNotFoundError, OSError, ValueError, KeyError) as e:
            return False, None, f"Failed to read HDF: {e!s}"

        print(f" Started reading preprocessed categorical data from file: {self.categorical_data_path}")
        if not self.categorical_data_path.exists():
            return False, None, f"File not found: {self.categorical_data_path}"
        try:
            self.categorical_data = pd.read_hdf(self.categorical_data_path)
            print(f' \tShape of categorical data is: {self.categorical_data.shape}')
        except (FileNotFoundError, OSError, ValueError, KeyError) as e:
            return False, None, f"Failed to read HDF: {e!s}"
        return self.load_data, self.categorical_data

    @staticmethod
    def _standardize_base(col: str) -> str:
        c = col.strip().lower()
        for ch in ["_", "-", "."]:
            c = c.replace(ch, " ")
        c = " ".join(c.split())
        return c

    @staticmethod
    def _titlecase_no_spaces(col: str) -> str:
        """TitleCase with no inner spaces (for demo attributes)."""
        parts = [p for p in col.replace("_", " ").split(" ") if p]
        return "_".join(w.capitalize() for w in parts)

    def unify_columns_load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        variants: Dict[str, str] = {
            # Household id
            "household": "Household",
            "household id": "Household",
            "householdid": "Household",
            "hhid": "Household",
            "hh": "Household",

            # Time
            "time": "Time",
            "timestamp": "Time",
            "date time": "Time",
            "datetime": "Time",
            "date": "Time",
            "interval start": "Time",
            "start time": "Time",

            # Consumption (kWh)
            "consumption": "Consumption",
            "consumption kwh": "Consumption",
            "consumption (kwh)": "Consumption",
            "energy": "Consumption",
            "energy kwh": "Consumption",
            "kwh": "Consumption",
        }

        mapping = {}
        for c in df.columns:
            base = self._standardize_base(c)
            if base in variants:
                mapping[c] = variants[base]

        out = df.rename(columns=mapping).copy()

        # Minimal required columns
        required = {"Time", "Consumption", "Household"}
        missing = required - set(out.columns)
        if missing:
            raise ValueError(f"Load data is missing required columns after normalization: {missing}")

        # Get only the columns that are required
        df_subset = out[[c for c in required if c in out.columns]]

        # Coerce types
        df_subset.loc[:, "Time"] = pd.to_datetime(df_subset["Time"], errors="coerce")
        df_subset.loc[:, "Consumption"] = pd.to_numeric(df_subset["Consumption"], errors="coerce")
        return df_subset

    def unify_columns_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize categorical columns:
          - Force 'Household' key (handles 'household', 'hhid', etc.)
          - Other attributes are TitleCased with underscores (e.g., 'swimming_pool' -> 'Swimming_Pool').
        """
        # First ensure Household column
        household_variants: Iterable[str] = [
            "household", "household id", "householdid", "hhid", "hh", "meter id", "meter", "customer id",
            "customer", "ID", "Id", "id" ]
        rename_map = {}
        found_household = None
        for c in df.columns:
            if self._standardize_base(c) in household_variants and found_household is None:
                rename_map[c] = "Household"
                found_household = c

        out = df.rename(columns=rename_map).copy()
        if "Household" not in out.columns:
            raise ValueError("Demo data must contain a Household column (any of: household, hhid, meter_id, ...).")

        # Optional: pin some friendly names for common socio-demographic fields
        pin_variants: Dict[str, str] = {
            "apartment": "Apartment",
            "apartment type": "Apartment",
            "flat": "Apartment",
            "dwelling type": "Apartment",
            "housing_apartment": "Apartment",
            "housing apartment": "Apartment",
            "Housing_Apartment": "Apartment",
            "sauna": "Sauna",
            "has sauna": "Sauna",
            "sauna bool": "Sauna",
            "sauna present": "Sauna",
            "Age": "Age_Person",
            "age": "Age_Person",
            "age person": "Age_Person",
            "person age": "Age_Person",
            "Person age": "Age_Person",
            "Age_Class": "Age_Person",
            "Age_class": "Age_Person",
            "age class": "Age_Person",
            "House_age": "Age_House",
            "house_age": "Age_House",
            "House Age": "Age_House",
            "house Age": "Age_House",
            "age house": "Age_House",
            "Age House": "Age_House",
            "house age": "Age_House",
            "building age": "Age_House",
            "Employment_status": "Employment_Status",
            "employment status": "Employment_Status",
            "employed": "Employment_Status",
            "income": "Income_Group",
            "Income_level": "Income_Group",
            "income level": "Income_Group",
            "income_category": "Income_Group",
            "income category": "Income_Group",
            "Income Category": "Income_Group",
            "Income_Category": "Income_Group",
            "income class": "Income_Group",
            "Income_Class": "Income_Group",
            "retired": "Retired",
            "floor_area": "Floor_Area",
            "floor area": "Floor_Area",
            "house size": "Floor_Area",
            "family_size": "Family_Home",
            "Heating_Heat_Pump": "Heat_Pump",
            "heat pump heating": "Heat_Pump",
            "heating heat pump": "Heat_Pump",
            "heating_heat_pump": "Heat_Pump",
            "heat pump": "Heat_Pump",
            "Pool": "Swimming_Pool",
            "swimming_pool": "Swimming_Pool",
            "has swimming pool": "Swimming_Pool",
            "education_level": "Education_Level",
            "education level": "Education_Level",
            "higher education": "Education_Level",
            "appliances_total": "Number_Appliances",
            "appliances total": "Number_Appliances",
            "Appliances_Total": "Number_Appliances",
            "number_of_appliances": "Number_Appliances",
            "num_appliances": "Number_Appliances",
            "single family": "Single",
            "Single Family": "Single",
        }

        # Create a renaming map for non-household attributes
        more_map = {}
        for c in out.columns:
            if c == "Household":
                continue
            base = self._standardize_base(c)
            if base in pin_variants:
                more_map[c] = pin_variants[base]
            else:
                # Generic TitleCase fallback (e.g., 'num_children' -> 'NumChildren')
                more_map[c] = self._titlecase_no_spaces(c)

        out = out.rename(columns=more_map)
        out.loc[:, "Household"] = out["Household"].astype(int)
        return out

    @staticmethod
    def rearrange_data_frame(load_data: pd.DataFrame,
                             time_column='Time',
                             household_column='Household',
                             value_column='Consumption',
                             agg='sum') -> pd.DataFrame:
        """
            Convert long (Household, Time, Consumption) -> wide matrix Household x Time.
            agg: how to resolve duplicates within (Household, Time): 'sum'|'mean'|'first'
        """
        print(
            f" \n############################ 2.Step - Rearrange load data to wide format ########################################## ")
        print(f' \tShape of load data is: {load_data.shape}')
        load_data = load_data.dropna(subset=[time_column])
        load_data = load_data.sort_values([household_column, household_column])

        # If duplicates exist, aggregate first
        if agg is not None:
            load_data = (load_data
                  .groupby([household_column, time_column], as_index=False)[value_column]
                  .agg(agg))

        # Pivot to wide
        wide = load_data.pivot(index=household_column, columns=time_column, values=value_column)

        # Sort time columns chronologically
        wide = wide.sort_index(axis=1)
        print(f' \tShape of load data after rearranging is: {wide.shape}')
        return wide

    @staticmethod
    def create_weekly_load_data(load_data: pd.DataFrame,
                                time_column='Time',
                                household_column='Household',
                                value_column='Consumption',
                                agg='sum',
                                freq='15min',
                                start_date='2017-10-01') -> pd.DataFrame:
        print(
            f" \n############################ 2.Step - Creating weekly load data structure ########################################## ")
        print(f' \tShape of load data is: {load_data.shape}')
        load_data[value_column] = load_data[value_column].astype('float32')
        load_data = load_data.dropna(subset=[time_column])
        time_resolution = int(freq[0:2])
        values_per_day = int(24 * (60/time_resolution))

        print(f' \tStarted aggregating load data if duplicates exist.')
        # If duplicates exist, aggregate first
        if agg is not None:
            load_data = (load_data
                  .groupby([household_column, time_column], as_index=False)[value_column]
                  .agg(agg))

        # Get Year and Week from datetime (keep as small ints)
        load_data[time_column] = pd.to_datetime(load_data[time_column])
        iso = load_data[time_column].dt.isocalendar()
        load_data["Year"] = iso.year.astype("int16")
        load_data["Week"] = iso.week.astype("int8")

        load_data["Year_Week"] = (load_data["Year"].astype("int32") * 100 + load_data["Week"].astype("int32")).astype(
            "int32")

        # Ensure the data is sorted by ID, Year-Week, and Time
        load_data = load_data.sort_values(by=[household_column, time_column, 'Year_Week'])

        # Get the expected number of measurement for oen week, starting with monday
        week_start = pd.Timestamp(start_date)  # any Monday
        week_end = week_start + pd.Timedelta(days=7)
        expected_intervals = len(pd.date_range(start=week_start, end=week_end, freq=freq, inclusive='left'))

        print(f' \tStarted filtering incomplete weeks: {load_data.shape}')
        # Filter incomplete weeks
        complete_weeks = load_data.groupby([household_column, 'Year_Week']).filter(
            lambda x: len(x) >= expected_intervals
        )
        print(f' \tShape of load data after filtering incomplete weeks: {complete_weeks.shape}')

        # 15-min slot within day: 0..95
        slot = (complete_weeks[time_column].dt.hour * 60 + complete_weeks[time_column].dt.minute) // time_resolution
        day = complete_weeks[time_column].dt.dayofweek.astype("int8")

        # Compact day_time code: 0..(7*96-1)=671
        complete_weeks["Day_Time"] = (day.astype("int16") * values_per_day + slot.astype("int16")).astype("int16")

        print(f' \tStarted checking duplicates in (Household, Year_Week, Day_Time).')
        # Check if any duplicates have been created
        dupes = complete_weeks.duplicated(subset=[household_column, 'Year_Week', 'Day_Time'])
        print(f' \tAny duplicates in (Household, Year_Week, Day_Time)?, {dupes.any()}')

        print(f' \tStarted with pivoting dataframe! ')
        complete_weeks = complete_weeks.drop_duplicates(
            subset=[household_column, "Year_Week", "Day_Time"],
            keep="first"
        )

        pivot_df = complete_weeks.pivot(
            index=[household_column, "Year_Week"],
            columns="Day_Time",
            values=value_column
        )
        print(f' \tFinished with pivoting dataframe! ')

        # Pivot into 1 row per (unique_id, Year_Week), 672 columns like "Monday_00:15:00"
        # pivot_df = complete_weeks.pivot_table(
        #     index=[household_column, 'Year_Week'],
        #     columns='Day_Time',
        #     values=value_column
        # )

        if pivot_df.isnull().values.any():
            print(f' \tSome NaNs occurred during computing pivot table! ')
            pivot_df = pivot_df.fillna(0)

        # Convert numeric Day_Time columns (0..671) -> "d_HH:MM:SS"
        cols = pivot_df.columns.to_numpy()
        day = cols // values_per_day
        slot = cols % values_per_day
        minutes = slot * time_resolution
        hh = minutes // 60
        mm = minutes % 60
        pivot_df.columns = [f"{d}_{h:02d}:{m:02d}:00" for d, h, m in zip(day, hh, mm)]

        idx = pivot_df.index
        hh = idx.get_level_values("Household").astype(str)
        yw = idx.get_level_values("Year_Week").astype(int)

        year = yw // 100
        week = yw % 100

        pivot_df.index = pd.MultiIndex.from_arrays(
            [hh.astype(int), [f"{y}-{w:02d}" for y, w in zip(year, week)]],
            names=["Household", "Year_Week"]
        )

        print(f" \tShape of the computed weekly snippets: {pivot_df.shape}")
        return pivot_df

    @staticmethod
    def filter_households_by_not_enough_weeks(weekly_load_data: pd.DataFrame) -> pd.DataFrame:
        print(
            f" \n############################ 3.Step - Filter households by not enough weeks ########################################## ")
        print(f' \tShape of weekly load data is: {weekly_load_data.shape}')
        data = weekly_load_data.reset_index()
        mask = data.groupby("Household")["Year_Week"].transform("nunique") >= 51
        df_filtered = data[mask]

        df_filtered = (
            df_filtered
            .set_index(["Household", "Year_Week"])
            .sort_index()
        )
        print(f" \tShape of the filtered weekly snippets: {df_filtered.shape}")
        return df_filtered

    @staticmethod
    def filter_datasets_by_household_id(load_data: pd.DataFrame, categorical_data: pd.DataFrame) -> pd.DataFrame:
        print(f" \n############################ 4.Step - Filter load data by household id ########################################## ")
        print(f' \tShape of load data is: {load_data.shape}')
        load_data = load_data.reset_index()
        unique_ids = set(load_data["Household"].unique())
        filtered_categorical = categorical_data[categorical_data["Household"].isin(unique_ids)].copy()
        filtered_categorical_unique = filtered_categorical.drop_duplicates(subset="Household", keep="first")
        filtered_load_data = load_data[load_data['Household'].isin(set(filtered_categorical_unique['Household']))]

        filtered_load_data = (
            filtered_load_data
            .set_index(["Household", "Year_Week"])
            .sort_index()
        )
        print(f' \tShape of load data after filtering is: {filtered_load_data.shape}')
        print(f' \tShape of categorical data after filtering is: {filtered_categorical_unique.shape}')
        return filtered_load_data, filtered_categorical_unique