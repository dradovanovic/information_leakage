from pathlib import Path
from Preprocessor import preprocess_data

# -----------------------
# Main
# -----------------------
def main():
    # Base folder for resolving relative paths
    data_folder = Path(__file__).resolve().parent / "data"

    # Specify input data paths
    input_categorical = 'socio_data_2011.h5'
    input_load = '00_raw_data/input_load_profiles_2011.h5'
    # input_categorical = 'cer_socio_data_2011.h5'
    # input_load = '00_raw_data/cer_load_data_2011.h5'
    year = input_categorical.split("_")[-1].split(".")[0]

    # Desired output form (yearly or weekly)
    load_data_format = 'weekly'

    # Combine base folder with data paths
    path_categorical = data_folder / input_categorical
    path_load = data_folder / input_load

    # Read load and categorical data
    data_preprocessor = preprocess_data.PreProcessor(path_load, path_categorical)
    load_df, categorical_df = data_preprocessor.read_data()

    # Normalize dataframe columns (standard naming convention)
    load_df = data_preprocessor.unify_columns_load_data(load_df)
    categorical_df = data_preprocessor.unify_columns_categorical_data(categorical_df)

    if load_data_format == 'yearly':
        # Rearrange the long dataframe format to wide format for the yearly snippet
        rearranged_yearly_load_data = data_preprocessor.rearrange_data_frame(load_df)

        output_path_yearly_load = data_folder / ('yearly_structure_' + year + '.h5')
        rearranged_yearly_load_data.to_hdf(output_path_yearly_load, key='yearly_load_data')
        print(f"✅ Finished with creating yearly load data!")
        print(f"💾 Saved yearly load data to: {output_path_yearly_load}")

    elif load_data_format == 'weekly':
        # Create weekly snippet
        if year == '2011' and 'cer' in input_load:
            weekly_load_data = data_preprocessor.create_weekly_load_data(load_df,
                                                                         freq='30min',
                                                                         start_date='2009-07-14')
            # Define output paths
            output_path_weekly_load = data_folder / ('cer_weekly_structure_' + year + '.h5')
            output_path_categorical = data_folder / ('cer_filtered_categorical_' + year + '.h5')

        elif year == '2011':
            weekly_load_data = data_preprocessor.create_weekly_load_data(load_df,
                                                                         freq='15min',
                                                                         start_date='2010-04-01')
            # Define output paths
            output_path_weekly_load = data_folder / ('weekly_structure_' + year + '.h5')
            output_path_categorical = data_folder / ('filtered_categorical_' + year + '.h5')

        else:
            weekly_load_data = data_preprocessor.create_weekly_load_data(load_df,
                                                                         freq='15min',
                                                                         start_date='2017-10-01')
            # Define output paths
            output_path_weekly_load = data_folder / ('weekly_structure_' + year + '.h5')
            output_path_categorical = data_folder / ('filtered_categorical_' + year + '.h5')

        weekly_load_data = data_preprocessor.filter_households_by_not_enough_weeks(weekly_load_data)

        # Filter households where no labels are present
        filtered_load_data, filtered_categorical_data = data_preprocessor.filter_datasets_by_household_id(weekly_load_data,
                                                                                                          categorical_df)

        print(f"✅ Finished with crating weekly load data!")
        filtered_load_data.to_hdf(output_path_weekly_load, key='weekly_load_data')
        print(f"💾 Saved weekly load data to {output_path_weekly_load}")

        # Save Categorical Data
        print(f'✅ Finished with filtering categorical data!')
        filtered_categorical_data.to_hdf(output_path_categorical, key='filtered_categorical_data')
        print(f"💾 Saved categorical data to {output_path_categorical}")

    else:
        print(f' \t ❌ Specified load data format: {load_data_format} is not supported.')

if __name__ == "__main__":
    main()
