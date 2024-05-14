import gc
import numpy as np
def aggregated_features_single_column(
    in_data,
    key_for_grouped_by="prop_id",
    target_column="price_usd",
    agg_methods=["mean", "median", "min", "max"],
    transform_methods={"mean": ["substract"]},
):
    df = in_data.groupby(key_for_grouped_by).agg({target_column: agg_methods})

    if isinstance(key_for_grouped_by, list):
        str_key_for_grouped_by = "|".join(key_for_grouped_by)
    else:
        str_key_for_grouped_by = key_for_grouped_by

    df.columns = df.columns.droplevel()
    col = {}
    for method in agg_methods:
        col[method] = (
            method.upper() + "_" + str_key_for_grouped_by + "_" + target_column + "_"
        )

    df.rename(columns=col, inplace=True)

    in_data = in_data.merge(df.reset_index(), on=key_for_grouped_by)
    for method_name in transform_methods:
        for applying_function in transform_methods[method_name]:
            function_data = in_data[
                method_name.upper()
                + "_"
                + str_key_for_grouped_by
                + "_"
                + target_column
                + "_"
            ]
            column_data = in_data[target_column]
            if applying_function == "substract":
                result = column_data - function_data
            elif applying_function == "divide":
                result = column_data / function_data
            else:
                continue
            in_data[
                applying_function.upper()
                + "_"
                + target_column
                + "_"
                + method_name.upper()
                + "_"
            ] = result
    gc.collect()

    return in_data
def detect_invalid_values(df, column_name):
    invalid_mask = df[column_name].isna() | np.isinf(df[column_name]) | (df[column_name].abs() > 1e308)
    if invalid_mask.any():
        print(f"Invalid values detected in {column_name} before log transformation:")
        print(df[invalid_mask])

def normalize_features(input_df, group_key, target_column, take_log10=False):

    # for numerical stability
    epsilon = 1e-4
    detect_invalid_values(input_df, target_column)

    if take_log10:
        input_df[target_column] = np.log10(input_df[target_column] + epsilon)
    methods = ["mean", "std"]

    df = input_df.groupby(group_key).agg({target_column: methods})

    df.columns = df.columns.droplevel()
    col = {}
    for method in methods:
        col[method] = target_column + "_" + method

    df.rename(columns=col, inplace=True)
    df_merge = input_df.merge(df.reset_index(), on=group_key)

    detect_invalid_values(df_merge, target_column + "_mean")
    detect_invalid_values(df_merge, target_column + "_std")


    df_merge[target_column + "_norm_by_" + group_key] = (
        df_merge[target_column] - df_merge[target_column + "_mean"]
    ) / df_merge[target_column + "_std"]
    df_merge = df_merge.drop(labels=[col["mean"], col["std"]], axis=1)

    detect_invalid_values(df_merge, target_column + "_norm_by_" + group_key)
    gc.collect()
    return df_merge

def input_estimated_position(training_data, srch_id_dest_id_dict):
    # Merge the estimated positions into the training data
    training_data = training_data.merge(
        srch_id_dest_id_dict, how="left", on=["srch_destination_id", "prop_id"]
    )
    print(training_data.head())

    # Fill NaN values in estimated_position
    # Calculate the mean for each prop_id and overall mean
    prop_mean = training_data.groupby('prop_id')['estimated_position'].transform('mean')
    overall_mean = training_data['estimated_position'].mean()

    # Fill NaNs with prop_id mean first, then overall mean if still NaN
    training_data['estimated_position'] = training_data['estimated_position'].fillna(prop_mean)
    training_data['estimated_position'] = training_data['estimated_position'].fillna(overall_mean)

    return training_data


