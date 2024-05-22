import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from rank_ndcg_batch.utils.data_pre_utils import aggregated_features_single_column, normalize_features, input_estimated_position
import gc

categorical_features = [
    "bool_same_country",
    "prop_country_id", "prop_starrating", "prop_review_score",
    "prop_brand_bool", "promotion_flag",
    "visitor_location_country_id",
    "srch_destination_id",
    "srch_saturday_night_bool"
]

output_column = 'label'

query_column = 'srch_id'

hotel_column = 'prop_id'

numerical_features = [
    "starrating_diff", "usd_diff", "ad_vs_real",
    "avg_comp_rate",
    "avg_rate_percent_diff",
    "roomcount_bookwindow", "adultcount_childrencount",
    "dayofweek",
    "month",
    "hour",
     "prop_location_score1",
    "prop_location_score2", "prop_log_historical_price",
    "price_usd",
    "visitor_hist_adr_usd",
    "srch_length_of_stay",
     "srch_query_affinity_score",
    "booked_perc", "clicked_perc", "orig_destination_distance",
'MEAN_prop_id_price_usd_',
 'SUBSTRACT_price_usd_MEAN_', 'MEAN_srch_id_prop_starrating_',
 'SUBSTRACT_prop_starrating_MEAN_', 'MEAN_srch_id_prop_location_score2_',
 'SUBSTRACT_prop_location_score2_MEAN_',
 'MEAN_srch_id_prop_location_score1_',
 'SUBSTRACT_prop_location_score1_MEAN_',
 'MEAN_srch_destination_id_price_usd_', 'MEAN_srch_id_prop_review_score_',
 'SUBSTRACT_prop_review_score_MEAN_', 'MEAN_srch_id_promotion_flag_',
 'SUBSTRACT_promotion_flag_MEAN_', "estimated_position",
    # 'price_usd_norm_by_srch_destination_id',
'price_usd_norm_by_srch_id',
    # 'prop_location_score2_norm_by_srch_destination_id',
    # 'prop_location_score1_norm_by_srch_destination_id',
    # 'prop_review_score_norm_by_srch_destination_id',
# 'prop_starrating_norm_by_srch_id',
    # 'prop_location_score2_norm_by_prop_id',
    # 'prop_location_score1_norm_by_prop_id',
    # 'prop_review_score_norm_by_prop_id',
    # 'price_usd_norm_by_prop_id',
# 'price_usd_norm_by_srch_id',
#     'prop_starrating_norm_by_srch_id',
#     'prop_location_score2_norm_by_srch_id',
#     'prop_location_score1_norm_by_srch_id',
#     'prop_review_score_norm_by_srch_id',
'price_usd_norm_by_srch_destination_id',
    # 'prop_starrating_norm_by_prop_country_id',
    # 'prop_location_score2_norm_by_prop_country_id',
    # 'prop_location_score1_norm_by_prop_country_id',
    # 'prop_review_score_norm_by_prop_country_id',

]

training_extras = [
    "gross_bookings_usd", "position", "booking_bool", "click_bool",
]

drop_columns = [
    "srch_id", "date_time", "prop_id",
    # "visitor_location_country_id",
    # "srch_destination_id",
]

emb_dims = [[35, 18], [173, 86], [7, 7], [12, 6], [3, 3], [3, 3], [3, 3], [3, 3], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]

def get_n_features():
    return len(numerical_features) + len(categorical_features)

def get_n_numerical():
    return len(numerical_features)

def scaler_fit(train_df_in):
    all_features = numerical_features + categorical_features
    scaler = StandardScaler()
    scaler = scaler.fit(train_df_in[all_features])
    train_df_in[all_features] = scaler.transform(train_df_in[all_features])
    gc.collect()
    return train_df_in, scaler

def normalized_before_aggreation(data_for_training):
    # do not normalize 2 times with take_log10
    # 'price_usd_norm_by_srch_id', 'price_usd_norm_by_prop_id', 'prop_starrating_norm_by_srch_id',
    # 'prop_location_score2_norm_by_srch_id', 'prop_location_score1_norm_by_srch_id', 'prop_review_score_norm_by_srch_id'
    # price_usd_norm_by_srch_id
    data_for_training = normalize_features(
        data_for_training,
        group_key="srch_id",
        target_column="price_usd",
        take_log10=True,
    )
    data_for_training = normalize_features(
        data_for_training,
        group_key="visitor_location_country_id",
        target_column="visitor_hist_adr_usd",
        take_log10=True,
    )
    data_for_training = normalize_features(
        data_for_training,
        group_key="srch_destination_id",
        target_column="price_usd",
        take_log10=False,
    )
    # price_usd_norm_by_prop_id
    # data_for_training = normalize_features(
    #     data_for_training, group_key="prop_id", target_column="price_usd", take_log10=True,
    # )
    # prop_starrating_norm_by_srch_id
    # data_for_training = normalize_features(
    #     data_for_training, group_key="srch_destination_id", target_column="prop_starrating",
    # )
    # data_for_training = normalize_features(
    #     data_for_training, group_key="prop_id", target_column="prop_starrating", take_log10=True,
    # )
    # data_for_training = normalize_features(
    #     data_for_training, group_key="prop_id", target_column="prop_location_score2", take_log10=True,
    # )
    # data_for_training = normalize_features(
    #     data_for_training, group_key="prop_id", target_column="prop_location_score1", take_log10=True,
    # )
    # data_for_training = normalize_features(
    #     data_for_training, group_key="prop_id", target_column="prop_review_score", take_log10=True,
    # )
    # data_for_training = normalize_features(
    #     data_for_training, group_key="srch_id", target_column="prop_starrating",
    # )
    # data_for_training = normalize_features(
    #     data_for_training, group_key="srch_id", target_column="prop_location_score2", take_log10=False,
    # )
    # prop_location_score1_norm_by_srch_id
    # data_for_training = normalize_features(
    #     data_for_training, group_key="srch_id", target_column="prop_location_score1", take_log10=False,
    # )
    # # prop_review_score_norm_by_srch_id
    # data_for_training = normalize_features(
    #     data_for_training, group_key="srch_id", target_column="prop_review_score", take_log10=False,
    # )
    return data_for_training

def normalized_after_aggreation(data_for_training):
    # prop_location_score2_norm_by_srch_id
    # data_for_training = normalize_features(
    #     data_for_training, group_key="srch_id", target_column="prop_location_score2", take_log10=True,
    # )
    # # prop_location_score1_norm_by_srch_id
    # data_for_training = normalize_features(
    #     data_for_training, group_key="srch_id", target_column="prop_location_score1", take_log10=True,
    # )
    # # prop_review_score_norm_by_srch_id
    # data_for_training = normalize_features(
    #     data_for_training, group_key="srch_id", target_column="prop_review_score", take_log10=True,
    # )
    return data_for_training
def aggreated_columns(data_for_training):
    data_for_training = aggregated_features_single_column(
        data_for_training, "prop_id", "price_usd", ["mean"]
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="prop_starrating",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="prop_location_score2",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="prop_location_score1",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_destination_id",
        target_column="price_usd",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="prop_review_score",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    data_for_training = aggregated_features_single_column(
        data_for_training,
        key_for_grouped_by="srch_id",
        target_column="promotion_flag",
        agg_methods=["mean"],
        transform_methods={"mean": ["substract"]},
    )
    return data_for_training
def fill_nan_1(train_df, test_df):
    # Select only the necessary columns for calculating means
    combined_cols = ['prop_id'] + ["srch_query_affinity_score", "orig_destination_distance",
                                   "prop_review_score", "prop_location_score2",
                                   "visitor_hist_starrating", "visitor_hist_adr_usd"]
    combined_df = pd.concat([train_df[combined_cols], test_df[combined_cols]], ignore_index=True)

    # For each feature, fill NaNs with corresponding values grouped by prop_id
    for feature in ["srch_query_affinity_score", "orig_destination_distance", "prop_review_score",
                    "prop_location_score2", "visitor_hist_starrating", "visitor_hist_adr_usd"]:
        # Calculate means based on the combined dataframe
        prop_means = combined_df.groupby('prop_id')[feature].mean()
        # Fill NaNs in train_df based on prop_means
        train_df[feature] = train_df.apply(lambda row: row[feature] if pd.notna(row[feature]) else prop_means[row['prop_id']], axis=1)
        # Fill NaNs in test_df based on prop_means
        test_df[feature] = test_df.apply(lambda row: row[feature] if pd.notna(row[feature]) else prop_means[row['prop_id']], axis=1)
    gc.collect()
    return train_df, test_df
def fill_nan(df_):
    df_['srch_query_affinity_score'] = np.exp(df_['srch_query_affinity_score']) #this is in log terms originally, so we bring it back to normal counts
    df_["srch_query_affinity_score"] = df_["srch_query_affinity_score"].fillna(0) #impute it to 0

    df_["orig_destination_distance"] = df_["orig_destination_distance"].replace(np.nan, df_[
        "orig_destination_distance"].quantile(0.75))
    df_["prop_review_score"] = df_["prop_review_score"].fillna(0)

    # Calculate the first quartile for each prop_country_id
    df_["prop_location_score2"] = df_["prop_location_score2"].replace(np.nan,
                                                                        df_["prop_location_score2"].median(axis=0,
                                                                                                            skipna=True))
    # fill with median
    # data=data.drop("gross_bookings_usd",axis=1)
    median_vhs = df_["visitor_hist_starrating"].median(axis=0, skipna=True)
    median_vha = df_["visitor_hist_adr_usd"].median(axis=0, skipna=True)
    df_['visitor_hist_starrating'] = df_['visitor_hist_starrating'].fillna(median_vhs)
    df_['visitor_hist_adr_usd'] = df_['visitor_hist_adr_usd'].fillna(median_vha)

    # removing outliers
    # indices_to_remove = df_[df_['price_usd'] > 50000].index.tolist()
    # df_ = df_.drop(indices_to_remove)

    return df_


def add_extra_features(df_):
    df_['date_time'] = pd.to_datetime(df_['date_time'])
    df_['month'] = df_["date_time"].dt.month
    df_['dayofweek'] = df_["date_time"].dt.dayofweek
    df_['hour'] = df_["date_time"].dt.hour
    df_.drop(columns=['date_time'], inplace=True)


    df_["starrating_diff"] = abs(df_["visitor_hist_starrating"] - df_["prop_starrating"])
    epsilon = 1e-4
    df_["usd_diff"] = df_["visitor_hist_adr_usd"]- df_["price_usd"]
    df_ = df_.replace([np.inf, -np.inf], np.nan)
    df_ = df_.fillna(value={"starrating_diff": 6, "usd_diff": 1.1})
    df_['ad_vs_real'] = df_['prop_starrating'] - df_['prop_review_score']
    df_['avg_comp_rate'] = df_[['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']].mean(axis=1)
    df_ = df_.fillna(value = {"avg_comp_rate": 0})
    df_.drop(columns=['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate',
                      'comp8_rate'], inplace=True)

    df_['avg_rate_percent_diff'] = df_[['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff']].mean(axis=1)
    df_ = df_.fillna(value = {"avg_rate_percent_diff": 0})
    df_.drop(columns=['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff'], inplace=True)



    df_['roomcount_bookwindow'] = df_['srch_room_count']*max(df_['srch_booking_window']) + df_['srch_booking_window']
    df_.drop(columns=['srch_room_count', 'srch_booking_window'], inplace=True)


    df_['adultcount_childrencount'] = df_['srch_adults_count']*max(df_['srch_children_count']) + df_['srch_children_count']
    df_.drop(columns=['srch_adults_count', 'srch_children_count'], inplace=True)

    df_['bool_same_country'] = 0
    df_.loc[(df_['visitor_location_country_id'] == df_['prop_country_id']), 'bool_same_country'] = 1
    # df_.drop(columns=['visitor_location_country_id'], inplace=True)


    return df_

def add_cb_bb_perc(df_train_in, df_test_in):
    hotel_quality = df_train_in.groupby('prop_id').agg(
        counts=('prop_id', 'size'),
        booked=('booking_bool', 'sum'),
        clicked=('click_bool', 'sum')
    )
    hotel_quality['booked_perc'] = hotel_quality['booked'] / hotel_quality['counts'] * 100
    hotel_quality['clicked_perc'] = hotel_quality['clicked'] / hotel_quality['counts'] * 100

    # Calculate mean booked and clicked percentages
    booked_perc_mean = hotel_quality['booked_perc'].mean()
    clicked_perc_mean = hotel_quality['clicked_perc'].mean()

    # Fill NaN values with mean percentages
    hotel_quality.fillna({'booked_perc': booked_perc_mean, 'clicked_perc': clicked_perc_mean}, inplace=True)

    # Merge calculated percentages back to the original DataFrame
    df_train_in = df_train_in.merge(hotel_quality[['booked_perc', 'clicked_perc']], left_on='prop_id', right_index=True, how='left')

    df_test_in = df_test_in.merge(hotel_quality[['booked_perc', 'clicked_perc']], left_on='prop_id', right_index=True, how='left')
    df_test_in['booked_perc'] = df_test_in['booked_perc'].fillna(booked_perc_mean)
    df_test_in['clicked_perc'] = df_test_in['clicked_perc'].fillna(clicked_perc_mean)
    return df_train_in, df_test_in
def add_label(df_in):
    # new_labels = df_in.groupby('srch_id').apply(assign_label).explode()
    # Add the new labels as a column to the original DataFrame
    # df_in['label'] = new_labels.values
    df_in['label'] = 5 * df_in['booking_bool'] + 1 * (df_in['click_bool'] & ~df_in['booking_bool'])
    return df_in

# Define a function to assign labels based on the rules
def assign_label(group):
    labels = []

    # Extract relevant columns
    booking_bool = group['booking_bool']
    click_bool = group['click_bool']
    position = group['estimated_position']
    random_bool = group['random_bool']
    num_positions = len(group)

    # Assign labels based on booking_bool and click_bool
    for bb, cb in zip(booking_bool, click_bool):
        if bb:
            labels.append(4)
        elif cb:
            labels.append(3)
        else:
            labels.append(0)

    # Calculate remaining positions to assign labels 1 or 2
    num_assigned_3_or_4 = labels.count(3) + labels.count(4)
    num_remaining = num_positions - num_assigned_3_or_4

    if num_remaining > 0:
        # Distribute remaining labels evenly based on position
        positions = sorted(set(position), reverse=True)
        num_labels_per_position = num_remaining // 3
        num_to_assigned = num_labels_per_position * 2
        assigned = 0

        for p in positions:
            p_indices = [i for i, pos in enumerate(position) if pos == p]
            num_indices = len(p_indices)

            for ind in range(num_indices):
                if booking_bool.iloc[p_indices[ind]] or click_bool.iloc[p_indices[ind]]:
                    continue
                labels[p_indices[ind]] = 2 if assigned < num_labels_per_position else 1
                assigned += 1
                if assigned >= num_to_assigned:
                    break
            if assigned >= num_to_assigned:
                break

    return labels

def obtain_estimated_postion(train_df_in):
    srch_id_dest_id_dict = train_df_in[["srch_destination_id", "prop_id", "position"]]
    srch_id_dest_id_dict = srch_id_dest_id_dict.groupby(["srch_destination_id", "prop_id"]).agg(
        {"position": "mean"}
    ).rename(columns={"position": "estimated_position"}).reset_index()

    srch_id_dest_id_dict["srch_destination_id"] = srch_id_dest_id_dict["srch_destination_id"].astype(int)
    srch_id_dest_id_dict["prop_id"] = srch_id_dest_id_dict["prop_id"].astype(int)
    srch_id_dest_id_dict["estimated_position"] = 1 / srch_id_dest_id_dict["estimated_position"]

    return srch_id_dest_id_dict

def adjust_features(data_for_training):
    data_for_training = normalized_before_aggreation(data_for_training)
    data_for_training = add_extra_features(data_for_training)
    data_for_training = normalized_after_aggreation(data_for_training)
    data_for_training = aggreated_columns(data_for_training)
    # data_for_training = data_for_training.sort_values("srch_id")

    return data_for_training

def split_train_eval(train_df_in):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    split = splitter.split(train_df_in, groups=train_df_in['srch_id'])
    train_inds, eval_inds = next(split)
    train_data_out = train_df_in.iloc[train_inds]
    eval_data_out = train_df_in.iloc[eval_inds]
    return train_data_out, eval_data_out


def load_train_eval_test_df(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = fill_nan(train_df)
    test_df = fill_nan(test_df)

    # train_df = add_extra_features(train_df)
    # test_df = add_extra_features(test_df)

    train_df = adjust_features(train_df)
    test_df = adjust_features(test_df)

    train_df, test_df = add_cb_bb_perc(train_df, test_df)

    srch_id_dest_id_dict = obtain_estimated_postion(train_df)

    train_df = input_estimated_position(train_df, srch_id_dest_id_dict)
    test_df = input_estimated_position(test_df, srch_id_dest_id_dict)

    train_df = add_label(train_df)

    train_df, scaler = scaler_fit(train_df)
    test_df[numerical_features + categorical_features]  = scaler.transform(test_df[numerical_features + categorical_features])

    train_df, eval_df = split_train_eval(train_df)

    check_nan_inf(train_df, "traing")
    check_nan_inf(eval_df, "eval")
    check_nan_inf(test_df, "test")

    return train_df, eval_df, test_df

def check_nan_inf(df, name):
    columns_with_nan = df.columns[df.isna().any()].tolist()
    print(name, "Columns with NaN values:", columns_with_nan)
    inf_columns = df.columns[df.isin([np.inf, -np.inf]).any()]
    print(name, "Columns with Inf values:", inf_columns)

def load_from_path(train_path, eval_path, test_path):
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    test_df = pd.read_csv(test_path)

    return train_df, eval_df, test_df

def load_df_to_features(df):
    if output_column not in df.columns:
        y = np.zeros(df.shape[0], dtype=np.float32)
    else:
        y = df[output_column].astype(np.float32).values
    x_num = df[numerical_features].astype(np.float32).values
    x_cat = df[categorical_features].astype(np.float32).values
    query_ids = df[[query_column, hotel_column]].values
    return x_num, x_cat, y, query_ids


if __name__ == "__main__":
    base_path = "/Users/zhinuanguo/vu_projects/period5/data_mining_technique/assignments/assignment2/code/ExpediaPersonalizedSortCompetition/data/"
    train_path = base_path + "training_set_VU_DM.csv"
    test_path = base_path + "test_set_VU_DM.csv"
    train_df, eval_df, test_df = load_from_path(train_path, test_path)
    train_df.to_csv(base_path + "train_imp.csv", index=False, columns=numerical_features + categorical_features + query_column + hotel_column + output_column)
    eval_df.to_csv(base_path + "eval_imp.csv", index=False, columns=numerical_features + categorical_features + query_column + hotel_column + output_column)
    test_df.to_csv(base_path + "test_imp.csv", index=False, columns=numerical_features + categorical_features + query_column + hotel_column)
    print(f"save to path successful")
# def load_index_cols(file_path):
#     df = pd.read_csv(file_path)
#     query_ids = df[query_column]
#     prop_ids = df[hotel_column]
#     return query_ids, prop_ids
