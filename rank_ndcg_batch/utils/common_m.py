import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

categorical_features = [
    "site_id", "prop_country_id", "prop_starrating", "prop_review_score",
    "prop_brand_bool", "promotion_flag",
    "srch_saturday_night_bool", "random_bool", "comp1_rate", "comp1_inv",
     "comp2_rate", "comp2_inv",
    "comp3_rate", "comp3_inv",  "comp4_rate", "comp4_inv",
     "comp5_rate", "comp5_inv",
    "comp6_rate", "comp6_inv",  "comp7_rate", "comp7_inv",
     "comp8_rate", "comp8_inv"
]

# categorical_features = [
#     "prop_brand_bool", "promotion_flag",
#     "srch_saturday_night_bool", "random_bool", "comp1_rate", "comp1_inv",
#      "comp2_rate", "comp2_inv",
#     "comp3_rate", "comp3_inv",  "comp4_rate", "comp4_inv",
#      "comp5_rate", "comp5_inv",
#     "comp6_rate", "comp6_inv",  "comp7_rate", "comp7_inv",
#      "comp8_rate", "comp8_inv"
# ]

output_column = 'label'

query_column = 'srch_id'

hotel_column = 'prop_id'

numerical_features = [
    "visitor_hist_starrating", "visitor_hist_adr_usd", "prop_location_score1",
    "prop_location_score2", "prop_log_historical_price", "price_usd", "srch_length_of_stay",
    "srch_booking_window", "srch_adults_count", "srch_children_count",
    "srch_room_count", "srch_query_affinity_score", "orig_destination_distance",
    "comp1_rate_percent_diff", "comp2_rate_percent_diff", "comp3_rate_percent_diff",
    "comp4_rate_percent_diff", "comp5_rate_percent_diff", "comp6_rate_percent_diff",
    "comp7_rate_percent_diff", "comp8_rate_percent_diff"
]

add_features = [
    "year", "day", "month"
]

training_extras = [
    "gross_bookings_usd", "position", "booking_bool", "click_bool",
]

drop_columns = [
    "srch_id", "date_time", "visitor_location_country_id", "prop_id", "srch_destination_id"
]

in_scaler = None

emb_dims = [[35, 18], [173, 86], [7, 7], [12, 6], [3, 3], [3, 3], [3, 3], [3, 3], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]

def get_n_numerical():
    return len(numerical_features)

def compute_missing_values(df_):
    concat_features = numerical_features + categorical_features
    for feat in concat_features:
        # if 'rate_percent_diff' in feat or 'srch_query_affinity_score' in feat:
        mean_ = df_[feat].mean()
        df_[feat] = df_[feat].fillna(0)
    return df_
def add_date_time(df_):
    df_['date_time'] = pd.to_datetime(df_['date_time'])
    df_['day'] = df_["date_time"].apply(lambda x: x.day)
    df_['month'] = df_["date_time"].apply(lambda x: x.month)
    df_['year'] = df_["date_time"].apply(lambda x: x.year)
    return df_

def add_label(df_in):
    new_labels = df_in.groupby('srch_id').apply(assign_label).explode()
    # Add the new labels as a column to the original DataFrame
    df_in['label'] = new_labels.values
    # df_in['label'] = 5 * df_in['booking_bool'] + (df_in['click_bool'] & ~df_in['booking_bool'])
    return df_in

def scaler_fit(df_in):
    all_features = numerical_features + categorical_features + add_features
    if in_scaler is not None:
        df_in[all_features] = in_scaler.transform(df_in[all_features])
    else:
        scaler = StandardScaler()
        scaler = scaler.fit(df_in[all_features])
        df_in[all_features] = scaler.transform(df_in[all_features])
    return df_in

def split_train_eval(train_df_in):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    split = splitter.split(train_df_in, groups=train_df_in['srch_id'])
    train_inds, eval_inds = next(split)
    train_data_out = train_df_in.iloc[train_inds]
    eval_data_out = train_df_in.iloc[eval_inds]
    return train_data_out, eval_data_out
def assign_label(group):
    labels = []

    # Extract relevant columns
    booking_bool = group['booking_bool']
    click_bool = group['click_bool']
    position = group['position']
    group_size = len(group)
    label_idx = group_size - 1

    # Assign labels based on booking_bool and click_bool
    for bb in booking_bool:
        if bb:
            labels.append(label_idx)
            label_idx -= 1
        else:
            labels.append(0)
    for i in range(group_size):
        if click_bool.iloc[i] and not booking_bool.iloc[i]:
            labels[i] = label_idx
            label_idx -= 1

    # Calculate remaining positions to assign labels 1 or 2
    positions_with_label_bb_or_cb = [p for p, bb, cb in zip(position, booking_bool, click_bool,) if bb or cb]

    # Distribute remaining labels evenly based on position
    positions = sorted(set(position) - set(positions_with_label_bb_or_cb))

    for p in positions:
        p_indices = [i for i, pos in enumerate(position) if pos == p]
        num_indices = len(p_indices)

        for ind in range(num_indices):
            labels[p_indices[ind]] = label_idx
            label_idx -= 1

    return labels

# Define a function to assign labels based on the rules
# def assign_label(group):
#     labels = []
#
#     # Extract relevant columns
#     booking_bool = group['booking_bool']
#     click_bool = group['click_bool']
#     position = group['position']
#     num_positions = len(group)
#
#     # Assign labels based on booking_bool and click_bool
#     for bb, cb in zip(booking_bool, click_bool):
#         if bb:
#             labels.append(4)
#         elif cb:
#             labels.append(3)
#         else:
#             labels.append(0)
#
#     # Calculate remaining positions to assign labels 1 or 2
#     positions_with_label_3_or_4 = [p for p, label in zip(position, labels) if label in (3, 4)]
#     num_assigned_3_or_4 = labels.count(3) + labels.count(4)
#     num_remaining = num_positions - num_assigned_3_or_4
#
#     if num_remaining > 0:
#         # Distribute remaining labels evenly based on position
#         positions = sorted(set(position) - set(positions_with_label_3_or_4))
#         num_labels_per_position = num_remaining // 3
#         num_to_assigned = num_labels_per_position * 2
#         assigned = 0
#
#         for p in positions:
#             p_indices = [i for i, pos in enumerate(position) if pos == p]
#             num_indices = len(p_indices)
#
#             for ind in range(num_indices):
#                 labels[p_indices[ind]] = 2 if assigned < num_labels_per_position else 1
#                 assigned += 1
#                 if assigned >= num_to_assigned:
#                     break
#             if assigned >= num_to_assigned:
#                 break
#
#     return labels


def load_train_eval_df(file_path):
    df = pd.read_csv(file_path)
    df = compute_missing_values(df)
    df = add_date_time(df)
    df = add_label(df)
    df = scaler_fit(df)
    df_train, df_eval = split_train_eval(df)
    return df_train, df_eval

def load_test_df(file_path):
    df_test = pd.read_csv(file_path)
    df_test = compute_missing_values(df_test)
    df_test = add_date_time(df_test)
    df_test = scaler_fit(df_test)
    return df_test

def load_df_to_features(df):
    if output_column not in df.columns:
        y = np.zeros(df.shape[0], dtype=np.float32)
    else:
        y = df[output_column].astype(np.float32).values
    x_num = df[numerical_features + add_features].astype(np.float32).values
    x_cat = df[categorical_features].astype(np.float32).values
    query_ids = df[[query_column, hotel_column]].values
    return x_num, x_cat, y, query_ids

# def load_index_cols(file_path):
#     df = pd.read_csv(file_path)
#     query_ids = df[query_column]
#     prop_ids = df[hotel_column]
#     return query_ids, prop_ids