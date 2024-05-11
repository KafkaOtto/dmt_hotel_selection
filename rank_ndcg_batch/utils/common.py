import pandas as pd
import numpy as np

categorical_features = [
    "prop_starrating", "prop_review_score",
    "prop_brand_bool", "promotion_flag",
    "srch_saturday_night_bool", "random_bool", "comp1_rate", "comp1_inv",
     "comp2_rate", "comp2_inv",
    "comp3_rate", "comp3_inv",  "comp4_rate", "comp4_inv",
     "comp5_rate", "comp5_inv",
    "comp6_rate", "comp6_inv",  "comp7_rate", "comp7_inv",
     "comp8_rate", "comp8_inv"
]

categorical_features = [
    "prop_starrating", "prop_review_score",
    "prop_brand_bool", "promotion_flag",
    "srch_saturday_night_bool", "random_bool", "comp1_rate", "comp1_inv",
     "comp2_rate", "comp2_inv",
    "comp3_rate", "comp3_inv",  "comp4_rate", "comp4_inv",
     "comp5_rate", "comp5_inv",
    "comp6_rate", "comp6_inv",  "comp7_rate", "comp7_inv",
     "comp8_rate", "comp8_inv"
]

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

training_extras = [
    "gross_bookings_usd", "position", "booking_bool", "click_bool",
]

drop_columns = [
    "srch_id", "date_time", "visitor_location_country_id", "prop_id", "srch_destination_id"
]

emb_dims = [[35, 18], [173, 86], [7, 7], [12, 6], [3, 3], [3, 3], [3, 3], [3, 3], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]

def get_n_numerical():
    return len(numerical_features)
def load_data_ele(file_path):
    df = pd.read_csv(file_path)
    if output_column not in df.columns:
        y = np.zeros(df.shape[0], dtype=np.float32)
    else:
        y = df[output_column].astype(np.float32).values
    x_num = df[numerical_features].astype(np.float32).values
    x_cat = df[categorical_features].astype(np.float32).values
    query_ids = df[[query_column, hotel_column]].values
    return x_num, x_cat, y, query_ids

# def load_index_cols(file_path):
#     df = pd.read_csv(file_path)
#     query_ids = df[query_column]
#     prop_ids = df[hotel_column]
#     return query_ids, prop_ids