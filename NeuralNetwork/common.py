
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