customer_features = {
    "visitor_location_country_id": "The ID of the country the customer is located",
    "visitor_hist_starrating": "The mean star rating of hotels the customer has previously purchased; null signifies there is no purchase history on the customer",
    "visitor_hist_adr_usd": "The mean price per night (in US$) of the hotels the customer has previously purchased; null signifies there is no purchase history on the customer"
}

hotel_features = {
    "prop_country_id": "The ID of the country the hotel is located in",
    "prop_id": "The ID of the hotel",
    "prop_starrating": "The star rating of the hotel, from 1 to 5, in increments of 1. A 0 indicates the property has no stars, the star rating is not known or cannot be publicized",
    "prop_review_score": "The mean customer review score for the hotel on a scale out of 5, rounded to 0.5 increments. A 0 means there have been no reviews, null that the information is not available",
    "prop_brand_bool": "+1 if the hotel is part of a major hotel chain; 0 if it is an independent hotel",
    "prop_location_score1": "A (first) score outlining the desirability of a hotel’s location",
    "prop_location_score2": "A (second) score outlining the desirability of the hotel’s location",
    "prop_log_historical_price": "The logarithm of the mean price of the hotel over the last trading period. A 0 will occur if the hotel was not sold in that period",
    "price_usd": "Displayed price of the hotel for the given search. Note that different countries have different conventions regarding displaying taxes and fees and the value may be per night or for the whole stay",
    "promotion_flag": "+1 if the hotel had a sale price promotion specifically displayed"
}

search_features = {
    "srch_id": "The ID of the search",
    "date_time": "Date and time of the search",
    "srch_destination_id": "ID of the destination where the hotel search was performed",
    "srch_length_of_stay": "Number of nights stay that was searched",
    "srch_booking_window": "Number of days in the future the hotel stay started from the search date",
    "srch_adults_count": "The number of adults specified in the hotel room",
    "srch_children_count": "The number of (extra occupancy) children specified in the hotel room",
    "srch_room_count": "Number of hotel rooms specified in the search",
    "srch_saturday_night_bool": "+1 if the stay includes a Saturday night, starts from Thursday with a length of stay is less than or equal to 4 nights (i.e. weekend); otherwise 0",
    "srch_query_affinity_score": "The log of the probability a hotel will be clicked on in Internet searches (hence the values are negative) A null signifies there are no data (i.e. hotel did not register in any searches)",
    "orig_destination_distance": "Physical distance between the hotel and the customer at the time of search. A null means the distance could not be calculated",
    "random_bool": "+1 when the displayed sort was random, 0 when the normal sort order was displayed"
}

website_features = {
    "site_id": "ID of the Expedia point of sale (i.e. Expedia.com, Expedia.co.uk, Expedia.co.jp, ..)",
    "comp1_rate": "+1 when the displayed sort was random, 0 when the normal sort order was displayed",
    "comp1_inv": "+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data",
    "comp1_rate_percent_diff": "The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data",
    "comp2_rate": "+1 when the displayed sort was random, 0 when the normal sort order was displayed",
    "comp2_inv": "+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data",
    "comp2_rate_percent_diff": "The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data",
    "comp3_rate": "+1 when the displayed sort was random, 0 when the normal sort order was displayed",
    "comp3_inv": "+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data",
    "comp3_rate_percent_diff": "The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data",
    "comp4_rate": "+1 when the displayed sort was random, 0 when the normal sort order was displayed",
    "comp4_inv": "+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data",
    "comp4_rate_percent_diff": "The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data",
    "comp5_rate": "+1 when the displayed sort was random, 0 when the normal sort order was displayed",
    "comp5_inv": "+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data",
    "comp5_rate_percent_diff": "The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data",
    "comp6_rate": "+1 when the displayed sort was random, 0 when the normal sort order was displayed",
    "comp6_inv": "+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data",
    "comp6_rate_percent_diff": "The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data",
    "comp7_rate": "+1 when the displayed sort was random, 0 when the normal sort order was displayed",
    "comp7_inv": "+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data",
    "comp7_rate_percent_diff": "The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data",
    "comp8_rate": "+1 when the displayed sort was random, 0 when the normal sort order was displayed",
    "comp8_inv": "+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data",
    "comp8_rate_percent_diff": "The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data",
}

training_outputs = {
    "position": "Hotel position on Expedia’s search results page.",
    "click_bool": "1 if the user clicked on the property, 0 if not",
    "booking_bool": "1 if the user booked the property, 0 if not",
    "gross_booking_usd": "Total value of the transaction. This can differ from the price_usd due to taxes, fees, conven- tions on multiple day bookings and purchase of a room type other than the one shown in the search"
}