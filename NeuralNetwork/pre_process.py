import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from tqdm.notebook import tqdm
from sklearn.model_selection import GroupShuffleSplit
from utils import seed_everything
import argparse
from common import categorical_features, numerical_features

def operate_missing_value(train_df_, test_df_):
    for feat in tqdm(categorical_features):
        train_df_[feat] = train_df_[feat].fillna(-2)
        test_df_[feat] = test_df_[feat].fillna(-2)
    train_test_numerical = pd.concat([train_df_[numerical_features], test_df_[numerical_features]])
    for feat in tqdm(numerical_features):
        if (feat == 'srch_query_affinity_score'):
            # fill srch_query_affinity_score with smallest possibility
            smallest_ = train_test_numerical[feat].min()
            train_df_[feat] = train_df_[feat].fillna(smallest_)
            test_df_[feat] = test_df_[feat].fillna(smallest_)
        if 'rate_percent_diff' in feat:
            train_df_[feat] = train_df_[feat].fillna(0)
            test_df_[feat] = test_df_[feat].fillna(0)
        else:
            # fill others with mean value
            mean_ = train_test_numerical[feat].mean()
            train_df_[feat] = train_df_[feat].fillna(mean_)
            test_df_[feat] = test_df_[feat].fillna(mean_)
    return train_df_, test_df_

def add_label(df_in):
    df_in['label'] = 2 * df_in['booking_bool'] + (df_in['click_bool'] & ~df_in['booking_bool'])
    return df_in

def normalized_categorical_features(train_df_in, test_df_in):
    train_test_categorial = pd.concat([train_df_in[categorical_features], test_df_in[categorical_features]])
    for feat in categorical_features:
        lbl_enc = LabelEncoder()
        lbl_enc = lbl_enc.fit(train_test_categorial[feat].values)
        train_df_in[feat] = lbl_enc.transform(train_df_in[feat].values)
        test_df_in[feat] = lbl_enc.transform(test_df_in[feat].values)
    return train_df_in, test_df_in


def normalized_numerical_features(train_df_in, test_df_in):
    # Concatenate train and test dataframes
    train_test_numerical = pd.concat([train_df_in[numerical_features], test_df_in[numerical_features]], axis=0)
    # Initialize StandardScaler
    stnd_scl = StandardScaler()
    # Fit the StandardScaler on the concatenated dataframe
    stnd_scl.fit(train_test_numerical)
    # Transform train and test numerical features
    train_numerical_features = stnd_scl.transform(train_df_in[numerical_features])
    test_numerical_features = stnd_scl.transform(test_df_in[numerical_features])
    # Assign transformed features back to dataframes
    train_df_in.loc[:, numerical_features] = train_numerical_features
    test_df_in.loc[:, numerical_features] = test_numerical_features

    return train_df_in, test_df_in

def split_train_eval(train_df_in):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    split = splitter.split(train_df_in, groups=train_df_in['srch_id'])
    train_inds, eval_inds = next(split)
    train_data_out = train_df_in.iloc[train_inds]
    eval_data_out = train_df_in.iloc[eval_inds]
    return train_data_out, eval_data_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/', help='The data directory.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--batch', type=int, default=2000, help='Load Batch Size')
    parser.add_argument('--epoches', type=int, default=100, help='Number of Epoches')
    parser.add_argument('--seed', type=int, default=123, help='Random Seed')
    args = vars(parser.parse_args())

    seed_everything(seed=args['seed'])
    training_path = "../data/training_set_VU_DM.csv"
    test_path = "../data/test_set_VU_DM.csv"
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(test_path)
    # integrate booking_bool and click_bool into label
    train_df['label'] = 2 * train_df['booking_bool'] + (train_df['click_bool'] & ~train_df['booking_bool'])
    # test_df['label'] = test_df.apply(assign_target, axis=1)
    # apply LabelCoder into categorical variables
    train_df, test_df = operate_missing_value(train_df, test_df)
    train_test_categorial = pd.concat([train_df[categorical_features], test_df[categorical_features]])
    for feat in categorical_features:
        lbl_enc = LabelEncoder()
        lbl_enc = lbl_enc.fit(train_test_categorial[feat].values)
        train_df[feat] = lbl_enc.transform(train_df[feat].values)
        test_df[feat] = lbl_enc.transform(test_df[feat].values)
        print(f"for feat: {feat}, unique value for train: {train_df[feat].unique()}, for test: {test_df[feat].unique()}")
    train_test_numerical = pd.concat([train_df[numerical_features], test_df[numerical_features]])
    stnd_scl = StandardScaler()
    stnd_scl = stnd_scl.fit(train_test_numerical)
    train_numerical_features = stnd_scl.transform(train_df[numerical_features])
    test_numerical_features = stnd_scl.transform(test_df[numerical_features])
    # Splitting data based on srch_id

    # train_df[categorical_features] = train_categorical_features
    # test_df[categorical_features] = test_categorical_features
    train_df[numerical_features] = train_numerical_features
    test_df[numerical_features] = test_numerical_features
    print(train_df.describe())
    print(test_df.describe())
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args['seed'])
    split = splitter.split(train_df, groups=train_df['srch_id'])
    train_inds, eval_inds = next(split)
    train_data = train_df.iloc[train_inds]
    eval_data = train_df.iloc[eval_inds]

    # Keep only necessary columns for train_data and eval_data
    # train_data = train_data[['label'] + categorical_features + numerical_features]
    # eval_data = eval_data[['label'] + categorical_features + numerical_features]

    # Keep only categorical and numerical columns for test_data

    # Save training data
    train_data.to_csv("../data/train_data.csv", index=False)
    # Save evaluation data
    eval_data.to_csv("../data/eval_data.csv", index=False)
    # Save test data
    test_df.to_csv("../data/test_data.csv", index=False)
