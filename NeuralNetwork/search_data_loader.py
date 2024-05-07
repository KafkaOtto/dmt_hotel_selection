import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np

from common import categorical_features, numerical_features, drop_columns, training_extras
from pre_process import operate_missing_value, add_label,normalized_categorical_features, normalized_numerical_features, split_train_eval

# class LoadDataSearch():
#     def __init__(self, path="../data/"):
#         self.path = path
#         self.trainfile = self.path + "training_set_VU_DM.csv"
#         self.features_M = {}
#         self.construct_df()
#
#     def construct_df(self):
#         self.data_train = pd.read_csv(self.trainfile)
#
#         self.field_dims = []
#
#         for i in self.all_data.columns[1:]:
#             maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
#             self.all_data[i] = self.all_data[i].map(maps)
#             self.features_M[i] = maps
#             self.field_dims.append(len(set(self.all_data[i])))
#         self.all_data[0] = self.all_data[0].apply(lambda x: max(x, 0))

class SearchDataSet(torch.utils.data.Dataset):
    def __init__(self, data, cat_cols, output_col=None):
        self.n = data.shape[0]
        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns
                          if col not in self.cat_cols + [output_col]]
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]

def load_data(path="../data/"):
    # train_df = pd.read_csv(path + "training_set_VU_DM.csv")
    # test_df = pd.read_csv(path + "test_set_VU_DM.csv")
    #
    # train_df = add_label(train_df)
    # train_df, test_df = operate_missing_value(train_df, test_df)
    # train_df, test_df = normalized_categorical_features(train_df, test_df)
    # train_df, test_df = normalized_numerical_features(train_df, test_df)
    # train_df, eval_df = split_train_eval(train_df)
    train_df = pd.read_csv(path + "train_data.csv")
    test_df = pd.read_csv(path + "test_data.csv")
    eval_df = pd.read_csv(path + "eval_data.csv")

    print(f"training columns before drop: {train_df.columns}")
    print(f"eval columns before drop: {eval_df.columns}")
    print(f"test columns before drop: {test_df.columns}")

    train_df.drop(columns=drop_columns + training_extras, axis=1, inplace=True)
    eval_df.drop(columns=drop_columns + training_extras, axis=1, inplace=True)
    test_df.drop(columns=drop_columns, axis=1, inplace=True)

    print(f"training columns after drop: {train_df.columns}")
    print(f"eval columns after drop: {eval_df.columns}")
    print(f"test columns after drop: {test_df.columns}")

    return train_df, eval_df, test_df
def calculate_cat_dim(train_cat_data, eval_cat_data):
    combined_cat_data = np.concatenate([train_cat_data, eval_cat_data], axis=0)
    cat_dim = []
    for feature_values in combined_cat_data.T:  # Iterate over each categorical feature
        unique_values = np.unique(feature_values)  # Count unique values in the feature
        cat_dim.append(len(unique_values) + 1)  # Add 1 to handle out-of-range indices
    return cat_dim

def get_search_dataloader(train_df, eval_df, test_df, batch_size = 64):

    all_df = pd.concat([train_df, eval_df])
    cat_dims = [int(all_df[col].nunique()) for col in categorical_features]
    emb_dims = [[x, min(200, (x + 1) // 2)] for x in cat_dims]
    for el in emb_dims:
        if el[0] < 10:
            el[1] = el[0]
    print(emb_dims)

    train_set = SearchDataSet(train_df, categorical_features, output_col='label')
    eval_set = SearchDataSet(eval_df, categorical_features, output_col='label')
    test_set = SearchDataSet(test_df, categorical_features, output_col=None)

    print(len(train_set))
    print(len(eval_set))
    print(len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size)
    eval_loader = DataLoader(eval_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return emb_dims, train_loader, eval_loader, test_loader