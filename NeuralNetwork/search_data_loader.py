import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np

from common import categorical_features, numerical_features, drop_columns, training_extras

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
    def __init__(self, cat_data, num_data, targets=None):
        self.cat_data = cat_data
        self.num_data = num_data
        self.targets = targets

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        result = None
        data_cat = torch.tensor(self.cat_data[idx].astype('float32'))

        data_cont = torch.tensor(self.num_data[idx].astype('float32'))

        data = [data_cat, data_cont]

        if self.targets is not None:
            result = {'data': data,
                      'target': torch.tensor(self.targets[idx])}
        else:
            result = {'data': data}

        return result

def get_col_idx(in_df, col_names):
    idxs = list()
    for idx, col in enumerate(in_df.columns):
        if col in col_names:
            idxs.append(idx)
    return idxs

def load_data(path="../data/"):
    train_df = pd.read_csv(path + "train_data.csv")
    test_df = pd.read_csv(path + "test_data.csv")
    eval_df = pd.read_csv(path + "eval_data.csv")
    return train_df, eval_df, test_df
def calculate_cat_dim(train_cat_data, eval_cat_data):
    combined_cat_data = np.concatenate([train_cat_data, eval_cat_data], axis=0)
    cat_dim = []
    for feature_values in combined_cat_data.T:  # Iterate over each categorical feature
        unique_values = np.unique(feature_values)  # Count unique values in the feature
        cat_dim.append(len(unique_values) + 1)  # Add 1 to handle out-of-range indices
    return cat_dim

def get_search_dataloader(train_df, eval_df, test_df, batch_size = 64):
    train_cat_data = train_df[categorical_features].values
    train_num_data = train_df[numerical_features].values
    train_target = train_df['label'].values

    eval_cat_data = eval_df[categorical_features].values
    eval_num_data = eval_df[numerical_features].values
    eval_target = eval_df['label'].values


    cat_dim = calculate_cat_dim(train_cat_data, eval_cat_data)
    cat_dim = [[x, min(200, (x + 1) // 2)] for x in cat_dim]

    for el in cat_dim:
        if el[0] < 10:
            el[1] = el[0]
    print(cat_dim)

    test_cat_data = test_df[categorical_features].values
    test_num_data = test_df[numerical_features].values

    train_set = SearchDataSet(train_cat_data, train_num_data, targets=train_target)
    eval_set = SearchDataSet(eval_cat_data, eval_num_data, targets=eval_target)
    test_set = SearchDataSet(test_cat_data, test_num_data, targets=None)

    print(len(train_set))
    print(len(eval_set))
    print(len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size)
    eval_loader = DataLoader(eval_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return cat_dim, train_loader, eval_loader, test_loader