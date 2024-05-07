import argparse
import torch
from utils import seed_everything
from model import AdaptedGateCorssNetwork, FeedForwardNN
import numpy as np
from sklearn import metrics
from common import categorical_features, numerical_features, drop_columns, training_extras
from search_data_loader import load_data, get_search_dataloader
import pandas as pd
from tqdm.notebook import tqdm

def train_network(model, train_loader, val_loader, loss_fn, optimizer, epochs, saved_model='model.pt'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_losses = list()
    valid_losses = list()

    valid_loss_min = np.Inf
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # train_auc = 0.0
        # valid_auc = 0.0

        model.train()
        for y, cont_x, cat_x in tqdm(train_loader):
            optimizer.zero_grad()

            output = model(cont_x.to(device), cat_x.to(device))

            target = y.to(device, dtype=torch.float)

            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()
            # train_auc += metrics.roc_auc_score(batch['target'].cpu().numpy(),
            #                                    output.detach().cpu().numpy())

            train_loss += loss.item()

        model.eval()
        for y, cont_x, cat_x in tqdm(eval_loader):
            output = model(cont_x.to(device), cat_x.to(device))

            target = y.to(device, dtype=torch.float)

            loss = loss_fn(output, target)

            # valid_auc += metrics.roc_auc_score(batch['target'].cpu().numpy(),
            #                                    output.detach().cpu().numpy())
            valid_loss += loss.item()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {}. Training loss: {:.6f}. Validation loss: {:.6f}'
              .format(epoch, train_loss, valid_loss))
        # print('Training AUC: {:.6f}. Validation AUC: {:.6f}'
        #       .format(train_auc, valid_auc))

        if valid_loss < valid_loss_min:  # let's save the best weights to use them in prediction
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'
                  .format(valid_loss_min, valid_loss))

            torch.save(model.state_dict(), saved_model)
            valid_loss_min = valid_loss

    return train_losses, valid_losses



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/', help='The data directory.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--batch', type=int, default=2000, help='Load Batch Size')
    parser.add_argument('--epoches', type=int, default=100, help='Number of Epoches')
    parser.add_argument('--seed', type=int, default=123, help='Random Seed')
    args = vars(parser.parse_args())

    seed_everything(seed=args['seed'])
    path = args['path']

    train_df, eval_df, test_df = load_data(path)
    cat_dim, train_loader, eval_loader, test_loader = get_search_dataloader(train_df, eval_df, test_df, args['batch'])

    # model = FeedForwardNN(cat_dim, no_of_cont=len(numerical_features), lin_layer_sizes=[50, 100],
    #                       output_size=1, emb_dropout=0.04,
    #                       lin_layer_dropouts=[0.001, 0.01])

    model = AdaptedGateCorssNetwork(cat_dim, no_of_numerical=len(numerical_features),
                                    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()
    train_losses, val_losses = train_network(model, train_loader, eval_loader,
                                             loss_fn, optimizer, args['epoches'])