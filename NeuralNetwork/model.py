import torch.nn as nn
import torch
import torch.nn.functional as F

class AdaptedGateCorssNetwork(nn.Module):
    def __init__(self, emb_dims, no_of_numerical=None, cn_layers=3, dropout_rate=0):
        super(AdaptedGateCorssNetwork, self).__init__()
        self.embedding = EmbeddingLayer(emb_dims, no_of_numerical, dropout_rate)
        self.no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_numerical = 0
        if no_of_numerical is not None:
            self.no_of_numerical = no_of_numerical
        self.emb_out = self.no_of_embs + self.no_of_numerical
        self.cross_net = GateCorssLayer(self.emb_out, cn_layers)
        self.pred_layer = nn.Linear(self.emb_out, 1)

    def forward(self, x_num, x_cat):
        x_embed = self.embedding(x_num, x_cat)
        cross_cn = self.cross_net(x_embed)
        pred_y = self.pred_layer(cross_cn)
        return pred_y


class EmbeddingLayer(nn.Module):
    def __init__(self, emb_dims, no_of_numerical=None, emb_dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                               for x, y in emb_dims])
        self.no_of_embs = sum([y for x, y in emb_dims])
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.no_of_numerical = 0
        if no_of_numerical is not None:
            self.no_of_numerical = no_of_numerical
            self.bn_numerical = nn.BatchNorm1d(no_of_numerical)

    def forward(self, x_numerical, x_cat):
        # x = []
        if self.no_of_embs != 0:
            # for i, emb_layer in enumerate(self.emb_layers):
            #     x.append(emb_layer(x_cat[:, i]))  # Append tensors to the list
            x = [emb_layer(x_cat[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]

            x = torch.cat(x, 1)
            x = self.emb_dropout(x)

        if self.no_of_numerical != 0:
            x_cont = self.bn_numerical(x_numerical)

            if self.no_of_embs != 0:
                x = torch.cat([x, x_cont], 1)
            else:
                x = x_cont
        return x

class GateCorssLayer(nn.Module):
    #  The core structure： gated corss layer.
    def __init__(self, input_dim, cn_layers=3):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.wg = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])

        self.b = nn.ParameterList([nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

        for i in range(cn_layers):
            nn.init.uniform_(self.b[i].data)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x) # Feature Crossing
            xg = self.activation(self.wg[i](x)) # Information Gate
            x = x0 * (xw + self.b[i]) * xg + x
        return x


class FeedForwardNN(nn.Module):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes,
                 output_size, emb_dropout, lin_layer_dropouts):

        """
        Parameters
        ----------
    
        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.
    
        no_of_cont: Integer
          The number of continuous features in the data.
    
        lin_layer_sizes: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.
    
        output_size: Integer
          The size of the final output.
    
        emb_dropout: Float
          The dropout to be used after the embedding layers.
    
        lin_layer_dropouts: List of floats
          The dropouts to be used after each linear layer.
        """

        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                    lin_layer_sizes[0])

        self.lin_layers = \
            nn.ModuleList([first_lin_layer] + \
                          [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                           for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1],
                                      output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                        for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                            for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):

        try:
            x = []
            if self.no_of_embs != 0:
                for i, emb_layer in enumerate(self.emb_layers):
                    try:
                        x.append(emb_layer(cat_data[:, i]))
                    except Exception as e:
                        print(f"Exception occurred at index {i} in embedding layer.")
                        print("Unique values of cat_data at index", i, ":", torch.unique(cat_data[:, i]))
                        print("cat_data at index", i, ":", cat_data[:, i])
                        raise e

                x = torch.cat(x, 1)
                x = self.emb_dropout_layer(x)

            if self.no_of_cont != 0:
                normalized_cont_data = self.first_bn_layer(cont_data)

                if self.no_of_embs != 0:
                    x = torch.cat([x, normalized_cont_data], 1)
                else:
                    x = normalized_cont_data

            for lin_layer, dropout_layer, bn_layer in \
                    zip(self.lin_layers, self.droput_layers, self.bn_layers):
                x = F.relu(lin_layer(x))
                x = bn_layer(x)
                x = dropout_layer(x)

            x = self.output_layer(x)

            return x

        except Exception as e:
            print("Exception occurred in forward method.")
            raise e
