import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn as nn
from attr import asdict

from rank_ndcg_batch.models.transformer import make_transformer
from rank_ndcg_batch.utils.python_utils import instantiate_class
from rank_ndcg_batch.data.data_loading import PADDED_CAT_VALUE
from rank_ndcg_batch.utils.common import get_n_features

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
            x_cont = x_numerical

            if self.no_of_embs != 0:
                x = torch.cat([x, x_cont], 1)
            else:
                x = x_cont
        return x


def first_arg_id(x, *y):
    return x


class FCModel(nn.Module):
    """
    This class represents a fully connected neural network model with given layer sizes and activation function.
    """
    def __init__(self, sizes, input_norm, activation, dropout, emb_dims, n_out=None):
        """
        :param sizes: list of layer sizes (excluding the input layer size which is given by n_features parameter)
        :param input_norm: flag indicating whether to perform layer normalization on the input
        :param activation: name of the PyTorch activation function, e.g. Sigmoid or Tanh
        :param dropout: dropout probability
        :param n_features: number of input features
        """
        super(FCModel, self).__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y, padding_idx=PADDED_CAT_VALUE)
                                         for x, y in emb_dims])
        self.emb_out = n_out
        sizes.insert(0, self.emb_out)
        layers = [nn.Linear(size_in, size_out) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.input_norm = nn.LayerNorm(self.emb_out) if input_norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else instantiate_class(
            "torch.nn.modules.activation", activation)
        self.dropout = nn.Dropout(dropout or 0.0)
        self.output_size = sizes[-1]

        self.layers = nn.ModuleList(layers)

    def forward(self, x_numerical, x_cat):
        """
        Forward pass through the FCModel.
        :param x: input of shape [batch_size, slate_length, self.layers[0].in_features]
        :return: output of shape [batch_size, slate_length, self.output_size]
        """

        # batch_size, item_size, _ = x_cat.size()
        # emb_x_cat = []
        # for i in range(item_size):
        #     x_item = x_cat[:, i]
        #     print(f"x_item: {x_item.shape}")
        #     emb_x_cat.append(torch.cat([emb_layer(x_item) for emb_layer in self.emb_layers], dim=-1))
        # emb_x_cat = torch.stack(emb_x_cat, dim=1)  # Shape: (batch_size, item_size, emb_out)
        x = torch.cat([x_numerical, x_cat], dim=-1)
        # print(f"x shape: {x.shape}")
        x = self.input_norm(x)
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return x


class LTRModel(nn.Module):
    """
    This class represents a full neural Learning to Rank model with a given encoder model.
    """
    def __init__(self, input_layer, encoder, output_layer):
        """
        :param input_layer: the input block (e.g. FCModel)
        :param encoder: the encoding block (e.g. transformer.Encoder)
        :param output_layer: the output block (e.g. OutputLayer)
        """
        super(LTRModel, self).__init__()
        self.input_layer = input_layer if input_layer else nn.Identity()
        self.encoder = encoder if encoder else first_arg_id
        self.output_layer = output_layer

    def prepare_for_output(self, x_num, x_cat, mask, indices):
        """
        Forward pass through the input layer and encoder.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: encoder output of shape [batch_size, slate_length, encoder_output_dim]
        """
        return self.encoder(self.input_layer(x_num, x_cat), mask, indices)

    def forward(self, x_num, x_cat, mask, indices):
        """
        Forward pass through the whole LTRModel.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: model output of shape [batch_size, slate_length, output_dim]
        """
        return self.output_layer(self.prepare_for_output(x_num, x_cat, mask, indices))

    def score(self, x_num, x_cat, mask, indices):
        """
        Forward pass through the whole LTRModel and item scoring.

        Used when evaluating listwise metrics in the training loop.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        return self.output_layer.score(self.prepare_for_output(x_num, x_cat, mask, indices))


class OutputLayer(nn.Module):
    """
    This class represents an output block reducing the output dimensionality to d_output.
    """
    def __init__(self, d_model, d_output, output_activation=None):
        """
        :param d_model: dimensionality of the output layer input
        :param d_output: dimensionality of the output layer output
        :param output_activation: name of the PyTorch activation function used before scoring, e.g. Sigmoid or Tanh
        """
        super(OutputLayer, self).__init__()
        self.activation = nn.Identity() if output_activation is None else instantiate_class(
            "torch.nn.modules.activation", output_activation)
        self.d_output = d_output
        self.w_1 = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Forward pass through the OutputLayer.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length, self.d_output]
        """
        return self.activation(self.w_1(x).squeeze(dim=2))

    def score(self, x):
        """
        Forward pass through the OutputLayer and item scoring by summing the individual outputs if d_output > 1.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length]
        """
        if self.d_output > 1:
            return self.forward(x).sum(-1)
        else:
            return self.forward(x)


def make_model(fc_model, transformer, post_model, emb_dims, no_of_numerical):
    """
    Helper function for instantiating LTRModel.
    :param fc_model: FCModel used as input block
    :param transformer: transformer Encoder used as encoder block
    :param post_model: parameters dict for OutputModel output block (excluding d_model)
    :param n_features: number of input features
    :return: LTR model instance
    """
    # no_of_embs = sum([y for x, y in emb_dims])
    # n_out = no_of_embs + no_of_numerical
    n_out = get_n_features()
    if fc_model:
        fc_model = FCModel(**fc_model, emb_dims=emb_dims, n_out=n_out)  # type: ignore
    d_model = n_out if not fc_model else fc_model.output_size
    if transformer:
        transformer = make_transformer(n_features=d_model, **asdict(transformer, recurse=False))  # type: ignore
    model = LTRModel(fc_model, transformer, OutputLayer(d_model, **post_model))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model