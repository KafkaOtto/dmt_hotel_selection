from typing import Tuple, Dict, List, Generator

import torch
from torch.utils.data.dataloader import DataLoader

import rank_ndcg_batch.models.losses as losses
from rank_ndcg_batch.config import Config
from rank_ndcg_batch.data.data_loading import HotelSearchDataset
from rank_ndcg_batch.model import LTRModel
from rank_ndcg_batch.utils.model_utils import get_torch_device


def rank_slates(datasets: Dict[str, HotelSearchDataset], model: LTRModel, config: Config) \
        -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Ranks given datasets according to a given model

    :param datasets: dictionary of role -> dataset that will be ranked
    :param model: a model to use for scoring documents
    :param config: config for DataLoaders
    :return: dictionary of role -> ranked dataset
        every dataset is a Tuple of torch.Tensor - storing X and y in the descending order of the scores.
    """

    dataloaders = {role: __create_data_loader(ds, config) for role, ds in datasets.items()}

    ranked_slates = {role: __rank_slates(dl, model) for role, dl in dataloaders.items()}

    return ranked_slates


def __create_data_loader(ds: HotelSearchDataset, config: Config) -> DataLoader:
    return DataLoader(ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)


def __rank_slates(dataloader: DataLoader, model: LTRModel) -> Tuple[torch.Tensor, torch.Tensor]:
    # reranked_X_num = []
    # reranked_X_cat = []
    reranked_q = []
    model.eval()
    device = get_torch_device()
    with torch.no_grad():
        for xnumb, xcatb, q, yb, _ in dataloader:
            X_num = xnumb.to(device=device)
            X_cat = xcatb.to(device=device)
            y_true = yb.to(device=device)
            q_torch = q.to(device=device)

            input_indices = torch.ones_like(y_true).type(torch.long)
            mask = (y_true == losses.PADDED_Y_VALUE)
            scores = model.score(X_num, X_cat, mask, input_indices)

            scores[mask] = float('-inf')

            _, indices = scores.sort(descending=True, dim=-1)
            indices_q = torch.unsqueeze(indices, -1).repeat_interleave(q_torch.shape[-1], -1)
            reranked_batch = torch.gather(q_torch, dim=1, index=indices_q).cpu()
            reranked_y = torch.gather(y_true, dim=1, index=indices).cpu()
            reranked_y_mask = (reranked_y == losses.PADDED_Y_VALUE)
            reranked_batch_filter = reranked_batch[~reranked_y_mask]
            reranked_q.append(reranked_batch_filter)

    # combined_X_num = torch.cat(reranked_X_num)
    # combined_X_cat = torch.cat(reranked_X_cat)
    reranked_ids = torch.cat(reranked_q)
    return reranked_ids
