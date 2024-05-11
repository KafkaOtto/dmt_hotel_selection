from urllib.parse import urlparse

import rank_ndcg_batch.models.losses as losses
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim
from rank_ndcg_batch.utils.model_utils import get_torch_device, load_state_dict_from_file
from rank_ndcg_batch.data.data_loading import load_search_dataset_role, create_data_loaders
from rank_ndcg_batch.utils.common import get_n_numerical, emb_dims
from model import make_model
from rank_ndcg_batch.utils.logging_r import  init_logger
from rank_ndcg_batch.training.training_utils import fit
from config import Config
from torch.utils.data import DataLoader
from rank_ndcg_batch.utils.inference_utils import __rank_slates

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--path", help="Base output path for all experiments",
                        default="/Users/zhinuanguo/vu_projects/period5/data_mining_technique/assignments/assignment2/code/ExpediaPersonalizedSortCompetition/data/")
    return parser.parse_args()


def run():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    path = args.path

    logger = init_logger(path)
    logger.info(f"created paths container {path}")

    json_path = path + "run.json"
    # read config
    config = Config.from_json(json_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    # train_ds, val_ds
    # max 38
    test_ds = load_search_dataset_role(
        role="test",
        input_path=config.data.path,
        slate_length=config.data.slate_length
    )

    test_dl = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(emb_dims=emb_dims, no_of_numerical=get_n_numerical(), **asdict(config.model, recurse=False))
    input_model_path = path + "model.pkl"
    model.load_state_dict(load_state_dict_from_file(input_model_path, dev))
    model.to(dev)
    logger.info(f"loaded model weights from {input_model_path}")

    reranked_ids= __rank_slates(test_dl, model)
    logger.info("reranked_ids_df shape: {}".format(reranked_ids.shape))
    reranked_ids_df = pd.DataFrame(reranked_ids.numpy(), columns=['srch_id', 'prop_id'])
    reranked_ids_df.to_csv(path + 'reranked_ids.csv', index=False)


if __name__ == "__main__":
    run()