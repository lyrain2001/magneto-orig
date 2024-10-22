import argparse
import numpy as np
import random
import torch

from dataset import PretrainTableDataset
from pretrain import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gdc")
    parser.add_argument("--logdir", type=str, default="../../models/")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default="roberta")
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default="semantic,exact")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--sample_meth", type=str, default="random")
    parser.add_argument("--top_k", type=int, default=50)

    parser.add_argument(
        "--dataset", default="chembl", help="Name of the dataset (without usecase)",
    )
    parser.add_argument(
        "--model_type",
        default="roberta-zs",
        help="Type of model (roberta-zs, roberta-ft, mpnet-zs, mpnet-ft, arctic-zs, arctic-ft)",
    )
    parser.add_argument(
        "--serialization",
        default="header_values_prefix",
        help="Column serialization method (header, header_values_default, header_values_prefix, header_values_repeat)",
    )

    args = parser.parse_args()

    seed = args.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    path = "datasets/%s" % args.task
    trainset = PretrainTableDataset.from_args(path, args)
    train(trainset, args)
