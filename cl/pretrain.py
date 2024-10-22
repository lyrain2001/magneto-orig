import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import os
import sys
import json

from model import BarlowTwinsSimCLR
from dataset import PretrainTableDataset
from loss import batch_all_triplet_loss


from tqdm import tqdm
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List


def train_step(train_iter, model, optimizer, scheduler, scaler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (BarlowTwinsSimCLR): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        scaler (GradScaler): gradient scaler for fp16 training
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        x_ori, x_aug, cls_indices = batch
        optimizer.zero_grad()

        if hp.fp16:
            with torch.cuda.amp.autocast():
                loss = model(x_ori, x_aug, cls_indices, mode="simclr")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss = model(x_ori, x_aug, cls_indices, mode="simclr")
            loss.backward()
            optimizer.step()

        scheduler.step()
        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


# ----------------------------------- train -----------------------------------
def train(trainset, hp):
    """Train and evaluate the model

    Args:
        trainset (PretrainTableDataset): the training set
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)
    Returns:
        The pre-trained table model
    """
    print("Start training")
    padder = trainset.pad
    train_iter = data.DataLoader(
        dataset=trainset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=padder,
    )

    # initialize model, optimizer, and LR scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    print("num_steps: ", num_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )
    best_precision = 0.0
    for epoch in range(1, hp.n_epochs + 1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, scaler, hp)
        print("epoch %d: " % epoch + "training done")

        if hp.task in ["arpa"]:
            # Train column matching models using the learned representations
            # store_embeddings = True if epoch == hp.n_epochs else False
            precision = evaluate_arpa_matching(model, trainset, best_precision, hp)

            if hp.save_model and precision > best_precision:
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                if hp.gpt:
                    ckpt_path = os.path.join(
                        hp.logdir,
                        hp.task,
                        "model_" + str(hp.top_k) + "_" + str(hp.run_id) + ".pt",
                    )
                else:
                    ckpt_path = os.path.join(
                        hp.logdir,
                        hp.task,
                        "model_" + str(hp.top_k) + "_starmie_" + str(hp.run_id) + ".pt",
                    )

                ckpt = {"model": model.state_dict(), "hp": hp}
                torch.save(ckpt, ckpt_path)

            best_precision = max(best_precision, precision)

            print("epoch %d: " % epoch + "precision=%f" % precision)
            print("Best precision so far: ", best_precision)


def inference_on_tables(
    tables: List[pd.DataFrame],
    model: BarlowTwinsSimCLR,
    unlabeled: PretrainTableDataset,
    batch_size=128,
    total=None,
):
    """Extract column vectors from a table.

    Args:
        tables (List of DataFrame): the list of tables
        model (BarlowTwinsSimCLR): the model to be evaluated
        unlabeled (PretrainTableDataset): the unlabeled dataset
        batch_size (optional): batch size for model inference

    Returns:
        List of np.array: the column vectors
    """
    total = total if total is not None else len(tables)
    batch = []
    results = []
    for tid, table in tqdm(enumerate(tables), total=total):
        x, _ = unlabeled._tokenize(table)

        batch.append((x, x, []))
        if tid == total - 1 or len(batch) == batch_size:
            # model inference
            with torch.no_grad():
                x, _, _ = unlabeled.pad(batch)
                # all column vectors in the batch
                column_vectors = model.inference(x)
                ptr = 0
                for xi in x:
                    current = []
                    for token_id in xi:
                        if token_id == unlabeled.tokenizer.cls_token_id:
                            current.append(column_vectors[ptr].cpu().numpy())
                            ptr += 1
                    results.append(current)

            batch.clear()

    return results


def load_checkpoint(ckpt):
    """Load a model from a checkpoint.
        ** If you would like to run your own benchmark, update the ds_path here
    Args:
        ckpt (str): the model checkpoint.

    Returns:
        BarlowTwinsSimCLR: the pre-trained model
        PretrainDataset: the dataset for pre-training the model
    """
    hp = ckpt["hp"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)
    model = model.to(device)
    model.load_state_dict(ckpt["model"])

    # dataset paths, depending on benchmark for the current task
    ds_path = "data/santos/datalake"
    if hp.task == "santosLarge":
        # Change the data paths to where the benchmarks are stored
        ds_path = "data/santos-benchmark/real-benchmark/datalake"
    elif hp.task == "tus":
        ds_path = "data/table-union-search-benchmark/small/benchmark"
    elif hp.task == "tusLarge":
        ds_path = "data/table-union-search-benchmark/large/benchmark"
    elif hp.task == "wdc":
        ds_path = "data/wdc/0"
    # ----------------------------------------------------------------
    elif hp.task == "arpa":
        ds_path = "data/gdc_train"
    dataset = PretrainTableDataset.from_hp(ds_path, hp)

    return model, dataset
