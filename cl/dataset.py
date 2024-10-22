import torch
import random
import pandas as pd
import os
import sys
import json

from torch.utils import data
from transformers import AutoTokenizer
from typing import List

from augment import augment
from preprocessor import computeTfIdf, tfidfRowSample, preprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import infer_column_dtype


lm_map = {
    "roberta": "roberta-base",
    "mpnet": "microsoft/mpnet-base",
}


class PretrainTableDataset(data.Dataset):
    """Table dataset for pre-training"""

    def __init__(
        self, hp,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_map[hp.lm])
        self.max_len = hp.max_len
        self.unique_columns, self.matches = self._load_json_file(hp)
        self.augment_op = hp.augment_op
        self.tokenizer_cache = {}

    def _load_json_file(self, hp):
        unique_columns_path = f"train_data/{hp.dataset}_unique_columns.json"
        if not os.path.exists(unique_columns_path):
            print(f"File {unique_columns_path} does not exist")
            exit()
        with open(unique_columns_path, "r") as file:
            unique_columns = json.load(file)

        matches_path = f"train_data/{hp.dataset}_synthetic_matches.json"
        if not os.path.exists(matches_path):
            print(f"File {matches_path} does not exist")
            exit()
        with open(matches_path, "r") as file:
            matches = json.load(file)

        return unique_columns, matches

    def _tokenize(self, header, values, tokens):
        if self.serialization == "header":
            return header

        data_type = infer_column_dtype(values)

        if self.serialization == "header_values_default":
            text = (
                self._tokenizer.cls_token
                + header
                + self._tokenizer.sep_token
                + data_type
                + self._tokenizer.sep_token
                + self._tokenizer.sep_token.join(tokens)
            )

        elif self.serialization == "header_values_prefix":
            text = (
                self._tokenizer.cls_token
                + "header:"
                + header
                + self._tokenizer.sep_token
                + " datatype:"
                + data_type
                + self._tokenizer.sep_token
                + " values:"
                + ", ".join(tokens)
            )

        elif self.serialization == "header_values_repeat":
            text = (
                self._tokenizer.cls_token
                + self._tokenizer.sep_token.join([header] * 5)
                + self._tokenizer.sep_token
                + data_type
                + self._tokenizer.sep_token
                + self._tokenizer.sep_token.join(tokens)
            )

        return text

    # --------------------------- old code --------------------------------

    def _tokenize(self, table: pd.DataFrame) -> List[int]:
        """Tokenize a DataFrame table

        Args:
            table (DataFrame): the input table

        Returns:
            List of int: list of token ID's with special tokens inserted
            Dictionary: a map from column names to special tokens
        """
        res = []
        max_tokens = self.max_len * 2
        budget = max(1, self.max_len - 1)
        tfidfDict = (
            computeTfIdf(table) if "tfidf" in self.sample_meth else None
        )  # from preprocessor.py

        # a map from column names to special token indices
        column_mp = {}

        if self.gpt:
            for column in table.columns:
                tokens = preprocess(
                    table[column], tfidfDict, max_tokens, self.sample_meth
                )  # from preprocessor.py
                col_text = (
                    self.tokenizer.cls_token
                    + column
                    + self.tokenizer.sep_token
                    + self.tokenizer.sep_token.join(tokens[:max_tokens])
                )
                # col_text = self.tokenizer.cls_token + " " + column +  self.tokenizer.sep_token  + \
                # self.tokenizer.sep_token.join(tokens[:max_tokens]) + " "
                # print(col_text)
                # exit()
                column_mp[column] = len(res)
                res += self.tokenizer.encode(
                    text=col_text,
                    max_length=budget,
                    add_special_tokens=False,
                    truncation=True,
                )

        # column-ordered preprocessing
        elif self.table_order == "column":
            if "row" in self.sample_meth:
                table = tfidfRowSample(table, tfidfDict, max_tokens)
            for column in table.columns:
                tokens = preprocess(
                    table[column], tfidfDict, max_tokens, self.sample_meth
                )  # from preprocessor.py
                col_text = (
                    self.tokenizer.cls_token + " " + " ".join(tokens[:max_tokens]) + " "
                )
                column_mp[column] = len(res)
                res += self.tokenizer.encode(
                    text=col_text,
                    max_length=budget,
                    add_special_tokens=False,
                    truncation=True,
                )
        else:
            # row-ordered preprocessing
            reached_max_len = False
            for rid in range(len(table)):
                row = table.iloc[rid : rid + 1]
                for column in table.columns:
                    tokens = preprocess(
                        row[column], tfidfDict, max_tokens, self.sample_meth
                    )  # from preprocessor.py
                    if rid == 0:
                        column_mp[column] = len(res)
                        col_text = (
                            self.tokenizer.cls_token
                            + " "
                            + " ".join(tokens[:max_tokens])
                            + " "
                        )
                    else:
                        col_text = (
                            self.tokenizer.pad_token
                            + " "
                            + " ".join(tokens[:max_tokens])
                            + " "
                        )

                    tokenized = self.tokenizer.encode(
                        text=col_text,
                        max_length=budget,
                        add_special_tokens=False,
                        truncation=True,
                    )

                    if len(tokenized) + len(res) <= self.max_len:
                        res += tokenized
                    else:
                        reached_max_len = True
                        break

                if reached_max_len:
                    break

        self.log_cnt += 1
        if self.log_cnt % 5000 == 0:
            print(self.tokenizer.decode(res))

        return res, column_mp

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.tables)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the first view
            List of int: token ID's of the second view
        """
        # print("idx: ", idx)
        # --------------------------------------------------------------
        if self.gpt and self.single_column:
            table_ori = self.tables[idx]
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]
            train_map = pd.read_csv("data/train.csv")
            gdc_table_synthetic = pd.read_csv("data/tables/gdc_table_synthetic.csv")
            row = train_map[train_map["l_column_id"] == col]
            table_aug = gdc_table_synthetic[row["r_column_id"]]
            table_aug = table_aug.sample(frac=1)
        # --------------------------------------------------------------
        else:
            # table_ori = self._read_table(idx)
            table_ori = self.tables[idx]
            # single-column mode: only keep one random column
            if self.single_column:
                col = random.choice(table_ori.columns)
                table_ori = table_ori[[col]]

            # apply the augmentation operator
            if "," in self.augment_op:
                op1, op2 = self.augment_op.split(",")
                table_tmp = table_ori
                table_ori = augment(table_tmp, op1)
                table_aug = augment(table_tmp, op2)
            else:
                table_aug = augment(table_ori, self.augment_op)

        # print("table_ori: ", table_ori)
        # print("table_aug: ", table_aug)

        # convert table into string
        x_ori, mp_ori = self._tokenize(table_ori)
        x_aug, mp_aug = self._tokenize(table_aug)

        # print("x_ori: ", x_ori)
        # print("x_aug: ", x_aug)
        # print("mp_ori: ", mp_ori)
        # print("mp_aug: ", mp_aug)

        # make sure that x_ori and x_aug has the same number of cls tokens
        # x_ori_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_ori])
        # x_aug_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_aug])
        # assert x_ori_cnt == x_aug_cnt

        # insertsect the two mappings
        cls_indices = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col]))
        # TODO: debug
        if self.gpt:
            cls_indices = [(0, 0)]

        return x_ori, x_aug, cls_indices

    def pad(self, batch):
        """Merge a list of dataset items into a training batch

        Args:
            batch (list of tuple): a list of sequences

        Returns:
            LongTensor: x_ori of shape (batch_size, seq_len)
            LongTensor: x_aug of shape (batch_size, seq_len)
            tuple of List: the cls indices
        """
        x_ori, x_aug, cls_indices = zip(*batch)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [
            xi + [self.tokenizer.pad_token_id] * (maxlen - len(xi)) for xi in x_ori
        ]
        x_aug_new = [
            xi + [self.tokenizer.pad_token_id] * (maxlen - len(xi)) for xi in x_aug
        ]

        # decompose the column alignment
        cls_ori = []
        cls_aug = []
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)

        return (
            torch.LongTensor(x_ori_new),
            torch.LongTensor(x_aug_new),
            (cls_ori, cls_aug),
        )
