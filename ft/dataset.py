import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rema_utils import detect_column_type, sentence_transformer_map, clean_element

class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        model_type="roberta",
        serialization="header_values_prefix",
        augmentation="exact_semantic",
    ):
        self.serialization = serialization
        self.tokenizer = AutoTokenizer.from_pretrained(
            sentence_transformer_map[model_type]
        )
        self.labels = []
        self.items = self._initialize_items(data, augmentation)

    def _initialize_items(self, data, augmentation):
        items = []
        class_id = 0

        for _, categories in data.items():
            for aug_type, columns in categories.items():
                if aug_type in augmentation or aug_type == "original":
                    for column_name, values in columns.items():
                        processed_column_name = (
                            column_name.rsplit("_", 1)[0]
                            if aug_type == "exact"
                            else column_name
                        )
                        # processed_column_name = clean_element(processed_column_name)
                        values = [clean_element(str(value)) for value in values]
                        items.append((processed_column_name, values, class_id))
                        self.labels.append(class_id)
            class_id += 1

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        key, values, class_id = self.items[idx]
        text = self._serialize(key, values)
        return text, class_id

    def _serialize(self, header, values):
        if values:
            col = pd.DataFrame({header: values})[header]
            data_type = detect_column_type(pd.DataFrame({header: values})[header])
        else:
            data_type = "unknown"
        serialization = {
            "header_values_default": f"{self.tokenizer.cls_token}{header}{self.tokenizer.sep_token}{data_type}{self.tokenizer.sep_token}{','.join(map(str, values))}",
            "header_values_prefix": f"{self.tokenizer.cls_token}header:{header}{self.tokenizer.sep_token}datatype:{data_type}{self.tokenizer.sep_token}values:{', '.join(map(str, values))}",
        }
        return serialization[self.serialization]