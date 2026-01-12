from __future__ import annotations

from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import ProjectConfig
from src.data.data import load_data
import pandas as pd


class SpamFrameDataset(Dataset):
    """
    Prepares a PyTorch Dataset from a DataFrame containing spam/ham emails.
    """

    def __init__(self,  df: pd.DataFrame, cfg: ProjectConfig):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.pretrained_model_name
        )
        self.label_map = {"ham": 0, "spam": 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        text = row[self.cfg.model.text_col]
        label_str = row[self.data_cfg.label_col]
        label = self.label_map[label_str]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.data_cfg.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item
