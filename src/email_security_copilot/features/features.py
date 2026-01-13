from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
import re


@dataclass
class EmailFeatureClass:
    """
    Compute and/or select features for emails
    """

    subject_col: str = "subject"
    content_col: str = "content"
    cols: tuple = ("num_cc", "num_hops", "subject_len", "content_len", "has_link")

    URL_RE = re.compile(r"https?://\S+|www\.\S+")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        if "subject_len" not in df.columns and self.subject_col in df.columns:
            df["subject_len"] = df[self.subject_col].fillna("").str.len()

        if "content_len" not in df.columns and self.content_col in df.columns:
            df["content_len"] = df[self.content_col].fillna("").str.len()

        if "has_link" not in df.columns:
            subj = df[self.subject_col].fillna("").astype(str)
            cont = df[self.content_col].fillna("").astype(str)
            df["has_link"] = (subj + " " + cont).str.contains(self.URL_RE).astype(int)

        if "num_cc" not in df.columns and "cc" in df.columns:
            df["num_cc"] = df["cc"].fillna("").apply(
                lambda x: len([a.strip() for a in x.split(", ") if a.strip()])
            )

        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        return df.copy()
