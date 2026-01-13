import pandas as pd
from pathlib import Path
import argparse
from email_security_copilot.data.data import load_data
from email_security_copilot.features.features import EmailFeatureClass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_root",
        default="data/raw/spamassassin",
        help="Path containing spamassassin folders (spam, easy_ham)",
    )
    ap.add_argument(
        "--out",
        default="data/processed/spamassassin.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    spam_dir = raw_root / "spam"
    ham_dir = raw_root / "easy_ham"

    if not spam_dir.exists():
        raise FileNotFoundError(f"Missing spam dir: {spam_dir}")
    if not ham_dir.exists():
        raise FileNotFoundError(f"Missing easy_ham dir: {ham_dir}")

    df = load_data(spam_dir=str(spam_dir), ham_dir=str(ham_dir))
    
    features_df = EmailFeatureClass()(df)
    # sanity: keep only expected columns (and order them nicely)
    keep = ['sender', 'recipient', 'num_cc', 'subject', 'content', 
        'subject_len', 'content_len', 'has_link', 'num_hops', 'label']
    missing = [c for c in keep if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"load_data() did not return expected columns: missing={missing}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df = features_df[keep].copy()

    # minimal cleanup
    raw_df["subject"] = raw_df["subject"].fillna("").astype(str)
    raw_df["content"] = raw_df["content"].fillna("").astype(str)

    raw_df.to_csv(out_path, index=False)
    print(f"Wrote {len(raw_df):,} rows -> {out_path}")
    print(raw_df["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()