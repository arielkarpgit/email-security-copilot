from __future__ import annotations

import argparse
import csv
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable, Optional

def find_dir_case_insensitive(root: Path, name: str) -> Optional[Path]:
    # finds folder by name regardless of casing
    name_lower = name.lower()
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == name_lower:
            return p
    return None

def iter_email_files(folder: Path) -> Iterable[Path]:
    # SpamAssassin files are typically plain files (no extension)
    for p in folder.rglob("*"):
        if p.is_file():
            yield p

def extract_text_from_email(raw_bytes: bytes) -> tuple[str, str]:
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    subject = msg.get("subject", "") or ""
    subject = str(subject)

    # Prefer text/plain parts; fallback to best body
    body = ""
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = part.get_content_disposition()
            if disp == "attachment":
                continue
            if ctype == "text/plain":
                try:
                    parts.append(part.get_content())
                except Exception:
                    try:
                        parts.append(part.get_payload(decode=True).decode(errors="replace"))
                    except Exception:
                        pass
        body = "\n".join([p for p in parts if p]).strip()
        if not body:
            # fallback: grab any text/* part
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype.startswith("text/"):
                    try:
                        body = (part.get_content() or "").strip()
                        if body:
                            break
                    except Exception:
                        continue
    else:
        try:
            body = (msg.get_content() or "").strip()
        except Exception:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, (bytes, bytearray)):
                body = payload.decode(errors="replace").strip()
            else:
                body = str(payload).strip()

    return subject, body

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to data/raw/spamassassin")
    ap.add_argument("--out", required=True, help="Output CSV path (e.g. data/processed/spamassassin.csv)")
    ap.add_argument("--max_per_class", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spam_dir = find_dir_case_insensitive(root, "SPAM")
    ham_dir = find_dir_case_insensitive(root, "EASY_HAM")

    if spam_dir is None:
        raise FileNotFoundError(f"Could not find SPAM under: {root}")
    if ham_dir is None:
        raise FileNotFoundError(f"Could not find EASY_HAM under: {root}")

    rows = []
    for folder, label in [(ham_dir, 0), (spam_dir, 1)]:
        n = 0
        for f in iter_email_files(folder):
            if args.max_per_class and n >= args.max_per_class:
                break
            try:
                raw = f.read_bytes()
                subject, body = extract_text_from_email(raw)
                # Skip empties
                if not (subject.strip() or body.strip()):
                    continue
                rows.append({"subject": subject, "body": body, "label": label})
                n += 1
            except Exception:
                continue

    # Write CSV
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["subject", "body", "label"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(f"Ham:  {sum(r['label']==0 for r in rows)}")
    print(f"Spam: {sum(r['label']==1 for r in rows)}")

if __name__ == "__main__":
    main()
