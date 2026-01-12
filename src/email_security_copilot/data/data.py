from __future__ import annotations
import os, re, pandas as pd
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses

class EmailDataset:
    """Loads one directory of emails into a DataFrame (text + metadata)."""

    def __init__(self, dir_path: str, label: str | None = None):
        self.dir_path = dir_path
        self.label = label

    @staticmethod
    def _dec(part):
        b = part.get_payload(decode=True)
        if b is None:
            return (part.get_payload() or "").encode("utf-8", "ignore")
        return b

    @staticmethod
    def _txt(part):
        cs = (part.get_content_charset() or "").strip().strip('"').lower() or "utf-8"
        try:
            return EmailDataset._dec(part).decode(cs, "replace")
        except Exception:
            return EmailDataset._dec(part).decode("latin-1", "replace")

    @staticmethod
    def _html2txt(s: str) -> str:
        s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", s)
        s = re.sub(r"(?is)<br\s*/?>|</p>", "\n", s)
        s = re.sub(r"(?is)<.*?>", "", s)
        return re.sub(r"\s+\n", "\n", s).strip()

    @classmethod
    def _get_body(cls, msg) -> str:
        if msg.is_multipart():
            for p in msg.walk():
                if p.get_content_type() == "text/plain":
                    return cls._txt(p)
            for p in msg.walk():
                if p.get_content_type() == "text/html":
                    return cls._html2txt(cls._txt(p))
            return ""
        else:
            ct = msg.get_content_type()
            if ct == "text/plain": return cls._txt(msg)
            if ct == "text/html":  return cls._html2txt(cls._txt(msg))
            return ""

    def load(self) -> pd.DataFrame:
        rows, skipped = [], 0
        for e in os.scandir(self.dir_path):
            if not e.is_file():
                continue
            try:
                with open(e.path, "rb") as fh:
                    raw = fh.read()
                msg = BytesParser(policy=policy.default).parsebytes(raw)
            except Exception:
                skipped += 1
                continue

            subj   = msg.get("Subject", "") or ""
            sender = msg.get("From", "") or ""
            to     = msg.get("To", "") or ""
            recvd  = msg.get_all("Received", []) or []

            cc_hdrs  = msg.get_all("Cc", []) or []
            cc_addrs = [addr for _name, addr in getaddresses(cc_hdrs)]
            num_cc   = sum(1 for a in cc_addrs if a)
            cc_txt   = ", ".join(a for a in cc_addrs if a)

            try:
                content = self._get_body(msg)
            except Exception:
                skipped += 1
                continue

            if not (subj or content or sender):
                skipped += 1
                continue

            row = {
                "file": e.name,
                "sender": sender,
                "recipient": to,
                "cc": cc_txt,
                "num_cc": num_cc,
                "subject": subj,
                "content": content,
                "num_hops": len(recvd),
            }
            if self.label is not None:
                row["label"] = self.label
            rows.append(row)

        if skipped:
            print(f"[EmailDataset] skipped {skipped} files in {self.dir_path}")
        return pd.DataFrame(rows)

def load_data(spam_dir: str, ham_dir: str) -> pd.DataFrame:
    spam_df = EmailDataset(spam_dir, label="spam").load()
    ham_df  = EmailDataset(ham_dir,  label="ham").load()
    return pd.concat([spam_df, ham_df], ignore_index=True)
