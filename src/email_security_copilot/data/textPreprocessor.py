from __future__ import annotations
import re, numpy as np, pandas as pd
import html
from dataclasses import dataclass
import sys
import unicodedata


@dataclass
class TextPreprocessor:
    """Text cleaning and normalization based on config options."""

    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    HANDLE_RE = re.compile(r"@\w+")
    HASHTAG_RE = re.compile(r"#\w+")
    HTML_TAG_RE = re.compile(r"<[^>]+>")
    DIGIT_RE = re.compile(r"\d+")
    PUNCT_TABLE = dict.fromkeys(
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self._nlp = None  # lazy-load spaCy if lemmatize=true

    def _lemmatize(self, text: str) -> str:
        if self._nlp is None:
            import spacy  # optional dep

            lang = self.cfg.preprocessing.language
            model = {"en": "en_core_web_sm"}.get(lang, "en_core_web_sm")
            try:
                self._nlp = spacy.load(model)
            except OSError:
                # If model not installed, skip lemmatization gracefully
                return text
        doc = self._nlp(text)
        return " ".join(tok.lemma_ for tok in doc)

    def __call__(self, text: str) -> str:
        if not self.cfg.preprocessing.enable:
            return text

        # Unescape HTML entities and optionally remove tags
        text = html.unescape(text)
        if self.cfg.preprocessing.remove_html:
            text = re.sub(self.HTML_TAG_RE, " ", text)

        if self.cfg.preprocessing.remove_urls:
            text = re.sub(self.URL_RE, " ", text)

        if self.cfg.preprocessing.remove_emails:
            text = re.sub(self.EMAIL_RE, " ", text)

        if self.cfg.preprocessing.remove_user_handles:
            text = re.sub(self.HANDLE_RE, " ", text)

        if self.cfg.preprocessing.remove_hashtags:
            text = re.sub(self.HASHTAG_RE, " ", text)

        if self.cfg.preprocessing.lowercase:
            text = text.lower()

        if self.cfg.preprocessing.remove_digits:
            text = re.sub(self.DIGIT_RE, " ", text)

        if self.cfg.preprocessing.remove_punctuation:
            text = text.translate(self.PUNCT_TABLE)

        if self.cfg.preprocessing.collapse_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        if self.cfg.preprocessing.lemmatize:
            text = self._lemmatize(text)

        return text
