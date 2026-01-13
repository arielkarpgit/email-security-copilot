# src/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any
import json

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class DataCfg:
    spam_dir: str = str(ROOT / "data" / "spam")
    ham_dir: str = str(ROOT / "data" / "easy_ham")
    
    text_col: str = "content"
    label_col: str = "label"
    
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    stratify: bool = True
    
    max_samples: int | None = None
    max_length: int = 256

    def validate(self) -> None:
        if not (0 < self.test_size < 1):
            raise ValueError("data.test_size must be in (0, 1).")
        if not (0 < self.val_size < 1):
            raise ValueError("data.val_size must be in (0, 1).")
        if self.test_size + self.val_size >= 1:
            raise ValueError("data.test_size + data.val_size must be < 1.")


@dataclass
class PreprocessingCfg:
    enable: bool = True
    language: str = "en" # spaCy language code (used only if lemmatize=True)
    
    remove_html: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_user_handles: bool = False
    remove_hashtags: bool = False
    
    lowercase: bool = False
    remove_digits: bool = False
    remove_punctuation: bool = False
    collapse_whitespace: bool = True
    
    lemmatize: bool = False # requires spaCy model

    def validate(self) -> None:
        pass

@dataclass
class ModelConfig:
    # Switch between encoders:
    #   "distilroberta-base"  - fast strong baseline
    #   "xlm-roberta-base"    - multilingual
    pretrained_model_name: str = "distilroberta-base"
    num_labels: int = 2
    text_col: str = "text_raw_clean"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    def validate(self) -> None:
        if self.num_labels <= 0:
            raise ValueError("model.num_labels must be > 0.")
        if self.use_lora and self.lora_r <= 0:
            raise ValueError("model.lora_r must be > 0 when use_lora=True.")


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    device: str = "cuda"
    base_output_dir: Path = ROOT / "notebooks" / "artifacts" / "spam_llm"
    
    logging_steps: int = 50
    eval_steps: int = 200

    def get_output_dir(self, model_name: str) -> Path:
        safe = model_name.replace("/", "_")
        return self.base_output_dir / safe
    
    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("train.batch_size must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("train.learning_rate must be > 0.")
        if self.num_epochs <= 0:
            raise ValueError("train.num_epochs must be > 0.")
        if not (0 <= self.warmup_ratio < 1):
            raise ValueError("train.warmup_ratio must be in [0, 1).")

@dataclass
class EmailFeatureCfg:
    use_features: List[str] = field(
        default_factory=lambda: [
            "num_cc", 
            "num_hops", 
            "subject_len", 
            "content_len", 
            "has_link"
        ]
    )

    def validate(self) -> None:
        if not self.use_features:
            raise ValueError("metadata.use_features cannot be empty.")


# -----------------------------
# Master project config
# -----------------------------

@dataclass
class ProjectConfig:
    data: DataCfg = field(default_factory=DataCfg)
    preprocessing: PreprocessingCfg = field(default_factory=PreprocessingCfg)
    metadata: EmailFeatureCfg = field(default_factory=EmailFeatureCfg)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- validation & convenience ----
    def validate(self) -> None:
        self.data.validate()
        self.preprocessing.validate()
        self.metadata.validate()
        self.model.validate()
        self.train.validate()

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ---- JSON IO ----
    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls._from_dict(cfg)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, ensure_ascii=False)

    # ---- internal: recursive build from dict ----
    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "ProjectConfig":
        return cls(
            data=DataCfg(**d.get("data", {})),
            preprocessing=PreprocessingCfg(**d.get("preprocessing", {})),
            metadata=EmailFeatureCfg(**d.get("metadata", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
        )