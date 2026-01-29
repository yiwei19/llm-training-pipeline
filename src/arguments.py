from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HF name for a pretrained model (post-train / SFT base)."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HF name for config.json (train-from-scratch)."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HF name for tokenizer."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code for model/tokenizer."},
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Model dtype: auto|bfloat16|float16|float32"},
    )

@dataclass
class DataTrainingArguments:
    train_files: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Training data paths (JSONL). Pretrain expects 'text'; SFT expects 'conversations'."}
    )
    block_size: Optional[int] = field(
        default=2048,
        metadata={"help": "Chunk length for pretrain grouping."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "Num workers for dataset.map()."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Debug: limit number of raw samples read from JSONL (SFT loader)."},
    )

@dataclass
class SFTDataArguments:
    max_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for SFT samples."},
    )
    system_message: str = field(
        default="You are a helpful assistant.",
        metadata={"help": "System prompt for chat template."},
    )
    ignore_token_id: int = field(
        default=-100,
        metadata={"help": "Label mask id (ignore index)."},
    )
