import json
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ROLES = {
    "human": "<|im_start|>human",
    "user": "<|im_start|>human",
    "assistant": "<|im_start|>assistant",
    "system": "<|im_start|>system",
}

def load_sft_jsonl(path: str, max_samples: int = None) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def preprocess(
    sources: List[List[dict]],
    tokenizer,
    max_len: int,
    system_message: str,
    ignore_token_id: int = -100,
):
    """
    Robust ChatML-style SFT preprocessing that:
    - Does NOT assume <|im_start|> / <|im_end|> are single tokens.
    - Builds input_ids and targets (labels) with exactly the same length, always.
    - Only trains on assistant message content tokens; all system/human tokens are ignored in labels.
    """

    # NOTE: We explicitly set add_special_tokens=False to avoid tokenizer adding BOS/EOS unexpectedly.
    im_start_ids = tokenizer("<|im_start|>", add_special_tokens=False).input_ids
    im_end_ids   = tokenizer("<|im_end|>",   add_special_tokens=False).input_ids
    nl_ids       = tokenizer("\n",           add_special_tokens=False).input_ids  # typically [198] for GPT2

    # Role tokens: these are the literal strings used in your data pipeline.
    # They may tokenize into MULTIPLE ids; we handle that.
    role_system_ids    = tokenizer("<|im_start|>system",    add_special_tokens=False).input_ids
    role_human_ids     = tokenizer("<|im_start|>human",     add_special_tokens=False).input_ids
    role_assistant_ids = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # GPT2 has no pad by default. Safer to pad with eos.
        pad_id = tokenizer.eos_token_id

    input_ids, targets = [], []
    attention_masks = []

    def _append(segment_ids: List[int], label_mode: str):
        """
        Append segment_ids to current input and corresponding labels to current target.
        label_mode:
          - "ignore": all labels are ignore_token_id
          - "copy":  labels equal to segment_ids (we typically do NOT use this for ChatML)
        """
        nonlocal input_id, target
        input_id += segment_ids
        if label_mode == "ignore":
            target += [ignore_token_id] * len(segment_ids)
        elif label_mode == "copy":
            target += segment_ids
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

    for i in tqdm(range(len(sources)), desc="preprocess(sft)"):
        source = sources[i]
        if not source:
            continue

        # Some datasets may start with assistant; if so, drop until first human/user.
        if source[0].get("from") not in ("human", "user"):
            source = source[1:]
        if not source:
            continue

        input_id, target = [], []

        # -------- System message block (ignored in loss) --------
        # Format:
        # <|im_start|>system\n {system_message} <|im_end|>\n
        _append(role_system_ids + nl_ids, "ignore")
        _append(tokenizer(system_message, add_special_tokens=False).input_ids, "ignore")
        _append(im_end_ids + nl_ids, "ignore")

        # -------- Conversation turns --------
        for msg in source:
            frm = msg.get("from")
            text = msg.get("value", "")
            if frm in ("human", "user"):
                role_ids = role_human_ids
                # Human block: ignore all tokens in labels
                _append(role_ids + nl_ids, "ignore")
                _append(tokenizer(text, add_special_tokens=False).input_ids, "ignore")
                _append(im_end_ids + nl_ids, "ignore")

            elif frm == "assistant":
                role_ids = role_assistant_ids

                # Assistant block:
                # We ignore role tokens + newline, but we TRAIN on the assistant content tokens.
                _append(role_ids + nl_ids, "ignore")

                content_ids = tokenizer(text, add_special_tokens=False).input_ids
                _append(content_ids, "copy")          # train on assistant content

                _append(im_end_ids + nl_ids, "ignore")  # usually ignore the control tokens

            else:
                raise ValueError(f"Unknown role in data: {frm}")

        # -------- Pad / truncate to max_len --------
        if len(input_id) < max_len:
            pad_len = max_len - len(input_id)
            input_id += [pad_id] * pad_len
            target   += [ignore_token_id] * pad_len
        else:
            input_id = input_id[:max_len]
            target   = target[:max_len]

        # Final safety check
        assert len(input_id) == len(target), f"length mismatch: {len(input_id)} vs {len(target)}"

        attn = [0 if tid == pad_id else 1 for tid in input_id]
        attention_masks.append(attn)
        input_ids.append(input_id)
        targets.append(target)

    return {"input_ids": input_ids, "labels": targets, "attention_mask": attention_masks}


class SupervisedDataset(Dataset):
    def __init__(self, raw_data: List[dict], tokenizer, max_len: int, system_message: str, ignore_token_id: int = -100):
        super().__init__()
        sources = [ex["conversations"] for ex in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len=max_len, system_message=system_message, ignore_token_id=ignore_token_id)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i], "attention_mask": self.attention_mask[i]}
