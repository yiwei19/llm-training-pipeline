from itertools import chain
from datasets import load_dataset

def load_pretrain_raw_dataset(train_files):
    # JSONL with {"text": "..."}
    return load_dataset("json", data_files=train_files)

def tokenize_function(examples, tokenizer):
    return tokenizer([item for item in examples["text"]])

def group_texts(examples, block_size: int):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
