# happy-llm Chapter 6  — Pretrain / SFT / LoRA (Transformers + Trainer)

This folder is a clean, modular project layout you can upload to Google Colab and run on a single GPU.
It is based on the happy-llm repo.

It implements:

- CLM Pretrain (next-token prediction) using Hugging Face Trainer
- SFT (instruction tuning) with a Qwen-style chat template and loss masking (only compute loss on assistant outputs)
- Optional LoRA fine-tuning via peft

DeepSpeed is included as an optional config; on free Colab it may be unavailable/unstable. Everything runs without DeepSpeed by default.

---

## 1) Colab quickstart

### A) Upload folder
Upload this whole folder to Colab (or mount Google Drive).

### B) Install deps
```bash
pip -q install -r requirements.txt
```

If you want LoRA:
```bash
pip -q install peft accelerate
```

### C) Pretrain smoke test
```bash
python scripts/pretrain.py   --model_name_or_path gpt2   --tokenizer_name gpt2   --train_files data/sample_pretrain.jsonl   --output_dir output/pretrain_demo   --block_size 128   --per_device_train_batch_size 2   --gradient_accumulation_steps 2   --learning_rate 5e-4   --num_train_epochs 1   --logging_steps 5   --save_steps 50   --save_total_limit 1
```

### D) SFT smoke test
```bash
python scripts/sft.py   --model_name_or_path gpt2   --tokenizer_name gpt2   --train_files data/sample_sft.jsonl   --output_dir output/sft_demo   --max_len 256   --per_device_train_batch_size 2   --gradient_accumulation_steps 2   --learning_rate 2e-4   --num_train_epochs 1   --logging_steps 5   --save_steps 50   --save_total_limit 1
```

### E) Generate after training
```bash
python scripts/generate.py   --model_path output/sft_demo   --prompt "你好，给我解释一下注意力机制。"
```

---

## 2) Data formats

### Pretrain JSONL
Each line:
```json
{"text": "..."}
```

### SFT JSONL
Each line:
```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "assistant", "value": "..."}
  ]
}
```

---

## 3) Project structure

```
happy_llm_ch6_colab/
  requirements.txt
  data/
    sample_pretrain.jsonl
    sample_sft.jsonl
  configs/
    ds_config_zero2.json   # optional
  src/
    arguments.py
    utils/
      logging_utils.py
      seed_utils.py
      hf_utils.py
    data/
      pretrain_data.py
      sft_data.py
  scripts/
    pretrain.py
    sft.py
    generate.py
```
