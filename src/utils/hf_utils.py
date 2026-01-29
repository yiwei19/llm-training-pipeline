import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def _dtype_from_str(torch_dtype: str):
    if torch_dtype is None or torch_dtype == "auto":
        return None
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

def load_tokenizer(tokenizer_name_or_path: str, trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_model(model_args, training_from_scratch: bool = False):
    dtype = _dtype_from_str(model_args.torch_dtype)

    if training_from_scratch:
        if model_args.config_name is None:
            raise ValueError("Training from scratch requires --config_name")
        config = AutoConfig.from_pretrained(model_args.config_name, trust_remote_code=model_args.trust_remote_code)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        return model, config

    if model_args.model_name_or_path is None:
        raise ValueError("Loading a pretrained model requires --model_name_or_path")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=dtype,
    )
    return model, model.config
