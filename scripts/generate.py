import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tok(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
