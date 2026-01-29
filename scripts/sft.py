from transformers import HfArgumentParser, Trainer, TrainingArguments

from src.arguments import ModelArguments, DataTrainingArguments, SFTDataArguments
from src.utils.logging_utils import setup_logging
from src.utils.seed_utils import seed_everything
from src.utils.hf_utils import load_model, load_tokenizer
from src.data.sft_data import load_sft_jsonl, SupervisedDataset

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTDataArguments, TrainingArguments))
    model_args, data_args, sft_args, training_args = parser.parse_args_into_dataclasses()

    logger = setup_logging(training_args)
    seed_everything(training_args.seed)

    model, _ = load_model(model_args, training_from_scratch=False)
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = load_tokenizer(tokenizer_name, trust_remote_code=model_args.trust_remote_code)

    if not data_args.train_files or len(data_args.train_files) != 1:
        raise ValueError("SFT expects exactly one JSONL path via --train_files (each line has 'conversations').")

    raw = load_sft_jsonl(data_args.train_files[0], max_samples=data_args.max_samples)
    logger.info(f"Loaded {len(raw)} SFT samples")

    train_dataset = SupervisedDataset(
        raw_data=raw,
        tokenizer=tokenizer,
        max_len=int(sft_args.max_len),
        system_message=sft_args.system_message,
        ignore_token_id=int(sft_args.ignore_token_id),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Start training (SFT)")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    logger.info("Done.")

if __name__ == "__main__":
    main()
