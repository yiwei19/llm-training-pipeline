from transformers import HfArgumentParser, Trainer, TrainingArguments, default_data_collator

from src.arguments import ModelArguments, DataTrainingArguments
from src.utils.logging_utils import setup_logging
from src.utils.seed_utils import seed_everything
from src.utils.hf_utils import load_model, load_tokenizer
from src.data.pretrain_data import load_pretrain_raw_dataset, tokenize_function, group_texts

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger = setup_logging(training_args)
    seed_everything(training_args.seed)

    training_from_scratch = (model_args.config_name is not None) and (model_args.model_name_or_path is None)

    if training_from_scratch:
        logger.warning("Initializing model FROM SCRATCH via --config_name")
        model, _ = load_model(model_args, training_from_scratch=True)
        tokenizer_name = model_args.tokenizer_name or model_args.config_name
    else:
        logger.warning("Loading PRETRAINED model via --model_name_or_path")
        model, _ = load_model(model_args, training_from_scratch=False)
        tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path

    tokenizer = load_tokenizer(tokenizer_name, trust_remote_code=model_args.trust_remote_code)

    if not data_args.train_files:
        raise ValueError("--train_files is required (JSONL with 'text').")

    ds = load_pretrain_raw_dataset(data_args.train_files)
    column_names = list(ds["train"].features)

    tokenized = ds.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Tokenizing dataset",
    )

    block_size = int(data_args.block_size)

    lm_datasets = tokenized.map(
        lambda ex: group_texts(ex, block_size=block_size),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    logger.info("Start training (pretrain)")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    logger.info("Done.")

if __name__ == "__main__":
    main()
