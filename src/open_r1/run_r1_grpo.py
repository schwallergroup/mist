import os
from datetime import datetime
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from transformers import AutoTokenizer

from utils import ExtendedGRPOConfig, ExtendedGRPOTrainer, setup_logger, get_checkpoint, get_reward_list
from trl import GRPOConfig, get_peft_config, ModelConfig, TrlParser
from tasks import CHEMTASKS

logger = setup_logger(__name__)


def grpo_function(
    model_args: ModelConfig, training_args: GRPOConfig
):
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            training_args.tokenizer_name_or_path
            if training_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load task
    if training_args.special_smiles_tags:
        begin_smiles_tag = "[BEGIN_SMILES]"
        end_smiles_tag = "[END_SMILES]"
    else:
        begin_smiles_tag = ""
        end_smiles_tag = ""

    task = CHEMTASKS[training_args.chem_task](
        dataset_id_or_path=training_args.dataset_id_or_path,
        dataset_splits=training_args.dataset_splits,
        task_variant=training_args.task_variant,
        task_mode=training_args.task_mode,
        begin_smiles_tag=begin_smiles_tag,
        end_smiles_tag=end_smiles_tag,
    )
    dataset = task.load()
    dataset = task.dataset_preprocess(tokenizer)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Instantiate GRPO trainer
    trainer = ExtendedGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=get_reward_list(task, training_args.rewards),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        metric_funcs=[getattr(task, "get_metrics")],
    )

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # Save model and create model card
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    logger.info("*** Training complete! ***")

def main():
    parser = TrlParser((ModelConfig, ExtendedGRPOConfig))
    model_args, training_args = parser.parse_args_and_config()
    grpo_function(model_args, training_args)

if __name__ == "__main__":
    main()
