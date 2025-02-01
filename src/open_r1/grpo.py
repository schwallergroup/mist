# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wandb
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from tasks.reactions import ForwardReaction
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from prompts import system_prompt

def main(script_args, training_args, model_args):
    # Load the dataset
    task = ForwardReaction(
        root_dir="data/USPTO",
        src_train_file="src-train.txt",
        tgt_train_file="tgt-train.txt",
        src_test_file="src-test.txt",
        tgt_test_file="tgt-test.txt",
    )
    dataset = task.load()

    # Get reward functions
    reward_funcs = [task.accuracy_reward, task.format_reward]

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns("messages")

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    accelerator = trainer.accelerator
    if accelerator.is_main_process:
        wandb.init(
            project="R1-Zero",
            name=training_args.run_name,
            config={
                **script_args.__dict__,
                **model_args.__dict__,
                "system_prompt": system_prompt,
                "question_template": task.question_template,

            }
        )
    accelerator.wait_for_everyone()

    trainer.train()

    if accelerator.is_main_process:
        wandb.finish()
    accelerator.wait_for_everyone()

    # # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
