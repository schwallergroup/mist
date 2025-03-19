import logging
import os
from dataclasses import dataclass
from typing import List

from transformers.trainer_utils import get_last_checkpoint

from pydantic import Field
from trl import GRPOConfig


@dataclass
class ExtendedGRPOConfig(GRPOConfig):
    dataset_id_or_path: str = "/cache/data/"
    chem_task: str = "CountdownTask"
    tokenizer_name_or_path: str = None
    dataset_splits: str = "train"
    base_model_name: str = "None"
    rewards: List[str] = Field(default_factory=["accuracy", "format"])
    special_smiles_tags: bool = False


def setup_logger(name="logger"):
    """Setup logger with colored output."""
    logger = logging.getLogger(name)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def get_reward_list(task, rewards):
    rwds = []
    for r in rewards:
        rwds.append(getattr(task, r + "_reward"))
    return rwds
