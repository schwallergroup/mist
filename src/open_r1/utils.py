
import os
from typing import List, Optional, Union, Callable
from pydantic import Field
from dataclasses import dataclass
from trl import GRPOConfig, GRPOTrainer
import logging
from transformers.trainer_utils import get_last_checkpoint


MetricFunc = Callable[[], dict]

class ExtendedGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        metric_funcs: Optional[Union[MetricFunc, list[MetricFunc]]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Define metric functions to use (default is None)
        if metric_funcs is None:
            metric_funcs = []
        elif not isinstance(metric_funcs, list):
            metric_funcs = [metric_funcs]
        self._metric_funcs = metric_funcs

    def _generate_and_score_completions(self, *args, **kwargs):
        """
        Overload the method to add custom metric to log.
        """
        output = super()._generate_and_score_completions(*args, **kwargs)

        # Additional logging
        mode = "eval" if self.control.should_evaluate else "train"
        for metric_func in self._metric_funcs:
            metrics = metric_func()
            for metric_name, metric_value in metrics.items():
                self._metrics[mode][f"custom/{metric_name}"].append(metric_value)

        return output

@dataclass
class ExtendedGRPOConfig(GRPOConfig):
    dataset_id_or_path: str = "/cache/data/"
    chem_task: str = "CountdownTask"
    task_variant: str = "base"
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
        rwds.append(getattr(task, r+"_reward"))
    return rwds