import logging
import os
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from transformers.trainer_utils import get_last_checkpoint

from pydantic import Field
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams


MetricFunc = Callable[[], dict]


class ExtendedGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args_,
        args: GRPOConfig,
        metric_funcs: Optional[Union[MetricFunc, list[MetricFunc]]] = None,
        **kwargs_,
    ):
        # Note: current trl library version used in the sink repo: 0.14.0
        super().__init__(*args_, args=args, **kwargs_)

        # Define metric functions to use (default is None)
        if metric_funcs is None:
            metric_funcs = []
        elif not isinstance(metric_funcs, list):
            metric_funcs = [metric_funcs]
        self._metric_funcs = metric_funcs

        # Update SamplingParams to use
        if isinstance(args, ExtendedGRPOConfig):
            sampling_params_dict = {
                "n": self.num_generations,  # default from TRL 0.14.0
                "temperature": args.temperature,  # default from TRL 0.14.0
                "max_tokens": self.max_completion_length,  # default from TRL 0.14.0
            }
            sampling_params_dict.update(args.sampling_params_config)  # added from the sampling_params config -> overwrite default values
            self.sampling_params = SamplingParams(sampling_params_dict)
        print(f"SamplingParams used in ExtendedGRPOTrainer: {self.sampling_params}")

    def add_custom_metrics(self):
        # Additional logging
        mode = "eval" if self.control.should_evaluate else "train"
        for metric_func in self._metric_funcs:
            metrics = metric_func()
            for metric_name, metric_value in metrics.items():
                if mode in self._metrics.keys():
                    # Compatible with the current version 'main' of trl repository
                    self._metrics[mode][f"custom/{metric_name}"].append(
                        metric_value
                    )
                else:
                    # Compatible with the "older versions" of trl repository (0.14.0 included)
                    self._metrics[f"custom/{metric_name}"].append(metric_value)

    def compute_loss(self, *args, **kwargs):
        """
        Override the method to add custom metrics to log.
        """
        output = super().compute_loss(*args, **kwargs)

        # Additional logging
        self.add_custom_metrics()

        return output


@dataclass
class ExtendedGRPOConfig(GRPOConfig):
    dataset_id_or_path: str = "/cache/data/"
    chem_task: str = "CountdownTask"
    task_mode: str = "base"
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_name_or_path: str = None
    dataset_splits: str = "train"
    base_model_name: str = "None"
    base_model_id: str = "None"
    rewards: List[str] = Field(default_factory=["accuracy", "format"])
    special_smiles_tags: bool = False
    task_recipe: str = "None"
    task_recipe_suffix: str = "None"
    slurm_job_id: str = "None"
    slurm_resume_job_id: str = "None"
    sampling_params_config_name: str = "default"
    sampling_params_config: Dict[str, Any] = field(default_factory=dict)


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


def load_sampling_params_config(training_args: ExtendedGRPOConfig):
    """
    Update the training arguments with the sampling parameters
    :param training_args: Training arguments [ExtendedGRPOConfig]
    :return: Updated training arguments [ExtendedGRPOConfig]
    """
    sampling_params_dir = "/Documents/sink/sampling_params"

    # Read model_default_sampling_params.txt
    model_default_sampling_params = dict()
    with open(f"{sampling_params_dir}/model_default_sampling_params.txt", mode='r') as f:
        for line in f:
            if ':' not in line:
                continue
            line_split = line.split(':')
            assert len(line_split) == 2, f"Invalid format in model_default_sampling_params.txt -> each line should be in the format 'key: value' (with a single ':', found {len(line_split)-1} instead of 1)"
            model_id = line_split[0].strip()
            sampling_params_config_name = line_split[1].strip()
            assert model_id not in model_default_sampling_params, f"Invalid format in model_default_sampling_params.txt -> model_id {model_id} is duplicated"
            if sampling_params_config_name != "default":
                assert os.path.isfile(f"{sampling_params_dir}/{sampling_params_config_name}.json"), f"Invalid format in model_default_sampling_params.txt -> sampling_params_config_name ({sampling_params_config_name}.json) does not exist"
            model_default_sampling_params[model_id] = sampling_params_config_name

    # Update training_args.sampling_params_config_name (if needed)
    if training_args.sampling_params_config_name == "default":
        if training_args.base_model_id in model_default_sampling_params:
            training_args.sampling_params_config_name = model_default_sampling_params[training_args.base_model_id]

    # Update training_args.sampling_params_config (if needed)
    if training_args.sampling_params_config_name != "default":
        sampling_params = json.load(open(f"{sampling_params_dir}/{training_args.sampling_params_config_name}.json", mode='r'))
        training_args.sampling_params_config = sampling_params
    else:
        # Ensure sampling_params/default.json does not exist (the default should never be modified)
        assert not os.path.isfile(f"{sampling_params_dir}/default.json"), f"The file sampling_params/default.json exists. The global default sampling_params can't be overwritten, please remove this file (or put the config in a new file)."

    return training_args
