import logging
import os
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from transformers import PreTrainedModel
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

        # Set up logging of good completions
        self.logging_completions = {
            "save_completions": args.save_completions,
            "save_completions_dir": args.save_completions_dir,
            "save_completions_min_reward_threshold": args.save_completions_min_reward_threshold,
            "save_completions_min_reward_adaptive": False,
            "save_completions_top_reward_percentage": args.save_completions_top_reward_percentage,
            "save_completions_chunk_size": args.save_completions_chunk_size,
            "current_chunk_id": 1,
            "n_completions": 0,
            "min_reward_seen": None,
            "max_reward_seen": None,
            "buffer": [],
            "good_completions": [],
        }
        if self.logging_completions["save_completions_min_reward_threshold"] is None:
            # If min_reward is None (default, not provided), use the adaptive mode
            self.logging_completions["save_completions_min_reward_threshold"] = -1e6
            self.logging_completions["save_completions_min_reward_adaptive"] = True
            assert self.logging_completions["save_completions_top_reward_percentage"] is not None, "[ExtendedGRPOTrainer.__init__] save_completions_top_reward_percentage should not be None if min_reward is None (adaptive mode)"
            assert 0 < self.logging_completions["save_completions_top_reward_percentage"] <= 1, "[ExtendedGRPOTrainer.__init__] save_completions_top_reward_percentage should be in (0, 1] (if min_reward is None, adaptive mode)"
        if self.logging_completions["save_completions"]:
            if os.path.isdir(self.logging_completions["save_completions_dir"]) is False:
                os.makedirs(self.logging_completions["save_completions_dir"], exist_ok=True)
            def _reward_func_wrapper(reward_func):
                def reward_func_wrapper(*args, **kwargs):
                    reward_key = f"_reward/{reward_func.__name__}"
                    assert reward_key not in kwargs, f"[ExtendedGRPOTrainer.__init__] reward_key ({reward_key}) should not be in kwargs"
                    rewards = reward_func(*args, **kwargs)
                    logged_completions = {
                        **kwargs,
                        reward_key: rewards,
                    }
                    self.logging_completions["buffer"].append(logged_completions)
                    return rewards
                return reward_func_wrapper
            for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, PreTrainedModel):
                    pass # TODO: no support yet
                else:
                    # Note: only supported if reward functions are called with keyworded parameters only (that is the case in TRL 0.14.0)
                    self.reward_funcs[i] = _reward_func_wrapper(reward_func)

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
            self.sampling_params = SamplingParams(**sampling_params_dict)
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

    def save_good_completions(self):
        # Check if completions should be logged
        if self.logging_completions["save_completions"] is False:
            return None

        # Check the number of new completions
        assert len(set([len(b) for b in self.logging_completions["buffer"]])) == 1, "[ExtendedGRPOTrainer.log_good_completions] Logged completions buffer elements should have the same length for all logged completions"
        n_new_completions = list(set([len(b) for b in self.logging_completions["buffer"]]))[0]

        # Merge self.logging_completions["buffer"] into logged_completions_buffer
        logged_completions_buffer = dict()
        for i, logged_completions in enumerate(self.logging_completions["buffer"]):
            for key, value in logged_completions.items():
                if key not in logged_completions_buffer:
                    logged_completions_buffer[key] = logged_completions[key]
                else:
                    assert logged_completions_buffer[key] == logged_completions[key], f"[ExtendedGRPOTrainer.log_good_completions] Logged completions buffer elements should have the same value for all logged completions (key: {key})"
        self.logging_completions["buffer"] = []

        # Check logged_completions_buffer and the existing keys (for reserved keys)
        assert "_completion_id" not in logged_completions_buffer, f"[ExtendedGRPOTrainer.log_good_completions] _completion_id should not already be in logged_completions_buffer"
        assert "_global_step" not in logged_completions_buffer, f"[ExtendedGRPOTrainer.log_good_completions] _global_step should not already be in logged_completions_buffer"
        assert "_rewards" not in logged_completions_buffer, f"[ExtendedGRPOTrainer.log_good_completions] _rewards should not already be in logged_completions_buffer"
        assert "_min_reward_threshold" not in logged_completions_buffer, f"[ExtendedGRPOTrainer.log_good_completions] _min_reward_threshold should not already be in logged_completions_buffer"

        # Add information about completion_id & global_step (in logged_completions_buffer)
        logged_completions_buffer["_completion_id"] = list(range(self.logging_completions["n_completions"]+1, self.logging_completions["n_completions"]+n_new_completions+1))
        logged_completions_buffer["_global_step"] = [self.state.global_step] * n_new_completions
        self.logging_completions["n_completions"] += n_new_completions

        # Processing of the rewards (in logged_completions_buffer)
        reward_keys = [k for k in list(logged_completions_buffer.keys()) if k.startswith('_reward/')]
        assert "reward" not in reward_keys, f"[ExtendedGRPOTrainer.log_good_completions] 'reward' key should not always be in reward_keys"
        logged_completions_buffer["_rewards"] = []
        for i in range(n_new_completions):
            _rewards = dict()
            for reward_key in reward_keys:
                _rewards[reward_key.split('/', 1)[1]] = logged_completions_buffer[reward_key][i]
            _rewards['reward'] = sum(list(_rewards.values()))
            logged_completions_buffer["_rewards"].append(_rewards)

        # Update min_reward_seen & max_reward_seen
        min_reward_seen = min([_rewards['reward'] for _rewards in logged_completions_buffer["_rewards"]])
        max_reward_seen = max([_rewards['reward'] for _rewards in logged_completions_buffer["_rewards"]])
        if self.logging_completions["min_reward_seen"] is None:
            self.logging_completions["min_reward_seen"] = min_reward_seen
        else:
            self.logging_completions["min_reward_seen"] = min(min_reward_seen, self.logging_completions["min_reward_seen"])
        if self.logging_completions["max_reward_seen"] is None:
            self.logging_completions["max_reward_seen"] = max_reward_seen
        else:
            self.logging_completions["max_reward_seen"] = max(max_reward_seen, self.logging_completions["max_reward_seen"])

        # Update save_completions_min_reward_threshold (if save_completions_min_reward_adaptive is True)
        if self.logging_completions["save_completions_min_reward_adaptive"]:
            reward_range = self.logging_completions["max_reward_seen"] - self.logging_completions["min_reward_seen"]
            new_save_completions_min_reward_threshold = self.logging_completions["max_reward_seen"] - self.logging_completions["save_completions_top_reward_percentage"] * reward_range
            self.logging_completions["save_completions_min_reward_threshold"] = max(self.logging_completions["save_completions_min_reward_threshold"], new_save_completions_min_reward_threshold)

        # Split logged_completions_buffer
        logged_keys = list(logged_completions_buffer.keys())
        for k in logged_keys:
            assert len(logged_completions_buffer[k]) == n_new_completions, f"[ExtendedGRPOTrainer.log_good_completions] logged_completions_buffer[{k}] should have the same length as n_new_completions ({n_new_completions})"
        logged_completions_buffer = [{k: logged_completions_buffer[k][i] for k in logged_keys} for i in range(n_new_completions)]

        # Remove bad completions (below min_reward)
        min_reward_threshold = self.logging_completions["save_completions_min_reward_threshold"]
        logged_completions_buffer = [x for x in logged_completions_buffer if x['_rewards']['reward'] >= min_reward_threshold]

        # Add good completions to the list (and remove bad completions)
        self.logging_completions["good_completions"].extend(logged_completions_buffer)
        self.logging_completions["good_completions"] = [x for x in self.logging_completions["good_completions"] if x['_rewards']['reward'] >= min_reward_threshold]

        # Save metadata
        metadata = {
            "save_completions": self.logging_completions["save_completions"],
            "save_completions_dir": self.logging_completions["save_completions_dir"],
            "save_completions_min_reward_threshold": self.logging_completions["save_completions_min_reward_threshold"],
            "save_completions_min_reward_adaptive": self.logging_completions["save_completions_min_reward_adaptive"],
            "save_completions_top_reward_percentage": self.logging_completions["save_completions_top_reward_percentage"],
            "save_completions_chunk_size": self.logging_completions["save_completions_chunk_size"],
            "current_chunk_id": self.logging_completions["current_chunk_id"],
            "n_completions": self.logging_completions["n_completions"],
            "min_reward_seen": self.logging_completions["min_reward_seen"],
            "max_reward_seen": self.logging_completions["max_reward_seen"],
        }
        json.dump(metadata, open(f"{self.logging_completions['save_completions_dir']}/metadata.json", mode='w'), indent=4)

        # Save good completions
        good_completions = self.logging_completions["good_completions"]
        for i in range(len(good_completions)):
            good_completions[i]['_min_reward_threshold'] = min_reward_threshold
        json.dump(good_completions, open(f"{self.logging_completions['save_completions_dir']}/good_completions_chunk_{self.logging_completions['current_chunk_id']}.json", mode='w'))

        # If the chunk size is reached, empty the good completions list & increment the chunk id
        if len(good_completions) >= self.logging_completions["save_completions_chunk_size"]:
            self.logging_completions["current_chunk_id"] += 1
            self.logging_completions["good_completions"] = []

    def compute_loss(self, *args, **kwargs):
        """
        Override the method to add custom metrics to log.
        """
        output = super().compute_loss(*args, **kwargs)

        # Additional logging
        self.add_custom_metrics()
        if self.logging_completions["save_completions"]:
            self.save_good_completions()

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
    save_completions: bool = True
    save_completions_dir: str = "/Documents/sink_good_completions"
    save_completions_min_reward_threshold: float = None
    save_completions_top_reward_percentage: float = 0.1
    save_completions_chunk_size: int = 1000


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
