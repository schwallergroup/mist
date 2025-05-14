import logging
import os
import json
import platform
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from transformers import PreTrainedModel
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from pydantic import Field
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from importlib.metadata import version

import torch
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from accelerate.utils import broadcast_object_list, gather_object
# from trl_functions14 import unwrap_model_for_generation, pad, selective_log_softmax TODO: add relative imports compatibility
# from trl17_profiling import profiling_decorator TODO: add relative imports compatibility

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

        # Set up parameters
        self.temperature = args.temperature
        self.loss_type = args.loss_type
        self.custom_kl_clipping = args.custom_kl_clipping
        self.custom_kl_clipping_mean = args.custom_kl_clipping_mean
        self.custom_kl_division_temperature = args.custom_kl_division_temperature
        self.custom_kl_nan_to_zero = args.custom_kl_nan_to_zero
        self.logging_kl = args.logging_kl
        self.logging_kl_min = args.logging_kl_min
        self.custom_clipped_surrogate_objective = args.custom_clipped_surrogate_objective
        self.custom_clipped_surrogate_objective_epsilon_high = args.custom_clipped_surrogate_objective_epsilon_high
        self.custom_clipped_surrogate_objective_epsilon_low = args.custom_clipped_surrogate_objective_epsilon_low
        self.custom_reward_tanh = args.custom_reward_tanh
        self.custom_reward_tanh_scale = args.custom_reward_tanh_scale

        # Logging
        if is_deepspeed_zero3_enabled():
            print("[ExtendedGRPOTrainer.__init__] Deepspeed ZeRO3 is enabled")
        else:
            print("[ExtendedGRPOTrainer.__init__] Deepspeed ZeRO3 is NOT enabled")
        self.logging_print_once = []

        # Set up logging of good completions
        self.logging_completions = {
            "save_completions": args.save_completions,
            "save_completions_dir": f"{args.save_completions_dir}/{args.slurm_job_id}/{platform.node()}",
            "save_completions_min_reward_threshold": args.save_completions_min_reward_threshold,
            "save_completions_min_reward_adaptive": False,
            "save_completions_top_reward_percentage": args.save_completions_top_reward_percentage,
            "save_completions_chunk_size": args.save_completions_chunk_size,
            "current_chunk_id": 1,
            "slurm_job_id": args.slurm_job_id,
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
                @functools.wraps(reward_func)  # used to keep the original function name (reward_func.__name__)
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
        assert len(set([len(b[k]) for b in self.logging_completions["buffer"] for k in b])) == 1, "[ExtendedGRPOTrainer.log_good_completions] Logged completions buffer elements should have the same length for all logged completions"
        n_new_completions = list(set([len(b[k]) for b in self.logging_completions["buffer"] for k in b]))[0]

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

    # @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        """
        Copy of _get_per_token_logps from TRL 0.17.0
        """
        # TODO: add compatibility with this function in _compute_loss_modified
        raise Exception(f"[ExtendedGRPOTrainer._get_per_token_logps] This method is not used in the current version of the code -> no support yet")
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _compute_loss_modified(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Modify the initial compute_loss method (TRL package version 0.14.0)
        This version is a mix between TRL 0.14.0 & TRL 0.17.0 and additional modifications
        """
        # Check TRL version
        if version("trl") != "0.14.0":
            raise Exception(f"[ExtendedGRPOTrainer._compute_loss_modified] TRL version should be 0.14.0 (found {version('trl')})")

        # Modified compute_loss method from here
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text) * self.num_generations

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_inputs_repeated = torch.repeat_interleave(prompt_inputs["input_ids"], self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, num_logits_to_keep):
            # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            # Divide by temperature (added in TRL 0.17.0)
            if self.custom_kl_division_temperature:
                logits = logits / self.temperature

            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        num_logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, num_logits_to_keep)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # KL logging
        self._logging_kl_previously_logged = False
        def _logging_kl(per_token_kl, completion_mask, completion_ids, preprint_message=''):
            if self.logging_kl:
                mean_kls = (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
                mean_kl = mean_kls.mean().item()
                if self.logging_kl_min is not None:
                    if mean_kl < self.logging_kl_min and self._logging_kl_previously_logged is False:
                        return None
                self._logging_kl_previously_logged = True
                n_kl = completion_mask.sum().item()
                n_elements = completion_mask.numel()
                n_kl_nan = torch.isnan(per_token_kl).sum().item()
                n_kl_inf = torch.isinf(per_token_kl).sum().item()
                mean_kls = [f"{k:.2f}" for k in mean_kls.tolist()]
                # Get the indices of the maximal value in per_token_kl
                max_kl_indices = torch.nonzero(per_token_kl == per_token_kl.max(), as_tuple=False)
                max_kl_index0 = max_kl_indices[0][0].item()
                max_kl_index1 = max_kl_indices[0][1].item()
                max_kl_completion_token = [completion_ids[max_kl_index0][max_kl_index1].item()]
                if max_kl_index1 != 0:
                    max_kl_completion_token = [completion_ids[max_kl_index0][max_kl_index1-1].item()] + max_kl_completion_token
                print(f"[LOGGING_KL]: {preprint_message}\n"
                      f"\t[LOGGING_KL]: per_token_kl.shape={per_token_kl.shape} (used={n_kl}/{n_elements}, {100 * n_kl / n_elements:.2f}%) | nan={n_kl_nan}, inf={n_kl_inf}\n"
                      f"\t[LOGGING_KL]: mean_kls: {mean_kls}\n"
                      f"\t[LOGGING_KL]: max_kl: {per_token_kl.max().item():.8f} -> {max_kl_completion_token}\n"
                      f"\t[LOGGING_KL]: mean_kl: {mean_kl:.8f}\n"
                      f"\t[LOGGING_KL]: std_kl: {torch.masked_select(per_token_kl, completion_mask.bool()).std().item():.8f}\n"
                      f"\t[LOGGING_KL]: median_kl: {torch.masked_select(per_token_kl, completion_mask.bool()).median().item():.8f}")
                kl_threshold = 1
                for _ in range(13):
                    n_kl_above_threshold = (per_token_kl >= kl_threshold).sum().item()
                    if n_kl_above_threshold >= 1:
                        print(f"\t[LOGGING_KL]: KL >= {kl_threshold}: {n_kl_above_threshold}/{n_kl} ({100 * n_kl_above_threshold / n_kl:.2f}%)")
                    # Update kl_threshold
                    if kl_threshold == 1:
                        kl_threshold = 5
                    elif kl_threshold == 5:
                        kl_threshold = 10
                    elif kl_threshold == 10:
                        kl_threshold = 25
                    elif kl_threshold == 25:
                        kl_threshold = 50
                    elif kl_threshold == 50:
                        kl_threshold = 100
                    else:
                        kl_threshold = kl_threshold * 10

        # Logging print (once)
        if 'print_prompt_inputs_attention_mask_shape' not in self.logging_print_once:
            print(f"[ExtendedGRPOTrainer._compute_loss_modified] prompt_inputs[\"attention_mask\"] shape: {prompt_inputs['attention_mask'].shape}")
            self.logging_print_once.append('print_prompt_inputs_attention_mask_shape')
        if 'print_completion_mask_shape' not in self.logging_print_once:
            print(f"[ExtendedGRPOTrainer._compute_loss_modified] completion_mask shape: {completion_mask.shape}")
            self.logging_print_once.append('print_completion_mask_shape')
        if 'print_completion_ids_shape' not in self.logging_print_once:
            print(f"[ExtendedGRPOTrainer._compute_loss_modified] completion_ids shape: {completion_ids.shape}")
            self.logging_print_once.append('print_completion_ids_shape')
        if 'print_per_token_kl_shape' not in self.logging_print_once:
            print(f"[ExtendedGRPOTrainer._compute_loss_modified] per_token_kl shape: {per_token_kl.shape}")
            self.logging_print_once.append('print_per_token_kl_shape')

        # KL clipping
        use_kl_divergence = True
        if torch.isnan(per_token_kl).sum().item() != 0:
            use_kl_divergence = False
            print(f"[ExtendedGRPOTrainer._compute_loss_modified] KL divergence is NaN ({torch.isnan(per_token_kl).sum().item()}) before modification, skipping KL divergence term")
        _logging_kl(per_token_kl, completion_mask, completion_ids)
        if self.custom_kl_nan_to_zero:
            per_token_kl = torch.nan_to_num(per_token_kl, nan=0.0, posinf=None, neginf=None)
            _logging_kl(per_token_kl, completion_mask, completion_ids, preprint_message=f"after custom_kl_nan_to_zero")
        if self.custom_kl_clipping is not None:
            per_token_kl = torch.clamp(per_token_kl, max=self.custom_kl_clipping)
            _logging_kl(per_token_kl, completion_mask, completion_ids, preprint_message=f"after custom_kl_clipping={self.custom_kl_clipping}")
        if self.custom_kl_clipping_mean is not None:
            mean_kls = (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            kl_rescaling = (torch.full_like(mean_kls, self.custom_kl_clipping_mean, device=device) / mean_kls.clamp(min=0.1*self.custom_kl_clipping_mean)).clamp(max=1.0)
            per_token_kl = per_token_kl * kl_rescaling.unsqueeze(1)
            _logging_kl(per_token_kl, completion_mask, completion_ids, preprint_message=f"after custom_kl_clipping_mean={self.custom_kl_clipping_mean}")
        if torch.isnan(per_token_kl).sum().item() != 0:
            use_kl_divergence = False
            print(f"[ExtendedGRPOTrainer._compute_loss_modified] KL divergence is NaN ({torch.isnan(per_token_kl).sum().item()}) after modification, skipping KL divergence term")
        if self.custom_kl_nan_to_zero:
            per_token_kl = torch.nan_to_num(per_token_kl, nan=0.0, posinf=None, neginf=None)
            _logging_kl(per_token_kl, completion_mask, completion_ids, preprint_message=f"after custom_kl_nan_to_zero")
        if torch.isnan(per_token_kl).sum().item() != 0:
            use_kl_divergence = False
            print(f"[ExtendedGRPOTrainer._compute_loss_modified] KL divergence is NaN ({torch.isnan(per_token_kl).sum().item()}) after nan_to_num, check the code")

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Apply tanh if needed
        rewards_base = rewards.clone()
        rewards_tanh = None
        if self.custom_reward_tanh:
            rewards = torch.tanh(rewards/self.custom_reward_tanh)
            rewards_tanh = rewards.clone()

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Compute the loss
        # x - x.detach() allows for preserving gradients from x
        coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
        if self.custom_clipped_surrogate_objective:
            coef_2 = torch.clamp(coef_1, 1 - self.custom_clipped_surrogate_objective_epsilon_low, 1 + self.custom_clipped_surrogate_objective_epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        else:
            per_token_loss = -coef_1 * advantages.unsqueeze(1)
        if self.beta != 0.0 and use_kl_divergence:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        # TODO: add this logging in wandb (require attention_mask)
        # self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        # self._metrics["num_tokens"] = [self.state.num_input_tokens_seen]

        # log prompt lengths, mean, min, max
        agg_prompt_mask = self.accelerator.gather_for_metrics(prompt_inputs["attention_mask"].sum(1))
        self._metrics["prompts/mean_length"].append(agg_prompt_mask.float().mean().item())
        self._metrics["prompts/min_length"].append(agg_prompt_mask.float().min().item())
        self._metrics["prompts/max_length"].append(agg_prompt_mask.float().max().item())

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics["completions/max_length"].append(agg_completion_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards_base).mean().item())
        if rewards_tanh is not None:
            self._metrics["reward_tanh"].append(self.accelerator.gather_for_metrics(rewards_tanh).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def compute_loss(self, *args, **kwargs):
        """
        Override the method to add custom metrics to log.
        """
        output = self._compute_loss_modified(*args, **kwargs)
        # output = super().compute_loss(*args, **kwargs) # This method was overridden by _compute_loss_modified

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
    loss_type: str = "grpo"
    custom_kl_clipping: float = None
    custom_kl_clipping_mean: float = None
    custom_kl_division_temperature: bool = True
    custom_kl_nan_to_zero: bool = True
    logging_kl: bool = False
    logging_kl_min: float = None
    custom_clipped_surrogate_objective: bool = False
    custom_clipped_surrogate_objective_epsilon_low: float = 0.1
    custom_clipped_surrogate_objective_epsilon_high: float = 0.1
    custom_reward_tanh: bool = False
    custom_reward_tanh_scale: float = 2

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


"""
FULL COPY of trl14_functions.py
(copied there because easier than modifying PYTHONPATH or relative imports for the moments)
TODO: add relative imports compatibility
"""

import itertools
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, Optional, Union
import deepspeed
from accelerate import Accelerator
from deepspeed.runtime.engine import DeepSpeedEngine
from torch.nn.parallel.distributed import DistributedDataParallel
from trl.models import PreTrainedModelWrapper


import torch
import torch.nn.functional as F
import numpy as np


# Copy of trl/models/utils.py
def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())
def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]
def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)
@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"],
    accelerator: "Accelerator",
    is_peft_model: bool = False,
    gather_deepspeed3_params: bool = True,
) -> Union["PreTrainedModelWrapper", "DeepSpeedEngine"]:
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model


# Copy of trl/trainer/utils.py
def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output
def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

