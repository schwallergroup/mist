"""Countdown task - toy example."""

from ..base import RLTask
import random
import re
import os


class CountdownTask(RLTask):

    def accuracy_reward(self, completions, target, nums, **kwargs):
        """
        Evaluates completions based on:
        2. Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers
        
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
                completion = "<think>" + completion
                # Check if the format is correct
                match = re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
                
                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue
                
                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builtins__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                    if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                        os.makedirs("completion_samples", exist_ok=True)
                        log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                        with open(log_file, "a") as f:
                            f.write(f"\n\n==============\n")
                            f.write(completion)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0) 
        return rewards

