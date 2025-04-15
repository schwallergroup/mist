Kinetic Reaction Classification
===================

.. currentmodule:: open_r1.tasks.kinetic_data

KineticDataClassification
-------------

.. autoclass:: KineticDataClassification
   :members:
   :show-inheritance:

Task Description
--------------

The Kinetic Reaction Classification task is designed to identify and classify chemical reaction mechanisms from kinetic data. The task presents experimental data from multiple runs with different initial conditions and requires the model to determine which of the 20 possible reaction mechanisms (M1-M20) best explains the observed behavior.

Features
--------

- Supports classification of 20 different reaction mechanisms (M1-M20)
- Processes normalized kinetic data from multiple experimental runs
- Handles substrate, product, and catalyst concentration data
- Provides detailed reasoning process for mechanism classification

Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.kinetic_data import KineticDataClassification

    # Initialize the task
    task = KineticDataClassification(
        dataset_id_or_path="path/to/kinetic/data",
    )

    # Load the dataset
    dataset = task.load()

    # Example of reward calculation
    completions = ["<think>Detailed reasoning...</think>\\boxed{M1}"]
    solution = ["M1"]
    rewards = task.accuracy_reward(completions, solution)

Data Format
----------

The task expects data files in the following format:

- ``path/to/kinetic/data/x_train/x1_train_M1_M20_train_val_test_set_part_0.pkl``: Training data for initial catalyst concentrations
- ``path/to/kinetic/data/x_train/x2_train_M1_M20_train_val_test_set_part_0.pkl``: Training data for time series measurements
- ``path/to/kinetic/data/y_train/y_train_M1_M20_train_val_test_set.pkl``: Training labels (mechanism classes)
- ``path/to/kinetic/data/x_val/x1_val_M1_M20_train_val_test_set.pkl``: Validation data for initial catalyst concentrations
- ``path/to/kinetic/data/x_val/x2_val_M1_M20_train_val_test_set.pkl``: Validation data for time series measurements
- ``path/to/kinetic/data/y_val/y_val_M1_M20_train_val_test_set.pkl``: Validation labels (mechanism classes)

Reward Functions
--------------
The accuracy reward is calculated by the weighted sum of the following rewards.

1. **Exact Match (accuracy_reward)**
   
   Calculates binary rewards based on exact mechanism classification:
   - Correct mechanism classification: 1.0
   - Incorrect mechanism classification: 0.0

2. **Class Coverage Reward (accuracy_reward)**

   Calculates the reward based on the percentage of the 20 reaction classes that the model considered during the reasoning.
   - If the model considered all 20 reaction classes, the reward is 1.0.

3. **Data Coverage Reward (accuracy_reward)**

   Calculates the reward based on the percentage of the 4 data runs that the model considered during the reasoning.
   - If the model considered all 4 data runs, the reward is 1.0.

4. **Category Match (accuracy_reward)**

   Calculates the reward based on the answer that the model gave is in the same category as the ground truth.
   - If the model gave the answer that is in the same category as the ground truth, the reward is 1.0.
   - For example, if the ground truth is M3, and the model gave M2, the reward is 1.0.
   - Category is defined as follows:
      - M1: Core mechanism
      - M2-M5: Bicatalytic steps
      - M6-M8: Activation steps
      - M9-M20: Deactivation steps

The format reward is calculated by the following reward.
5. **Format Reward (format_reward)**

   Calculates the reward based on the format of the answer.
   - If the answer includes ``<think>`` and ``\\boxed{}``, the reward is 1.0.
   - Otherwise, the reward is 0.0.

   **Note:**
   - This format is intended to train DeepSeek R1 Distill Qwen.
   - There is the suggestion to use the format in the prompt in their transformer website. I once tried to use ``<think>`` and ``<answer>``, but the format reward did not start to increase in the first 50 global steps.


Task Example
-----------

.. code-block:: text

   Input: 
   # Data Run 1
   - Initial concentration of catalyst: 0.1
   - Initial concentration of substrate: 1.0
   - Time series data: [0, 1, 2, 3]
   - Substrate data: [1.0, 0.8, 0.6, 0.4]
   - Product data: [0.0, 0.2, 0.4, 0.6]
   
   Output: M1
   Reasoning: <think>
   The reaction shows first-order kinetics with respect to substrate...
   The mechanism must involve a single catalytic site...
   Therefore, M1 is the most likely mechanism.
   </think>
   \\boxed{M1}