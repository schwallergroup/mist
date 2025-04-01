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

1. **Exact Match (accuracy_reward)**
   
   Calculates binary rewards based on exact mechanism classification:
   - Correct mechanism classification: 1.0
   - Incorrect mechanism classification: 0.0

If only exact match does not work, we could add the following reward function:
- if the model predicts the reaction class that belongs to the similar reaction class, such as the class that involves core mechanism(M1), bicatalytic steps(M2-M5), activation steps(M6-M8) and deactivation steps(M9-M20), we could add a small reward.
- if the model predicts the correct reaction class as the possible reaction class, we could add a small reward(But there is a risk of reward hacking by answering all as possible reaction class).

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