Addition/removal of hydrogens in SMILES
===================

.. currentmodule:: open_r1.tasks.smiles_understanding.smiles_hydrogen

SmilesHydrogen
-------------

.. autoclass:: SmilesHydrogen
   :members:
   :show-inheritance:

Task Description
--------------

The `SmilesHydrogen` task is designed to add or remove hydrogen atoms from SMILES notation. The task is divided into two sub-tasks: adding hydrogen atoms to a SMILES string (missing implicit hydrogens) and removing hydrogen atoms from a SMILES string (containing all hydrogens).

Features
--------

- Loads SMILES with/without hydrogen atoms from a dataset
- Converts SMILES without hydrogen atoms to SMILES with hydrogen atoms & vice versa
- Provides reward evaluations based on selected options ('format', 'reasoning_steps', 'levenstein_accuracy', 'smiles_validity', 'tanimoto_accuracy', 'accuracy' or alternatively 'sequential')
- Log custom metrics in WANDB for the two sub-tasks: addH and removeH

Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.smiles_understanding.smiles_hydrogen import SmilesHydrogen

    # Initialize the task
    task = SmilesHydrogen(
        dataset_id_or_path="/path/to/your/dataset.csv",
    )

    # Load the dataset
    dataset = task.load()

    # Example of reward calculation
    completions = ["<answer>SMILES_answer</answer>"]
    solution = ["SMILES_solution"]
    question_category = "addH"
    rewards = task.accuracy_reward(completions, solution, question_category)

Data Format
----------

The task expects data files formatted as CSV with the following columns:

- `SMILES_noHs`: SMILES strings without hydrogen atoms
- `SMILES_Hs`: SMILES strings with hydrogen atoms
- `levenshtein_distance`: levenshtein distance between SMILES_noHs and SMILES_Hs
- `length_diff`: character length difference between SMILES_noHs and SMILES_Hs

Reward Functions
--------------

Note: most of the base rewards are modified in order to have continuous rewards instead of discrete rewards (for example for the format_reward). It is an intended behaviour in order to have as much continuity as possible for the sequential reward.

1. **Format (format_reward)**

   This function evaluates the correctness of the completion format:
   - +1.0 if the completion format is perfectly correct
   - anything between -1.0 and +1.0 if the completion format is partially correct
   - -1.0 if the completion format is completely wrong

2. **Reasoning steps (reasoning_steps_reward)**

   This function evaluates if the reasoning traces contain step by step reasoning:
   - +1.0 if the reasoning traces contain at least 3 steps
   - +0.66 if the reasoning traces contain 2 steps
   - +0.33 if the reasoning traces contain 1 step
   - 0.0 if the reasoning traces contain no steps

3. **Levenshtein accuracy (levenstein_accuracy_reward)**

   This function computes the levenshtein distance between the answer and the expected solution:
   - +1.0 if the answer and the solution are identical
   - anything between +0.1 and +1.0 if the answer is better than the input SMILES
   - +0.1 if the answer and the input SMILES are identical
   - -0.5 if the answer is to far from the solution/input

4. **SMILES validity (smiles_validity_reward)**

   This function checks if the answer contains a valid SMILES string and penalizes invalid SMILES:
   - 0.0 if the answer is a valid SMILES string
   - -0.5 if the answer is not a valid SMILES string

5. **Tanimoto similarity (tanimoto_accuracy_reward)**

   This function checks the Tanimoto similarity between the answer SMILES and the solution SMILES:
   - +1.0 if the answer and the solution are identical
   - anything between -0.5 and +0.5 if the answer is different from the solution
   - -0.5 if the answer has a Tanimoto similarity of 0.0 with the solution

6. **Exact Match (accuracy_reward)**

   This function checks if the answer is perfectly correct:
   - +1.0 if the answer is identical to the solution
   - 0.0 else

7. **Sequential reward (sequential_reward)**

   This reward function can be used alone (without adding the other rewards). It is aggregating the 6 previous rewards to compute a final reward. The idea behind this reward is to promote the optimization of the rewards in a sequential manner. The order of the optimization is the following: format, reasoning_steps, levenstein_accuracy, smiles_validity, tanimoto_accuracy, accuracy. In order to optimize the next reward in the list, the previous reward value needs to be good, etc.
   The final reward is computed by summing the 6 rewards (r1 to r6) in the following manner:
   - Rescale each reward between 0 and 1 (r1s to r6s). For example, r1s = 0 if the format is completely wrong (format_accuracy of -1.0) and r1s = 1 if the format is perfectly correct (format_accuracy of 1.0).
   - The final reward is compute based on the previous reward rescaled: reward = r1 + r1s * r2 + r1s * r2s * r3 etc...
   - For example, if the format reward is r1=0.5 (r1s=0.75 which means 75% of the reward), then 75% of the second reward r1 will be added etc... Due to this aggregation methods, it is better to have continuous rewards in the list.
   - The best possible reward is +5.0 if all rewards are perfect.
   - The worst possible reward is -1.0 if the format is completely wrong.

Task Example
-----------

In this task, the model is asked to add or remove hydrogen atoms from a SMILES string:

.. code-block:: text

   Sub-task: removeH
   Input: SMILES with hydrogen atoms
   Output: SMILES without hydrogen atoms

.. code-block:: text

   Sub-task: addH
   Input: SMILES without hydrogen atoms
   Output: SMILES with hydrogen atoms
