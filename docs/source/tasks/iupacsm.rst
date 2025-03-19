IUPAC to SMILES
===============

.. currentmodule:: open_r1.tasks.iupac2smiles

Iupac2Smiles
------------

.. autoclass:: Iupac2Smiles
   :members:
   :show-inheritance:

Task Description
----------------

The `Iupac2Smiles` task is designed to convert chemical names in IUPAC format to SMILES notation, which is crucial for computational representations of molecular structures. The task involves using a provided dataset to predict SMILES given IUPAC names.

Features
--------

- Loads and processes chemical name and SMILES data from a CSV file
- Converts IUPAC names to SMILES notation
- Uses rewards based on exact match and Tanimoto similarity for evaluation
- Provides pre-defined response templates for model training

Usage Example
-------------

.. code-block:: python

    from open_r1.tasks.reactions.iupac2smi import Iupac2Smiles

    # Initialize the task
    task = Iupac2Smiles(
        dataset_id_or_path="/path/to/your/dataset.csv",
    )

    # Load the dataset
    dataset = task.load()

    # Example of reward calculation
    completions = ["<answer>CCO</answer>"]
    solution = ["CCO"]
    rewards = task.accuracy_reward(completions, solution)

Data Format
-----------

The task expects data files formatted as CSV with the following columns:

- `IUPAC`: Column containing the chemical name in IUPAC format
- `SMILES`: Column containing the corresponding SMILES notation

Reward Functions
----------------

1. **Exact Match (accuracy_reward)**

   This function compares the predicted SMILES with the true SMILES:
   - +1.0 for an exact match
   - +0.2 for a correct canonical smile match
   - -0.5 for incorrect but valid SMILES
   - -1.0 for invalid SMILES

2. **Tanimoto Similarity (tanimoto_accuracy_reward)**

   Calculates rewards based on Tanimoto similarity between the predicted and actual SMILES:
   - +1.0 for a perfect match
   - Scaled proportionally to similarity for partial matches
   - -0.5 for very poor predictions (similarity < 0.3)
   - -1.0 for invalid SMILES

Task Example
------------

In this task, given an IUPAC name, the model predicts the corresponding SMILES representation:

.. code-block:: text

   Input: IUPAC: Ethanol
   Output: SMILES: CCO
