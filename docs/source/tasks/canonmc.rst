Canonicalize SMILES Multiple Choice (MCQA)
==========================================

.. currentmodule:: open_r1.tasks.canonicalizesmilesmcqa

CanonicalizeSmilesMCQA
----------------------

.. autoclass:: CanonicalizeSmilesMCQA
   :members:
   :show-inheritance:

Task Description
----------------

The `CanonicalizeSmilesMCQA` task provides a multiple-choice question format for converting non-canonical SMILES to their canonical form. The task presents a list of SMILES options, and the model selects the correct canonical SMILES representation from these options.

Features
--------

- Loads and processes SMILES notation variations from a dataset
- Converts non-canonical SMILES into canonical form through multiple-choice questions
- Generates problem-specific prompts with multiple answer options
- Provides reward evaluations based on selected options

Usage Example
-------------

.. code-block:: python

    from open_r1.tasks.canon_mcqa import CanonicalizeSmilesMCQA

    # Initialize the task
    task = CanonicalizeSmilesMCQA(
        dataset_id_or_path="/path/to/your/dataset.csv",
    )

    # Load the dataset
    dataset = task.load()

    # Example of reward calculation
    completions = ["<answer>A</answer>"]
    solution = ["canonical_SMILES_here"]
    rewards = task.accuracy_reward(completions, solution, options=[["OptionA", "OptionB", "OptionC", "canonical_SMILES_here"]])

Data Format
-----------

The task expects data files formatted as CSV with the following columns:

- `SMILES_variant1`: Column containing non-canonical SMILES strings
- `SMILES`: Column containing the canonical SMILES notation (correct answer)
- `SMILES_variant2`, `SMILES_variant3`, `SMILES_variant4`: Additional non-canonical SMILES strings for forming multiple-choice options

Reward Functions
----------------

1. **Exact Match (accuracy_reward)**

   This function evaluates the correctness of selections based on the provided options:
   - +1.0 for selecting the correct option matching the canonical SMILES
   - 0.0 for incorrect selections

Task Example
------------

This example illustrates the multiple-choice selection process for canonicalizing SMILES:

.. code-block:: text

   Input: Non-canonical SMILES: [Input_SMILES]
   Options:
   A. [Option_SMILES_1]
   B. [Option_SMILES_2]
   C. [Option_SMILES_3]
   D. [Canonical_SMILES]
   Output: <answer>D</answer>
