Canonicalize SMILES
===================

.. currentmodule:: open_r1.tasks.canonical

CanonicalizeSmiles
------------------

.. autoclass:: CanonicalizeSmiles
   :members:
   :show-inheritance:

Task Description
----------------

The `CanonicalizeSmiles` task aims to convert non-canonical SMILES strings into their canonical form. This task is essential for ensuring consistent and standardized representations of chemical structures in computational chemistry.

Features
--------

- Reads and processes variations of SMILES notations from a dataset
- Converts varying SMILES strings into a canonical form
- Uses a template to guide the model in understanding how to format the response
- Features reward functions based on exact match and validity of SMILES

Usage Example
-------------

.. code-block:: python

    from open_r1.tasks.reactions.canonical import CanonicalizeSmiles

    # Initialize the task
    task = CanonicalizeSmiles(
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

- `SMILES_variant1`: Column containing non-canonical SMILES strings
- `SMILES`: Column containing the canonical SMILES notation (correct answer)

Reward Functions
----------------

1. **Exact Match (accuracy_reward)**

   This function rewards predictions based on exact canonical SMILES matches:
   - +1.0 for an exact match
   - +0.2 for correct canonical SMILES upon conversion
   - -0.5 for incorrect but valid SMILES
   - -1.0 for invalid SMILES

Task Example
------------

This example illustrates how the given non-canonical SMILES is converted to its canonical form:

.. code-block:: text

   Input: Non-canonical SMILES: [Input_SMILES]
   Output: Canonical SMILES: [Canonical_SMILES]
