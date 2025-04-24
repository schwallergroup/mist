Crystal Relaxing
===================

.. currentmodule:: open_r1.tasks.relaxing

BinaryCompoundRelaxing
------------------

.. autoclass:: BinaryCompoundRelaxing
   :members:
   :show-inheritance:

Task Description
----------------

The `BinaryCompoundRelaxing` task guides a language model through multiple steps of structural relaxation on perturbed binary compounds. Given a serialized CIF description of a compound, the model must iteratively propose adjustments to reduce the internal energy, documenting its reasoning within <think> tags and outputting a final relaxed structure within <answer> tags.

Features
--------

- Reads and processes variations of SMILES notations from a dataset
- Converts varying SMILES strings into a canonical form
- Uses a template to guide the model in understanding how to format the response
- Features reward functions based on exact match and validity of SMILES

Usage Example
-------------

.. code-block:: python

    from open_r1.tasks.crystal_structure.relaxing import BinaryCompoundRelaxing

    # Initialize the task, pointing to a local dataset directory
    task = BinaryCompoundRelaxing(dataset_id_or_path="/path/to/cif_data")

    # Load datasets
    dataset = task.load()
    train_ds = dataset["train"]
    test_ds  = dataset["test"]

    # Compute accuracy rewards for an example prediction
    completions = ["<think>…</think><answer>M2S serialized_cif …</answer>"]
    solutions   = ["M2S serialized_cif …"]
    rewards = task.accuracy_reward(completions, solutions)


Data Format
-----------

The task reads paired text files with multi-line CIF records separated by blank lines:

- `src-train.txt / src-test.txt`: Each record is a serialized CIF string of a perturbed binary structure.
- `tgt-train.txt / tgt-test.txt`: Each record is the ground‑truth CIF string after DFT relaxation.

Reward Functions
----------------

1. **Accuracy Reward (accuracy_reward)**
   - Sends each predicted structure (extracted via <answer> tags) together with the ground truth to a scoring server at /compute_score.
   - Receives an energy‑based reward (e.g., +1 for lower energy, –4 for higher energy, –10 for invalid).

Task Example
------------

This example illustrates how the given non-canonical SMILES is converted to its canonical form:

.. code-block:: text

   Input:  unstable Crystal structure [M2S format]
   Output: relaxed Crystal structure [M2S format]