Forward Reaction
===================

.. currentmodule:: open_r1.tasks.forward

ForwardReaction
-------------

.. autoclass:: ForwardReaction
   :members:
   :show-inheritance:

Task Description
--------------

The ForwardReaction task handles chemical reaction predictions, specifically focusing on
predicting product SMILES notation given reactant SMILES.

Features
--------

- Automatic data downloading and preparation
- SMILES notation processing
- Multiple reward functions (exact match and Tanimoto similarity)
- Chemical structure validation using RDKit

Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.reactions.forward import ForwardReaction

    # Initialize the task
    task = ForwardReaction(
        dataset_id_or_path="path/to/data",
        dataset_splits="train"
    )

    # Load the dataset
    dataset = task.load()

    # Example of reward calculation
    completions = ["<answer>CC(=O)O</answer>"]
    solution = ["CC(=O)O"]
    rewards = task.accuracy_reward(completions, solution)

Data Format
----------

The task expects data files in the following format:

- ``src-train.txt``: Reactant SMILES
- ``tgt-train.txt``: Product SMILES
- ``src-test.txt``: Test set reactants (optional)
- ``tgt-test.txt``: Test set products (optional)

Reward Functions
--------------

1. **Exact Match (accuracy_reward)**
   
   Returns rewards based on exact SMILES matches:
   - +1.0 for exact match
   - -0.5 for incorrect but valid SMILES
   - -1.0 for invalid SMILES

2. **Tanimoto Similarity (tanimoto_accuracy_reward)**
   
   Returns rewards based on molecular similarity:
   - +1.0 for perfect match
   - Scaled value based on Tanimoto coefficient
   - Negative values for poor predictions


Task Example
-----------

.. image:: _static/forward_reaction_example.png
   :width: 200
   :align: center
   :alt: Reaction scheme showing acetyl chloride reacting with ammonia to form acetamide

In this task, the model gets the SMILES string of the reactants, and the task is to predict the product:

.. code-block:: text

   Input: CC(=O)Cl.N
   Output: CC(=O)N