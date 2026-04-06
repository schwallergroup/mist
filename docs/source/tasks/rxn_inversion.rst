Reaction Inversion (RxI)
========================

.. currentmodule:: open_r1.tasks.reactions.mcqa_inversion

SmilesInversion
---------------

.. autoclass:: SmilesInversion
   :members:
   :show-inheritance:

Task Description
----------------

Multiple-choice task where the model identifies the correct chemical reaction
from four options. Three distractors are created by inverting
(swapping) the reagent ordering in the original reaction SMILES.

Dataset format: CSV with columns ``true_reaction``, ``fake1``, ``fake2``,
``fake3``.

Reward Functions
----------------

- **accuracy**: +1.0 for selecting the option that matches the gold reaction,
  0.0 otherwise.
- **format**: incremental scoring for ``<think>...</think><answer>...</answer>``
  structure with a single letter (A/B/C/D).
- **thinking_length**: +1.0 if the reasoning block is at least 100 words.
