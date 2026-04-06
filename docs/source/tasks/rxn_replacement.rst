Reaction Replacement (RxR)
==========================

.. currentmodule:: open_r1.tasks.reactions.mcqa_reaction_diff

SmilesReplacement
-----------------

.. autoclass:: SmilesReplacement
   :members:
   :show-inheritance:

Task Description
----------------

Multiple-choice task where the model identifies the correct chemical reaction
from four options. Three distractors are created by replacing reagents with
similar molecules drawn from Enamine50k using Tanimoto similarity.

Dataset format: CSV with columns ``true_reaction``, ``fake1``, ``fake2``,
``fake3``.

Reward Functions
----------------

- **accuracy**: +1.0 for selecting the option that matches the gold reaction,
  0.0 otherwise.
- **format**: incremental scoring for ``<think>...</think><answer>...</answer>``
  structure with a single letter (A/B/C/D).
- **thinking_length**: +1.0 if the reasoning block is at least 100 words.
