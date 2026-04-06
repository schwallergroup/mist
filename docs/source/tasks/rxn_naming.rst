Reaction Naming (RxN)
=====================

.. currentmodule:: open_r1.tasks.reactions.reaction2name

Smiles2Name
-----------

.. autoclass:: Smiles2Name
   :members:
   :show-inheritance:

Task Description
----------------

Classification task where the model predicts the reaction type from a SMILES
reaction string. The model chooses from ten categories: Acylation, Aromatic
Heterocycle Formation, C-C Coupling, Deprotection, Functional Group Addition,
Functional Group Interconversion, Heteroatom Alkylation and Arylation,
Miscellaneous, Protection, and Reduction.

Dataset format: CSV with columns ``REACTION_PROMPT`` and ``CLASS``.

Reward Functions
----------------

- **accuracy**: +1.0 for an exact class match, +0.2 for selecting any valid
  class, -0.2 for an invalid answer. An additional +0.1 bonus is awarded if
  the reasoning block mentions a valid class name.
- **format**: ratio of content inside matched
  ``<think>...</think><answer>...</answer>`` tags to total completion length.
