Conditional Material Generation (CMG)
======================================

.. currentmodule:: open_r1.tasks.condmatgen.condmatgen

ConditionalMaterialGeneration
-----------------------------

.. autoclass:: ConditionalMaterialGeneration
   :members:
   :show-inheritance:

Task Description
----------------

Given a set of chemical elements, the model proposes a novel crystalline
compound (element list and space group number). The model wraps its reasoning
in ``<think>...</think>`` tags and its answer in ``<answer>...</answer>`` tags.

Reward Functions
----------------

- **accuracy**: multi-component scoring including SMACT validity, element
  precision, space group validity, and novelty bonus.
- **format**: checks presence and ordering of think/answer tags, penalizes
  short reasoning.
