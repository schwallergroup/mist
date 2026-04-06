Reaction True/False (RxTF)
==========================

.. currentmodule:: open_r1.tasks.reactions.reaction_truefalse

ReactionTrueFalse
-----------------

.. autoclass:: ReactionTrueFalse
   :members:
   :show-inheritance:

Task Description
----------------

Binary classification task where the model determines whether a given chemical
reaction (in SMILES notation) is valid or not. The model answers "True" or
"False".

Dataset format: CSV with columns ``reaction`` and ``label`` (values: ``true``
or ``false``).

Reward Functions
----------------

- **accuracy**: uses majority-vote over all "true"/"false" mentions in both
  ``<answer>`` and ``<think>`` spans. Awards +1.0 if the predicted label
  matches the gold label, -0.5 otherwise.
- **format**: ratio of content inside matched
  ``<think>...</think><answer>...</answer>`` tags to total completion length.
