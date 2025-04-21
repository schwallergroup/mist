Reaction Name Classification
===================

.. currentmodule:: open_r1.tasks.reactions

Smiles2Name
-------------

.. autoclass:: Smiles2Name
   :members:
   :show-inheritance:

Task Description
--------------
The Reaction Name Classification task is designed to identify and classifify chemical reactions provided as SMILES to one of the different reaction classes. The task was created using USPTO Reaction dataset and Rxn-INSIGHT librairy to label chemical reactions. 

Features
--------

- Supports classification of varied reactions 
- Handles substrate, product, and catalyst SMILES data
- Provides detailed reasoning process for the classification choice

Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.reactions.reaction2name import Smiles2Name

    # Initialize the task
    task = [TaskClassName](
        dataset_id_or_path="path/to/reaction_class_data",
    )

    # Load the dataset
    dataset = task.load()

    # Example of reward calculation
    completions = ["<answer>A</answer>"]
    solution = ["Formation of Sulfonic Esters"]
    rewards = task.accuracy_reward(completions, solution, options=[["OptionA", "OptionB", "OptionC", "canonical_SMILES_here"]])

Data Format
----------

The task expects data files in the following format:

- `REACTION`: Column containing the reaciton SMILES as "reagent1". ... "reagentN">"solvent"."catalyst"."ligands">"product1". ... "productN"
- `NAME`: Column with the name of the reaction

Reward Functions
--------------

1. **Partial String Matching (accuracy_reward)**
   
This method iterates over each pair of generated completion and true solution to compute a reward. If the processed answer is empty or “none,” it immediately assigns a penalty of –1 and skips further checks. Otherwise, it uses a sequence‐matching algorithm to calculate a similarity ratio and awards a full reward of +1 (logging correct matches) for ratios above 0.9, a partial reward of 0.2 for ratios above 0.8, and a penalty of –0.5 for anything lower. 

Task Example
-----------

.. code-block:: text

   Datta Example: 
   REACTION, NAME
   CCS(=O)(=O)Cl.OCCBr>>CCS(=O)(=O)OCCBr, Formation of Sulfonic Esters
   
   Output: CCS(=O)(=O)Cl.OCCBr>>CCS(=O)(=O)OCCBr
   Reasoning: <think>
   The reaction shows the presence of ...
   The product formed is ...
   Therefore, the reaction should be Sulfonic Esters Formation
   </think>
   </answer>
   \\boxed{Formation of Sulfonic Esters}
   </answer>