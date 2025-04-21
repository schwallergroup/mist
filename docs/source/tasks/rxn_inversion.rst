Correct Reaction MCQA
===================

.. currentmodule:: open_r1.tasks.reactions

SmilesReplacement
-------------

.. autoclass:: SmilesReplacement
   :members:
   :show-inheritance:

Task Description
--------------
The Correct Reaction MCQA task is designed to identify the correct chemical reactions provided as SMILES when provided as a MCQA type question, where four choices are presented. The task was created using USPTO molecular dataset.Fake reactions were designed by inverted the position of the longest SMILEWS string in the reagents side of the reaction with a random product. 

Features
--------

- Provides testing groud for classification of varied reactions 
- Handles substrate, product, and catalyst SMILES data
- Provides detailed reasoning process for the classification choice


Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.reactions.mcqa_reaction_diff import SmilesReplacement

    # Initialize the task
    task = [TaskClassName](
        dataset_id_or_path="path/to/reaction_mcqa_replacement_data",
    )

    # Load the dataset
    dataset = task.load()
   
    completions = ["<answer>A</answer>"]
    solution = ["In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCS(=O)(=O)OCCBr [END_SMILES]."]

Data Format
----------

The task expects data files in the following format:

- `prompt_true`: This column contains the correct instance to be classified. In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCS(=O)(=O)OCCBr [END_SMILES].
- `fake1`: This column contains a similar prompt as the "prompt_true", but contains a first example of a fake reaction.
- `fake2`: This column contains a similar prompt as the "prompt_true", but contains a second example of a fake reaction.
- `fake3`: This column contains a similar prompt as the "prompt_true", but contains a third example of a fake reaction.


Reward Functions
--------------

1. **Partial String Matching (accuracy_reward)**
   
completions = ["<answer>A</answer>"]
solution = ["prompt_true"]
rewards = task.accuracy_reward(completions, solution, options=[["prompt_true", "fake1", "fake2", "fake3"]])

Task Example
-----------

.. code-block:: text

   Datta Example: 
   prompt_true, true_reaction, fake1, fake2, fake3

   "In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCS(=O)(=O)OCCBr [END_SMILES]."	"In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCS(=O)(=O)OCCBr [END_SMILES].",	
   
   "In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCC(=O)N1CCC(NS(=O)(=O)c2cn(C)c(C)n2)C1 [END_SMILES].",
   
   "In the following reaction, the reagents are: [BEGIN_SMILES] O=C1CC(C(=O)N2C[C@@H](F)C[C@H]2CO)CN1Cc1ccccc1 [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCS(=O)(=O)OCCBr [END_SMILES].",
   
   "In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCOc1ccccc1-c1csc(NC(=O)c2cc(OC)c(OC)cc2C)n1 [END_SMILES].",

   "Output: In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCS(=O)(=O)OCCBr [END_SMILES]."

   Reasoning: <think>
   The first reaction contains ...
   The second reaction would be wrong ...
   ...
   Therefore, the first reaction is the correct one ...
   </think>
   </answer>
   \\boxed{A}
   </answer>