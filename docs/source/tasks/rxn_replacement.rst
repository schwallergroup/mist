Correct Inversion Reaction MCQA
===================

.. currentmodule:: open_r1.tasks.reactions

SmilesReplacement
-------------

.. autoclass:: SmilesReplacement
   :members:
   :show-inheritance:

Task Description
--------------
The Correct Inversion Reaction MCQA task is designed to identify the correct chemical reactions provided as SMILES when provided as a MCQA type question, where four choices are presented. The task was created using USPTO molecular dataset as basis. Fake reactions were designed by randomly replacement a SMILES string in the reaction with the most similar compound from a randomly sampled pool of SMILES from the Enamine50k dataset. The similarities were computed with Tanimoto calculations.

Features
--------

- Provides testing groud for classification of varied reactions 
- Handles substrate, product, and catalyst SMILES data
- Provides detailed reasoning process for the classification choice


Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.reactions.mcqa_inversion import SmilesReplacement

    # Initialize the task
    task = SmilesReplacement(
        dataset_id_or_path="path/to/reaction_mcqa_inversion_data",
    )

    # Load the dataset
    dataset = task.load()
   
    completions = ["<answer>A</answer>"]
    solution = ["prompt_true"]

Data Format
----------

The task expects data files in the following format:

- `prompt_true`: "In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCS(=O)(=O)OCCBr [END_SMILES]."	"In the following reaction, the reagents are: [BEGIN_SMILES] CCS(=O)(=O)Cl [END_SMILES], [BEGIN_SMILES] OCCBr [END_SMILES], the conditions are: [BEGIN_SMILES] CCN(CC)CC [END_SMILES], [BEGIN_SMILES] CCOCC [END_SMILES], and the product is: [BEGIN_SMILES] CCS(=O)(=O)OCCBr [END_SMILES]".
- `fake1`: This column contains a similar prompt as the "prompt_true", but contains a first example of a fake reaction.
- `fake2`: This column contains a similar prompt as the "prompt_true", but contains a second example of a fake reaction.
- `fake3`: This column contains a similar prompt as the "prompt_true", but contains a third example of a fake reaction.


Reward Functions
--------------

1. **Option Matching (accuracy_reward)**
   
completions = ["<answer>A</answer>"]
solution    = ["prompt_true"]
options     = [["prompt_true","fake1","fake2","fake3"]]

rewards = task.accuracy_reward(completions, solution, options=options)

Task Example
-----------

.. code-block:: text

   Datta Example: 
   prompt_true, fake1, fake2, fake3

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