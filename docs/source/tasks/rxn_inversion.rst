Correct Reaction MCQA
===================

.. currentmodule:: open_r1.tasks.reactions

SmilesInversion
-------------

.. autoclass:: SmilesInversion
   :members:
   :show-inheritance:

Task Description
--------------
The Correct Reaction MCQA task is designed to identify the correct chemical reactions provided as SMILES when provided as a MCQA type question, where four choices are presented. The task was created using USPTO molecular dataset.Fake reactions were designed by inverted the position of the longest SMILES string in the reagents side of the reaction with a random product. 

Features
--------

- Provides testing groud for classification of varied reactions 
- Handles substrate, product, and catalyst SMILES data
- Provides detailed reasoning process for the classification choice


Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.reactions.mcqa_reaction_diff import SmilesInversion

    # Initialize the task
    task = SmilesInversion(
        dataset_id_or_path="path/to/reaction_mcqa_replacement_data",
    )

    # Load the dataset
    dataset = task.load()
   
    completions = ["<answer>A</answer>"]
    solution = ["prompt_true"]

Data Format
----------

The task expects data files in the following format:

- `prompt_true`: In the following reaction, the reagents are: [BEGIN_SMILES] CCCCOc1nc(N)c2nc(OC)n(CCCC3CCCCN3C(=O)OCc3ccccc3)c2n1 [END_SMILES], [BEGIN_SMILES] CCC[C@H](C)Oc1nc(N)c2nc(OC)[nH]c2n1 [END_SMILES], [BEGIN_SMILES] O=C(O)C(F)(F)F [END_SMILES], [BEGIN_SMILES] O=C(OCc1ccccc1)N1CCCC(CCCBr)C1 [END_SMILES] and the product is: [BEGIN_SMILES] CCC[C@H](C)Oc1nc(N)c2nc(OC)n(CCCC3CCCN(C(=O)OCc4ccccc4)C3)c2n1 [END_SMILES].
- `fake1`: This column contains a similar prompt as the "prompt_true", but contains a first example of a fake reaction.
- `fake2`: This column contains a similar prompt as the "prompt_true", but contains a second example of a fake reaction.
- `fake3`: This column contains a similar prompt as the "prompt_true", but contains a third example of a fake reaction.


Reward Functions
--------------

1. **Option String Matching (accuracy_reward)**
   
completions = ["<answer>A</answer>"]
solution    = ["prompt_true"]
options     = [["prompt_true","fake1","fake2","fake3"]]

rewards = task.accuracy_reward(completions, solution, options=options)

Task Example
-----------

.. code-block:: text

   Datta Example: 
   prompt_true, true_reaction, fake1, fake2, fake3

   "In the following reaction, the reagents are: [BEGIN_SMILES] CCCCOc1nc(N)c2nc(OC)n(CCCC3CCCCN3C(=O)OCc3ccccc3)c2n1 [END_SMILES], [BEGIN_SMILES] CCC[C@H](C)Oc1nc(N)c2nc(OC)[nH]c2n1 [END_SMILES], [BEGIN_SMILES] O=C(O)C(F)(F)F [END_SMILES], [BEGIN_SMILES] O=C(OCc1ccccc1)N1CCCC(CCCBr)C1 [END_SMILES] and the product is: [BEGIN_SMILES] CCC[C@H](C)Oc1nc(N)c2nc(OC)n(CCCC3CCCN(C(=O)OCc4ccccc4)C3)c2n1 [END_SMILES]."	
   
   "In the following reaction, the reagents are: [BEGIN_SMILES] CCOCC [END_SMILES], [BEGIN_SMILES] Cl [END_SMILES], [BEGIN_SMILES] FB(F)F [END_SMILES], [BEGIN_SMILES] Fc1ccccc1F [END_SMILES], [BEGIN_SMILES] [Li]CCCC [END_SMILES], [BEGIN_SMILES] CCCCCCC(O)Cc1cccc(F)c1F [END_SMILES], the condition is: [BEGIN_SMILES] C1COCC1 [END_SMILES], and the product is: [BEGIN_SMILES] CCCCCCC1CO1 [END_SMILES]."	
   
   "In the following reaction, the reagents are: [BEGIN_SMILES] O=Cc1ccc2[nH]ncc2c1 [END_SMILES], [BEGIN_SMILES] O=Cc1ccc2c(cnn2Cc2ccc(Cl)cc2C(F)(F)F)c1 [END_SMILES] and the product is: [BEGIN_SMILES] FC(F)(F)c1cc(Cl)ccc1CBr [END_SMILES]."
   
   "In the following reaction, the reagents are: [BEGIN_SMILES] CCN(C(C)C)C(C)C [END_SMILES], [BEGIN_SMILES] CCN=C=NCCCN(C)C [END_SMILES], [BEGIN_SMILES] Cl [END_SMILES], [BEGIN_SMILES] NC1CCCC(CNC(=O)OCc2ccccc2)C1 [END_SMILES], [BEGIN_SMILES] On1nnc2cccnc21 [END_SMILES], [BEGIN_SMILES] Cc1onc(-c2ncc(Cl)cc2Cl)c1C(=O)NC1CCCC(CNC(=O)OCc2ccccc2)C1 [END_SMILES], the condition is: [BEGIN_SMILES] CN(C)C=O [END_SMILES], and the product is: [BEGIN_SMILES] Cc1onc(-c2ncc(Cl)cc2Cl)c1C(=O)O [END_SMILES]."

   Reasoning: <think>
   The first reaction contains ...
   The second reaction would be wrong ...
   ...
   Therefore, the first reaction is the correct one ...
   </think>
   </answer>
   \\boxed{A}
   </answer>