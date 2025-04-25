Kinetic Reaction Classification
===================

.. currentmodule:: open_r1.tasks.kinetic_data

KineticDataClassification
-------------

.. autoclass:: KineticDataClassificationWithMetrics
   :members:
   :show-inheritance:

Task Description
--------------

This task is the same as the KineticDataClassification task, but it calculates some metrics, and provide it in the prompt so that the reasoning process is shorter and less complicated.
The Kinetic Reaction Classification task is designed to identify and classify chemical reaction mechanisms from kinetic data. The task presents experimental data from multiple runs with different initial conditions and requires the model to determine which of the 20 possible reaction mechanisms (M1-M20) best explains the observed behavior.

Features
--------

- Supports classification of 20 different reaction mechanisms (M1-M20)
- Processes normalized kinetic data from multiple experimental runs
- Handles substrate, product, and catalyst concentration data
- Provides detailed reasoning process for mechanism classification

Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.kinetic_data import KineticDataClassificationWithMetrics

    # Initialize the task
    task = KineticDataClassificationWithMetrics(
        dataset_id_or_path="path/to/kinetic/data",
    )

    # Load the dataset
    dataset = task.load()

    # Example of reward calculation
    completions = ["<think>Detailed reasoning...</think>\\boxed{M1}"]
    solution = ["M1"]
    rewards = task.accuracy_reward(completions, solution)

Data Format
----------

The task expects data files in the following format:

- ``path/to/kinetic/data/x_train/x1_train_M1_M20_train_val_test_set_part_0.pkl``: Training data for initial catalyst concentrations
- ``path/to/kinetic/data/x_train/x2_train_M1_M20_train_val_test_set_part_0.pkl``: Training data for time series measurements
- ``path/to/kinetic/data/y_train/y_train_M1_M20_train_val_test_set.pkl``: Training labels (mechanism classes)
- ``path/to/kinetic/data/x_val/x1_val_M1_M20_train_val_test_set.pkl``: Validation data for initial catalyst concentrations
- ``path/to/kinetic/data/x_val/x2_val_M1_M20_train_val_test_set.pkl``: Validation data for time series measurements
- ``path/to/kinetic/data/y_val/y_val_M1_M20_train_val_test_set.pkl``: Validation labels (mechanism classes)

Reward Functions
--------------
The accuracy reward is calculated by the weighted sum of the following rewards.

1. **Exact Match (accuracy_reward)**
   
   Calculates binary rewards based on exact mechanism classification:
   - Correct mechanism classification: 1.0
   - Incorrect mechanism classification: 0.0

2. **Class Coverage Reward (accuracy_reward)**

   Calculates the reward based on the percentage of the 20 reaction classes that the model considered during the reasoning.
   - If the model considered all 20 reaction classes, the reward is 1.0.

3. **Data Coverage Reward (accuracy_reward)**

   Calculates the reward based on the percentage of the 4 data runs that the model considered during the reasoning.
   - If the model considered all 4 data runs, the reward is 1.0.

4. **Category Match (accuracy_reward)**

   Calculates the reward based on the answer that the model gave is in the same category as the ground truth.
   - If the model gave the answer that is in the same category as the ground truth, the reward is 1.0.
   - For example, if the ground truth is M3, and the model gave M2, the reward is 1.0.
   - Category is defined as follows:
      - M1: Core mechanism
      - M2-M5: Bicatalytic steps
      - M6-M8: Activation steps
      - M9-M20: Deactivation steps

The format reward is calculated by the following reward.
5. **Format Reward (format_reward)**

   Calculates the reward based on the format of the answer.
   - If the answer includes ``<think>`` and ``\\boxed{}``, the reward is 1.0.
   - Otherwise, the reward is 0.0.

   **Note:**
   - This format is intended to train DeepSeek R1 Distill Qwen.
   - There is the suggestion to use the format in the prompt in their transformer website. I once tried to use ``<think>`` and ``<answer>``, but the format reward did not start to increase in the first 50 global steps.


Task Example
-----------

.. code-block:: text

   Input: 
   # Data Run 1
   - Initial concentration of catalyst: 0.1
   - Initial concentration of substrate: 1.0
   - Time series data: [0, 1, 2, 3]
   - Substrate data: [1.0, 0.8, 0.6, 0.4]
   - Product data: [0.0, 0.2, 0.4, 0.6]
   
   Output: M1
   Reasoning: <think>
   The reaction shows first-order kinetics with respect to substrate...
   The mechanism must involve a single catalytic site...
   Therefore, M1 is the most likely mechanism.
   </think>
   \\boxed{M1}

Reward Functions
--------------
The accuracy reward is calculated by the weighted sum of the following rewards. The weights are set to 0.5, 0.2, 0.2 and 0.1 for exact match, class coverage reward, data coverage reward and category match respectively.

1. **Exact Match (accuracy_reward)**
   
   Calculates binary rewards based on exact mechanism classification:
   
   - Correct mechanism classification: 1.0
   
   - Incorrect mechanism classification: 0.0

2. **Class Coverage Reward (accuracy_reward)**

   Calculates the reward based on the percentage of the 20 reaction classes that the model considered during the reasoning.
   
   - If the model considered all 20 reaction classes, the reward is 1.0.

3. **Data Coverage Reward (accuracy_reward)**

   Calculates the reward based on the percentage of the 4 data runs that the model considered during the reasoning.
   
   - If the model considered all 4 data runs, the reward is 1.0.

4. **Category Match (accuracy_reward)**

   Calculates the reward based on the answer that the model gave is in the same category as the ground truth.
   
   - If the model gave the answer that is in the same category as the ground truth, the reward is 1.0.
   
   - For example, if the ground truth is M3, and the model gave M2, the reward is 1.0.
   
   - Category is defined as follows:
      - M1: Core mechanism
  
      - M2-M5: Bicatalytic steps
  
      - M6-M8: Activation steps
  
      - M9-M20: Deactivation steps


The format reward is calculated by the following reward.

1. **Format Reward (format_reward)**

   Calculates the reward based on the format of the answer.
   
   - If the answer includes ``<think>`` and ``\\boxed{}``, the reward is 1.0.
   
   - Otherwise, the reward is 0.0.

   **Note:**
   
   - This format is intended to train DeepSeek R1 Distill Qwen.
   
   - There is the suggestion to use the format in the prompt in their transformer website. I once tried to use ``<think>`` and ``<answer>``, but the format reward did not start to increase in the first 50 global steps.


Task Example
-----------

.. code-block:: text

   Input: 
   # Data Run 1
   - Initial concentration of catalyst: 0.1
   - Initial concentration of substrate: 1.0
   - Time series data: [0, 1, 2, 3]
   - Substrate data: [1.0, 0.8, 0.6, 0.4]
   - Product data: [0.0, 0.2, 0.4, 0.6]
   
   Output: M1
   Reasoning: <think>
   The reaction shows first-order kinetics with respect to substrate...
   The mechanism must involve a single catalytic site...
   Therefore, M1 is the most likely mechanism.
   </think>
   \\boxed{M1}


Experimental Method Details
------------------

Base Model Used
^^^^^^^^^^^^^^^
DeepSeek R1 Distill Qwen 1.5B

Job ID and the Run Name on WanDB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Job ID: 
  - 436558
  - 438902
  
- Run Name: 
  - grpo-436856-from_436856-DeepSeek-R1-Distill-Qwen-1.5B
  - grpo-438902-from_436856-DeepSeek-R1-Distill-Qwen-1.5B

Training Details
^^^^^^^^^^^^^^^^
- The base model was trained directly using GRPO without supervised fine-tuning.
  
- Training was conducted for 1775 global steps.

Datasets used for RL
^^^^^^^^^^^^^^^^^^^^^^^^
Datasets used in this task is based on the study "Organic reaction mechanism classification using machine learning".
Larrosa, Igor (2022). Training, validation and test set for M1-M20. University of Manchester. Dataset. https://doi.org/10.48420/16965292.v2

It originally contains 5,000,000 samples of kinetic profiles for reaction mechanisms transforming a substrate S into a product P, catalyzed by a catalyst, cat, and 1000 sample of 500000 samples were used for GRPO.
Each profile was generated by solving the corresponding ordinary differential equations corresponding to 20 typical mechanisms.

These mechanisms belong to four distinct categories: 

1) the core mechanism (M1), is the simplest Michaelis-Menten-type mechanism; 

2) mechanisms with bicatalytic steps (M2-M5), involve either catalyst dimerization (M2 and M3), or a reaction between two different catalytic species (M4 and M5); 
   
3) mechanisms with catalyst activation steps based on the core mechanism, where a precatalyst requires activation unimolecularly (M6), via substrate coordination (M7) or via ligand dissociation (M8); 
   
4) mechanisms with a variety of catalyst deactivation steps from either catalytic intermediate of the core mechanism (M9-M20). 


For each samples, the dataset contains four data runs, and each data run contains:

- The initial concentration of the catalyst: 1-dimentional, normalized.

- Time series data: 21 dimentional. The values are normalized to 0-1.

- Substrate data: 21 dimentional, normalized relative to the intial substrate concentration.

- Product data: 21 dimentional, normalized relative to the intial substrate concentration.

- The mechanism class: One of M1-M20.

The dataset is incorporated into the prompt as follows.


Prompt
^^^^^^^
I included each reaction mechanisms explanation and the metrics that were calculated in advance in the prompt as follows.

.. code-block:: text
      
   Reason and estimate the reaction class for the following reaction.
   The possible reaction classes are M1 to M20 indicated as follows.
   Please begin your response with "<think>", then provide a detailed, step-by-step reasoning process (including any intermediate reflections or re-evaluations), 
   then end with </think>, and finally put your final answer within \\boxed{{}} tags, for example \\boxed{{M1}}.

   # Possible reaction classes
   // M1 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2

   // M2 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2cat<=>cat2;k3,k-3

   // M3 Mechanism
   S+cat2<=>((cat)2S);k1,k-1|((cat)2S)<=>P+cat2;k2,k-2|2cat<=>cat2;k3,k-3

   // M4 Mechanism
   X+catS<=>S+cat;k1,k-1|X+catS<=>P+cat;k2,k-2

   // M5 Mechanism
   S+cat<=>catS;k1,k-1|catS+cat<=>catP;k2,k-2|catP<=>P+cat;k3,k-3

   // M6 Mechanism
   cat<=>cat*;k1,0|S+cat*<=>cat*S;k1,k-1|cat*S<=>P+cat*;k2,k-2

   // M7 Mechanism
   S+cat<=>catS;k1,k-1|S+catS<=>catS2;k3,k-3|catS<=>P+cat;k2,k-2

   // M8 Mechanism
   S+cat*<=>cat*S;k1,k-1|cat*S<=>P+cat*;k2,k-2|cat+L<=>cat*;k3,k-3

   // M9 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat<=>inactive cat;k3,0

   // M10 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|inhibitor+cat<=>inactive catI;k3,0

   // M11 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|S+cat<=>inactive catS;k-3,0

   // M12 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|P+cat<=>inactive catP;k-3,0

   // M13 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2cat<=>inactive cat2;k-3,0

   // M14 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|catS<=>inactive catS;k-3,0

   // M15 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|inhibitor+catS<=>inactive catSI;k-3,0

   // M16 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|S+catS<=>inactive catS2;k-3,0

   // M17 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|P+catS<=>inactive catSP;k-3,0

   // M18 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2catS<=>inactive cat2S2;k-3,0

   // M19 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat+catS<=>inactive cat2S;k3,0

   // M20 Mechanism
   S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat<=>inactive cat;k3,0|catS<=>inactive catS;k4,0

   # Data Run 1
   - Initial concentration of catalyst (normalized to [S]0): {{run_1[initial_concentration_of_catalyst]}}
   - Initial concentration of substrate (normalized to [S]0): {{run_1[substrate_data][0]}}
   - Initial concentration of ES: 0.0
   - Initial concentration of product (normalized to [S]0): {{run_1[product_data][0]}}
   - Time_data (normalized, unitless): {{run_1[time_data]}}
   - Substrate_data (normalized to [S]0): {{run_1[substrate_data]}}
   - Product_data (normalized to [S]0): {{run_1[product_data]}}

   # Data Run 2
   - Initial concentration of catalyst (normalized to [S]0): {{run_2[initial_concentration_of_catalyst]}}
   - Initial concentration of substrate (normalized to [S]0): {{run_2[substrate_data][0]}}
   - Initial concentration of ES: 0.0
   - Initial concentration of product (normalized to [S]0): {{run_2[product_data][0]}}
   - Time_data (normalized, unitless): {{run_2[time_data]}}
   - Substrate_data (normalized to [S]0): {{run_2[substrate_data]}}
   - Product_data (normalized to [S]0): {{run_2[product_data]}}

   # Data Run 3
   - Initial concentration of catalyst (normalized to [S]0): {{run_3[initial_concentration_of_catalyst]}}
   - Initial concentration of substrate (normalized to [S]0): {{run_3[substrate_data][0]}}
   - Initial concentration of ES: 0.0
   - Initial concentration of product (normalized to [S]0): {{run_3[product_data][0]}}
   - Time_data (normalized, unitless): {{run_3[time_data]}}
   - Substrate_data (normalized to [S]0): {{run_3[substrate_data]}}
   - Product_data (normalized to [S]0): {{run_3[product_data]}}

   # Data Run 4
   - Initial concentration of catalyst (normalized to [S]0): {{run_4[initial_concentration_of_catalyst]}}
   - Initial concentration of substrate (normalized to [S]0): {{run_4[substrate_data][0]}}
   - Initial concentration of ES: 0.0
   - Initial concentration of product (normalized to [S]0): {{run_4[product_data][0]}}
   - Time_data (normalized, unitless): {{run_4[time_data]}}
   - Substrate_data (normalized to [S]0): {{run_4[substrate_data]}}
   - Product_data (normalized to [S]0): {{run_4[product_data]}}

   <think>



Result
------

Transition of Rewards
^^^^^^^^^^^^
- The format reward increases from around 0.3 at the beggining to 0.99 at 300 global steps.

- The accuracy reward (which includes exact match reward, category reward, class coverage reward and data run coverage reward as explained above) is around 0.03 at the beggining, and reaches around 0.10 around 300 global steps. It keeps increasing to around 0.15 at 1000 global steps, and then it seeems to reach plateau around 0.15.
  The improvement until 300 global steps seems to correspond to the improvement in the format reward, both reward increase rouphly threehold, suggesting that the accuracy reward of the responses with correct format remains almost unchanged from the begginning.

- Looking into the exact match reward, category reward, class coverage reward and data run coverage reward, they are all improved until 300 global steps, corresponding the improvement in the format reward. However, after 300 global steps, three rewards except the category reward remains almost the same. The exact match reward remains around 0.05, which is the same reward that should be obtained from random responses.

- Only the class coverage reward out of the four rewards seems to be increasing after 300 global steps, and it reaches aroud 0.5 at 1000 global steps. 

The Confusion Matrix about the model's prediction after training
^^^^^^^^^^^^^^^^^^^^^^
at 175 global steps
It seems to be biased toward M1.

at 1775 global steps
It seems to be biased toward M9-M20. I think this 

Sample Responses
^^^^^^^^^^^^^^^^
The following is the final response on the log. Several observations can be made:

- The overall structure of the response is good. It starts by observing the data and links the observed features to potential reaction mechanisms.

- However, some weakness in the model's reasoning ability are noticible.
  
  - For example, it initially claims that M20 isn't the best fit, but it concludes that M20 is the most likely candidate without clear explantion for this change in opinion.
  
  - It confuses M20 and M4, which are distinct mechanisms.
  
  - It infers that from the observed data suggest a bimolecular reaction, despite the fact that M1-M20 are not bimolecular reactions.

- Hallucinations are also observed.
  
  - It mentioned "Drosophila formation" in the reasoning traces, which is not a recognized term in chemistry.


.. code-block:: text

   <think> Okay, so I'm trying to determine the reaction class of the given reaction mechanism, M1 to M20. I have data runs 1, 2, and 3, along with the data for each. I need to analyze the behavior of these data runs against each mechanism.

   Looking at Data Run 1 first: The initial concentration of the catalyst is [S]0.8 in normalized units. The substrate levels are decreasing, and the product also decreases. The product's peak is around t=2.4426, which isn't too high, so maybe it's the beginning phase of a reaction. The product levels are increasing initially and then perhaps stabilizing.

   In Data Run 2, the initial catalyst is [S]0.0382. There's a significant boost in product levels almost immediately, then the substrate decreases. The product seems to peak higher and then stabilize, which might indicate a bimolecular reaction or a step where some conversion occurs early.

   Data Run 3 has a lower initial catalyst concentration. Products start at even lower levels, but they rise higher. The product's peak is around t=2.6641, longer than Data Run 1 but similar to 2.

   Data Run 4 has a high initial catalyst concentration (0.382) and a lot of product quickly. The catalyst is used up, and product increases significantly at first, then decreases. This is similar to a boptimized reaction mechanism where enzymes might be overused, leading to some product formation and then inhibition.

   Looking at all the data points, it seems like these reaction mechanisms are related to a bimolecular reaction, maybe without complex steps like Drosophila formation or cooperativity. Each run starts with a catalyst in high concentrations, leading to early product formation and substrate depletion.

   Also, in Data Run 4, the product increases significantly before t=5, then decreases again. This could be indicative of a stepwise pathway where some conversion happens first before the rate decreases, which could be a three-component system or an S₄ step.

   Comparing all to M1 to M20, mechanisms that would show such early product peaks and substrate depletion are likely to be those that process the substrate and catalyst early on, without waiting for both to reach their full capacity.

   M1 might involve a catalyst first forming ES, which then deactivates the catalyst and becomes E, leading to product formation. M2 and M5 involve catalyst activation followed by E activation, again showing early peaks. M3 might involve higher certainty or steps without full catalyst activation.

   M4 seems to involve more complex steps, possibly allowing catalyst deactivation in the presence of product, leading to product accumulation quickly.

   Based on the initial data, M2, M3, M4 could fit, but without seeing the entire data set, it's hard to be certain. However, considering the presence of higher substrate levels early in Data Run 4, M20 (M4) isn't the best fit. M4 has product peaks and a complex catalyst turnover path.

   Another consideration is the presence of multiple steps. M1 might involve multiple transformations, but Data Run 4 seems to have a different pattern than other runs.

   After evaluating all, it seems M20 is the most likely candidate, as it involves a complex catalytic cycle with product formation early.
   </think>
