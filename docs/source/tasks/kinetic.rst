Kinetic Reaction Classification
===================

.. currentmodule:: open_r1.tasks.kinetic_data

KineticDataClassification
-------------

.. autoclass:: KineticDataClassification
   :members:
   :show-inheritance:

Task Description
--------------

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

    from open_r1.tasks.kinetic_data import KineticDataClassification

    # Initialize the task
    task = KineticDataClassification(
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
- ``path/to/kinetic/data/y_train/y_train_M1_M20_train_val_test_set_part_0.pkl``: Training labels (mechanism classes)
- ``path/to/kinetic/data/x_val/x1_val_M1_M20_train_val_test_set_part_0.pkl``: Validation data for initial catalyst concentrations
- ``path/to/kinetic/data/x_val/x2_val_M1_M20_train_val_test_set_part_0.pkl``: Validation data for time series measurements
- ``path/to/kinetic/data/y_val/y_val_M1_M20_train_val_test_set_part_0.pkl``: Validation labels (mechanism classes)

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


Experimental Method Details
------------------

Base Model Used
^^^^^^^^^^^^^^^
DeepSeek R1 Distill Qwen 1.5B

Job ID and the Run Name on WanDB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Job ID: 436856
- Run Name: grpo-436856-from_436856-DeepSeek-R1-Distill-Qwen-1.5B

Training Details
^^^^^^^^^^^^^^^^
- Directly train the base model using GRPO without supervised fine-tuning.
- Run 200 global steps.

Datasets used for RL
^^^^^^^^^^^^^^^^^^^^^^^^
I used the datasets from the article "Organic reaction mechanism classification using machine learning" by incorporating it into the prompt.
It contains 5,000,000 samples of kinetic profiles for reaction mechanisms transforming a substrate S into a product P, catalyzed by a catalyst, cat.
These were generated by solving the corresponding ordinary differential equations corresponding to 20 typical mechanisms.
These mechanisms belong to four distinct categories: 1) the core mechanism (M1), is the simplest Michaelis-Menten-type mechanism; 2) mechanisms with bicatalytic steps (M2-M5), involve either catalyst dimerization (M2 and M3), or a reaction between two different catalytic species (M4 and M5); 3) mechanisms with catalyst activation steps based on the core mechanism, where a precatalyst requires activation unimolecularly (M6), via substrate coordination (M7) or via ligand dissociation (M8); 4) mechanisms with a variety of catalyst deactivation steps from either catalytic intermediate of the core mechanism (M9-M20). 

For each samples, the dataset contains four data runs, and each data run contains:
- The initial concentration of the catalyst: 1 dimention
- Time series data: 21 dimention. The values are normalized to 0-1.
- Substrate data: 21 dimention. The values are normalized relative to the intial substrate concentration.
- Product data: 21 dimention. The values are normalized relative to the intial substrate concentration.
- The mechanism class (M1-M20)

Prompt
^^^^^^^
I included each reaction mechanisms explanation and four data runs in the prompt as follows.
```
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
```

Result
```````
- Initially, the accuracy reward (which includes exact match reward, category reward, class coverage reward and data run coverage reward as explained above) is around 0.02
