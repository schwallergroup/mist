[Task Name]
===================

.. currentmodule:: open_r1.tasks.[module_name]

[TaskClassName]
-------------

.. autoclass:: [TaskClassName]
   :members:
   :show-inheritance:

Task Description
--------------

[Brief description of what the task does and its main objective]

Features
--------

- [Key feature 1]
- [Key feature 2]
- [Key feature 3]
- [Key feature 4]

Usage Example
-----------

.. code-block:: python

    from open_r1.tasks.[module_path] import [TaskClassName]

    # Initialize the task
    task = [TaskClassName](
        dataset_id_or_path="path/to/data",
        dataset_splits="train"
    )

    # Load the dataset
    dataset = task.load()

    # Example of reward calculation
    completions = ["<answer>[example_answer]</answer>"]
    solution = ["[example_solution]"]
    rewards = task.accuracy_reward(completions, solution)

Data Format
----------

The task expects data files in the following format:

- ``[file1]``: [Description]
- ``[file2]``: [Description]
- ``[file3]``: [Description] (optional)
- ``[file4]``: [Description] (optional)

Reward Functions
--------------

1. **[Reward Function 1] ([method_name])**
   
   [Description of reward function 1]:
   - [Condition 1]: [Reward value]
   - [Condition 2]: [Reward value]
   - [Condition 3]: [Reward value]

2. **[Reward Function 2] ([method_name])**
   
   [Description of reward function 2]:
   - [Condition 1]: [Reward value]
   - [Condition 2]: [Reward value]
   - [Condition 3]: [Reward value]

Task Example
-----------

.. image:: _static/[task_name]_example.[extension]
   :width: 200
   :align: center
   :alt: [Description of the image]

[Brief explanation of the task example]

.. code-block:: text

   Input: [example_input]
   Output: [example_output]