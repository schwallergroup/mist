Base Task
=================

.. currentmodule:: open_r1.tasks.base

RLTask
------

.. autoclass:: RLTask
   :members:
   :show-inheritance:

Base Class Usage
---------------

The RLTask class serves as a base class for implementing specific reinforcement learning tasks.
Derived classes should implement task-specific logic while inheriting common functionality.

Example Implementation
~~~~~~~~~~~~~~~~~~~~

Here's a basic example of how to create a custom task:

.. code-block:: python

    from open_r1.tasks.base import RLTask

    class CustomTask(RLTask):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.question_template = "Custom question format: {}"

        def load(self) -> DatasetDict:
            # Implement dataset loading logic
            pass

        def accuracy_reward(self, completions, solution, **kwargs):
            # Implement reward calculation
            pass

Key Methods to Override
~~~~~~~~~~~~~~~~~~~~~~

- ``load()``: Load and prepare the dataset
- ``accuracy_reward()``: Define the reward function
- ``preprocess_response()``: Process model outputs

Common Attributes
~~~~~~~~~~~~~~~

- ``dataset_id_or_path``: Path to dataset
- ``dataset_splits``: Dataset split configuration
- ``system_prompt``: System instruction template
- ``response_print``: Response formatting template
