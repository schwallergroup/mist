from open_r1.tasks.kinetic_data.kinetic_data_classification import KineticDataClassification

response_wrong_format = """
<think>Reasoning about the mechanism and data coverage.</think>
<answer>M20</answer>
"""

response_correct_format = """
<think>Reasoning about M20 using data 1, data 2, and data 4 while comparing M1 and M20.</think>
\\boxed{M20}
"""


def build_task():
    return KineticDataClassification(dataset_id_or_path="unused")


def test_format_reward():
    task = build_task()
    assert task.format_reward([response_wrong_format, response_correct_format]) == [0.0, 1.0]


def test_accuracy_reward():
    task = build_task()
    rewards = task.accuracy_reward(
        [response_wrong_format, response_correct_format],
        ["M20", "M20"],
    )
    assert rewards == [0.0, 0.71]


def test_preprocess_response():
    task = build_task()
    assert task.preprocess_response(response_correct_format) == "M20"
    assert task.preprocess_response(response_wrong_format) == "NONE"


def test_class_coverage_reward():
    task = build_task()
    assert task.class_coverage_reward("M1 M2 M2 M20") == 3 / 20


def test_data_coverage_reward():
    task = build_task()
    assert task.data_coverage_reward("data 1, data 2, and data 2") == 2 / 4
