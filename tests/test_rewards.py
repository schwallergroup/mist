import pytest

from open_r1.tasks import ForwardReaction


@pytest.fixture
def reaction_task(tmp_path, monkeypatch):
    def fake_download(_data_path):
        return None

    monkeypatch.setattr(
        "open_r1.tasks.reactions.forward.download_data", fake_download
    )

    (tmp_path / "src-train.txt").write_text("CCO.C\n", encoding="utf-8")
    (tmp_path / "tgt-train.txt").write_text("CCOC\n", encoding="utf-8")
    (tmp_path / "src-test.txt").write_text("CC.C\n", encoding="utf-8")
    (tmp_path / "tgt-test.txt").write_text("CCC\n", encoding="utf-8")

    return ForwardReaction(dataset_id_or_path=str(tmp_path))


def test_load_uses_local_dataset_files(reaction_task):
    dataset = reaction_task.load()
    assert len(dataset["train"]) == 1
    assert len(dataset["test"]) == 1


def test_accuracy_reward_prefers_exact_answers(reaction_task):
    prompts = ["Reactants: CC.C"]
    correct = reaction_task.accuracy_reward(
        ["<think>valid reasoning</think><answer>CCC</answer>"],
        ["CCC"],
        prompts,
    )[0]
    incorrect = reaction_task.accuracy_reward(
        ["<think>valid reasoning</think><answer>CCO</answer>"],
        ["CCC"],
        prompts,
    )[0]
    invalid = reaction_task.accuracy_reward(
        ["<think>valid reasoning</think><answer>Xasd-</answer>"],
        ["CCC"],
        prompts,
    )[0]

    assert correct > incorrect
    assert incorrect >= invalid


def test_accuracy_reward_handles_missing_answer_tag(reaction_task):
    reward = reaction_task.accuracy_reward(
        ["<think>valid reasoning</think>CCC"],
        ["CCC"],
        ["Reactants: CC.C"],
    )[0]
    assert reward <= 0
