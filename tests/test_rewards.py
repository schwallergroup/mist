import csv
from pathlib import Path

import pytest

from open_r1.tasks import ForwardReaction
from open_r1.tasks.reactions.mcqa_inversion import SmilesInversion
from open_r1.tasks.reactions.mcqa_reaction_diff import SmilesReplacement
from open_r1.tasks.reactions.reaction2name import Smiles2Name
from open_r1.tasks.reactions.reaction_truefalse import ReactionTrueFalse


@pytest.fixture
def reaction_task(tmp_path, monkeypatch):
    def fake_download(_data_path):
        return None

    monkeypatch.setattr("open_r1.tasks.reactions.forward.download_data", fake_download)

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


# ---------------------------------------------------------------------------
# RxI — SmilesInversion
# ---------------------------------------------------------------------------


@pytest.fixture
def inversion_task(tmp_path):
    csv_path = tmp_path / "inversion.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_reaction", "fake1", "fake2", "fake3"])
        for _ in range(12):
            w.writerow(["CC(=O)O.CCO>>CC(=O)OCC", "CCO>>CC(=O)O", "CC>>O", "O>>CC"])
    return SmilesInversion(dataset_id_or_path=str(csv_path))


def test_inversion_load(inversion_task):
    ds = inversion_task.load()
    assert "train" in ds and "test" in ds
    assert len(ds["train"]) + len(ds["test"]) == 12


def test_inversion_accuracy_correct(inversion_task):
    inversion_task.load()
    # The solution is the true_reaction; options[0] is in a shuffled order.
    # We craft a scenario where the correct answer letter maps to the gold.
    options = [["CC(=O)O.CCO>>CC(=O)OCC", "CCO>>CC(=O)O", "CC>>O", "O>>CC"]]
    solution = ["CC(=O)O.CCO>>CC(=O)OCC"]
    completions = ["<think>thinking</think><answer>A</answer>"]
    rewards = inversion_task.accuracy_reward(completions, solution, options)
    assert rewards[0] == 1.0


def test_inversion_accuracy_wrong(inversion_task):
    inversion_task.load()
    options = [["CC(=O)O.CCO>>CC(=O)OCC", "CCO>>CC(=O)O", "CC>>O", "O>>CC"]]
    solution = ["CC(=O)O.CCO>>CC(=O)OCC"]
    completions = ["<think>thinking</think><answer>B</answer>"]
    rewards = inversion_task.accuracy_reward(completions, solution, options)
    assert rewards[0] == 0.0


def test_inversion_format_reward(inversion_task):
    good = inversion_task.format_reward(["p"], ["<think>reasoning</think>\n<answer>A</answer>"])
    bad = inversion_task.format_reward(["p"], ["just some text"])
    assert good[0] > bad[0]


# ---------------------------------------------------------------------------
# RxR — SmilesReplacement
# ---------------------------------------------------------------------------


@pytest.fixture
def replacement_task(tmp_path):
    csv_path = tmp_path / "replacement.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_reaction", "fake1", "fake2", "fake3"])
        for _ in range(12):
            w.writerow(["CC(=O)O.CCO>>CC(=O)OCC", "CCO>>CC(=O)O", "CC>>O", "O>>CC"])
    return SmilesReplacement(dataset_id_or_path=str(csv_path))


def test_replacement_load(replacement_task):
    ds = replacement_task.load()
    assert len(ds["train"]) + len(ds["test"]) == 12


def test_replacement_accuracy_correct(replacement_task):
    replacement_task.load()
    options = [["CC(=O)O.CCO>>CC(=O)OCC", "CCO>>CC(=O)O", "CC>>O", "O>>CC"]]
    solution = ["CC(=O)O.CCO>>CC(=O)OCC"]
    prompts = ["Which reaction is correct?"]
    completions = ["<think>thinking</think><answer>A</answer>"]
    rewards = replacement_task.accuracy_reward(prompts, completions, solution, options)
    assert rewards[0] == 1.0


def test_replacement_accuracy_wrong(replacement_task):
    replacement_task.load()
    options = [["CC(=O)O.CCO>>CC(=O)OCC", "CCO>>CC(=O)O", "CC>>O", "O>>CC"]]
    solution = ["CC(=O)O.CCO>>CC(=O)OCC"]
    prompts = ["Which reaction is correct?"]
    completions = ["<think>thinking</think><answer>C</answer>"]
    rewards = replacement_task.accuracy_reward(prompts, completions, solution, options)
    assert rewards[0] == 0.0


# ---------------------------------------------------------------------------
# RxN — Smiles2Name
# ---------------------------------------------------------------------------


@pytest.fixture
def naming_task(tmp_path):
    csv_path = tmp_path / "naming.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["REACTION_PROMPT", "CLASS"])
        classes = ["Acylation", "C-C Coupling", "Deprotection", "Reduction"]
        for i in range(20):
            w.writerow([f"CC(=O)Cl.CCO>>CC(=O)OCC.Cl reaction_{i}", classes[i % len(classes)]])
    return Smiles2Name(dataset_id_or_path=str(csv_path))


def test_naming_load(naming_task):
    ds = naming_task.load()
    assert len(ds["train"]) + len(ds["test"]) == 20


def test_naming_accuracy_exact_match(naming_task):
    naming_task.load()
    completions = ["<think>This is an acylation reaction</think><answer>Acylation</answer>"]
    rewards = naming_task.accuracy_reward(completions, solution=["Acylation"])
    assert rewards[0] >= 1.0  # exact match + reasoning bonus


def test_naming_accuracy_valid_but_wrong(naming_task):
    naming_task.load()
    completions = ["<think>thinking</think><answer>Reduction</answer>"]
    rewards = naming_task.accuracy_reward(completions, solution=["Acylation"])
    assert rewards[0] == pytest.approx(0.2)  # valid choice but wrong


def test_naming_accuracy_invalid(naming_task):
    naming_task.load()
    completions = ["<think>thinking</think><answer>banana</answer>"]
    rewards = naming_task.accuracy_reward(completions, solution=["Acylation"])
    assert rewards[0] == pytest.approx(-0.2)


def test_naming_format_reward(naming_task):
    good = naming_task.format_reward(["<think>reasoning</think><answer>Acylation</answer>"])
    bad = naming_task.format_reward(["just some text without tags"])
    assert good[0] > bad[0]


# ---------------------------------------------------------------------------
# RxTF — ReactionTrueFalse
# ---------------------------------------------------------------------------


@pytest.fixture
def truefalse_task(tmp_path):
    csv_path = tmp_path / "truefalse.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reaction", "label"])
        for i in range(12):
            label = "true" if i % 2 == 0 else "false"
            w.writerow([f"CC(=O)O.CCO>>CC(=O)OCC reaction_{i}", label])
    return ReactionTrueFalse(dataset_id_or_path=str(csv_path))


def test_truefalse_load(truefalse_task):
    ds = truefalse_task.load()
    assert len(ds["train"]) + len(ds["test"]) == 12


def test_truefalse_accuracy_correct(truefalse_task):
    truefalse_task.load()
    completions = ["<think>The reaction looks valid because...</think><answer>true</answer>"]
    rewards = truefalse_task.accuracy_reward(completions, solution=["true"])
    assert rewards[0] == 1.0


def test_truefalse_accuracy_wrong(truefalse_task):
    truefalse_task.load()
    completions = ["<think>The reaction is wrong</think><answer>false</answer>"]
    rewards = truefalse_task.accuracy_reward(completions, solution=["true"])
    assert rewards[0] == -0.5


def test_truefalse_majority_vote(truefalse_task):
    truefalse_task.load()
    # Majority of mentions is "true" even though the answer tag says false
    completions = ["<think>true true true reasoning</think><answer>false</answer>"]
    rewards = truefalse_task.accuracy_reward(completions, solution=["true"])
    # Majority vote: 3x true > 1x false → predicted "true" → matches gold
    assert rewards[0] == 1.0


def test_truefalse_format_reward(truefalse_task):
    good = truefalse_task.format_reward(["<think>reasoning</think><answer>true</answer>"])
    bad = truefalse_task.format_reward(["just some text"])
    assert good[0] > bad[0]
