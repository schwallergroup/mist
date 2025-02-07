
import pytest
from rdkit import Chem
from open_r1.tasks import ForwardReaction

class TestAccuracyReward:
    def setup_method(self):
        self.reward_calculator = ForwardReaction(
            root_dir="data/USPTO",
            src_train_file="src-train.txt",
            tgt_train_file="tgt-train.txt",
            src_test_file="src-test.txt",
            tgt_test_file="tgt-test.txt",
        ) 
        
    def test_correct_smiles_match(self):
        """Test when completion SMILES matches solution SMILES"""
        completions = [[{"content":"<answer>CC</answer>"}]]
        solution = ["CC"]
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [1]

    def test_incorrect_smiles(self):
        """Test when completion SMILES doesn't match solution SMILES"""
        completions = [[{"content":"<answer>CC</answer>"}]]
        solution = ["CCC"]
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [-0.5]

    def test_invalid_completion_smiles(self):
        """Test when completion contains invalid SMILES"""
        completions = [[{"content":"<answer>Xasd-</answer>"}]]
        solution = ["CC"]
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [-1]

    def test_invalid_solution_smiles(self):
        """Test when solution contains invalid SMILES"""
        completions = [[{"content":"<answer>CC</answer>"}]]
        solution = ["XXX"]
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [-1]

    def test_multiple_smiles(self):
        """Test with multiple SMILES strings"""
        completions = [[{"content":"<answer>CC</answer>"}], [{"content":"<answer>CCC</answer>"}]]
        solution = ["CC", "CCC"]
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [1, 1]

    def test_empty_inputs(self):
        """Test with empty inputs"""
        completions = []
        solution = []
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == []

    def test_different_representations_same_molecule(self):
        """Test different SMILES representations of the same molecule"""
        completions = [[{"content":"<answer>CC(C)C</answer>"}], [{"content":"<answer>CCC</answer>"}]]
        solution = ["C(C)(C)C", "CC(C)"]  # Different representation but same molecule
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [1, 1]

    def test_mixed_valid_invalid_smiles(self):
        """Test with a mix of valid and invalid SMILES"""
        completions = [[{"content":"<answer>CC</answer>"}], [{"content":"<answer>XXX</answer>"}], [{"content":"<answer>CCC</answer>"}]]
        solution = ["CC", "CC", "CCX"]
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [1, -1, -1]


    def test_mixed_format_cases(self):
            """Test with a mix of correct/incorrect formats"""
            completions = [
                [{"content": "<answer>CC</answer>"}],          # correct format, valid SMILES
                [{"content": "CC"}],                           # no tags (-1)
                [{"content": "<answer>CCC</answer>"}],         # correct format, valid SMILES
                [{"content": "SMILES: CC"}],                   # no tags (-1)
                [{"content": "<answer>CCO</answer>"}]          # correct format, valid SMILES
            ]
            solution = ["CC", "CC", "CCC", "CC", "CCO"]
            result = self.reward_calculator.accuracy_reward(completions, solution)
            assert result == [1, -1, 1, -1, 1]

    def test_various_incorrect_formats(self):
        """Test different ways of providing incorrect format"""
        completions = [
            [{"content": "The SMILES is CC"}],             # no tags (-1)
            [{"content": "[CC]"}],                         # no tags (-1)
            [{"content": "Product: CC"}],                  # no tags (-1)
            [{"content": "CC -> product"}],                # no tags (-1)
            [{"content": "<answer>CC</answer>"}]           # correct format
        ]
        solution = ["CC", "CC", "CC", "CC", "CC"]
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [-1, -1, -1, -1, 1]

    def test_mixed_format_and_validity(self):
        """Test mix of format issues and SMILES validity"""
        completions = [
            [{"content": "<answer>CC</answer>"}],          # correct format, valid SMILES
            [{"content": "Invalid-SMILES"}],               # no tags (-1)
            [{"content": "<answer>XXX</answer>"}],         # correct format, invalid SMILES (-1)
            [{"content": "CC(=O)C"}],                      # no tags (-1)
            [{"content": "<answer>CC(=O)C</answer>"}]      # correct format, valid SMILES
        ]
        solution = ["CC", "CC", "CC", "CC(=O)C", "CC(=O)C"]
        result = self.reward_calculator.accuracy_reward(completions, solution)
        assert result == [1, -1, -1, -1, 1]