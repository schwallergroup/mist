import os
import re
from random import random
from typing import Dict, Optional
from open_r1.download_data import download_data
import pandas as pd
from datasets import Dataset, DatasetDict
from ..base import RLTask
from dataclasses import field
from ase.io import read
import gemmi
from io import StringIO
from mace.calculators import mace_mp
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from .AIRS_preporcess._tokenizer import CIFTokenizer

cif_tokenizer = CIFTokenizer()
class BinaryCompoundRelaxing(RLTask):
    src_train_file: str = ""
    tgt_train_file: str = ""
    src_test_file: str = ""
    tgt_test_file: str = ""
    question_template: str = ""
    log_custom_metrics: bool = True
    custom_metrics: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(self.dataset_id_or_path):
            os.makedirs(self.dataset_id_or_path)
        download_data(self.dataset_id_or_path)

        self.src_train_file = os.path.join(
            self.dataset_id_or_path, "src-train.txt"
        )
        self.tgt_train_file = os.path.join(
            self.dataset_id_or_path, "tgt-train.txt"
        )
        self.src_test_file = (
            os.path.join(self.dataset_id_or_path, "src-test.txt")
            if "src-test.txt"
            else None
        )
        self.tgt_test_file = (
            os.path.join(self.dataset_id_or_path, "tgt-test.txt")
            if "tgt-test.txt"
            else None
        )
        self.question_template = (
            "<|im_start|>system You are a seasoned crystallographic structure analysis expert. "
            "Your task is to relax a binary compound to a stable state. <|im_end|>\n"
            "<|im_start|>user Given a perturbed binary compound:\n"
            "{}\n, perform multiple steps of Structural Relaxation on the given perturbed binary compound "
            "and reduce the internal energy. Please document your thought process within <think> </think> tags, and provide "
            "the final corrected structure in <answer> </answer> tags using the proper format as given in the example:\n"
            "serialized_cif formula Cd 1_int As 2_int \n"
            "space_group_symbol I4_122_sg\n"
            "lattice_parameters a 8.03811770 b 8.03811770 c 4.72563470 alpha 90.00000000 beta 90.00000000 gamma 90.00000000 \n"
            "Cd 4_int 0.00000000 0.00000000 0.00000000\n"
            "As 8_int 0.06170692 0.25000000 0.62500000\n"
            "<|im_end|>\n"
        )
        self.log_custom_metrics = True
        self.custom_metrics = {
            'val/rewards': [],
        }

        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/binary_compound_relaxing

    def read_files(self, src_file: str, tgt_file: str) -> Dict:
        """Read source and target files and create dataset dictionary."""
        def read_records(file_path: str) -> list:
            """Helper function to read multi-line records separated by blank lines."""
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            records = []
            current_record = []
            for line in lines:
                if line.strip() == "":  # Blank line indicates end of a record
                    if current_record:
                        records.append("\n".join(current_record))
                        current_record = []
                else:
                    current_record.append(line.strip())
            if current_record:  # Append the last record if file doesn't end with blank line
                records.append("\n".join(current_record))
            return records
        # Read records from source and target files
        src_records = read_records(src_file)
        tgt_records = read_records(tgt_file)

        # Generate problems using the question template
        problems = [self.question_template.format(record) for record in src_records]
        # Solutions are the raw target records (assuming no further processing needed)
        solutions = tgt_records

        return {
            "problem": problems,
            "solution": solutions,
        }

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        # Load training data
        train_dict = self.read_files(self.src_train_file, self.tgt_train_file)
        train_dataset = Dataset.from_dict(train_dict)

        # Load or create test data
        if self.src_test_file and self.tgt_test_file:
            test_dict = self.read_files(self.src_test_file, self.tgt_test_file)
            test_dataset = Dataset.from_dict(test_dict)
        else:
            # Create test split from training data
            train_test_split = train_dataset.train_test_split(test_size=0.1)
            train_dataset = train_test_split["train"].unique(column="solution")
            test_dataset = train_test_split["test"]

        # Combine into DatasetDict
        self.dataset = DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )

        return self.dataset
    
    def sanitize_cif(cif_str):
        lines = cif_str.splitlines()
        in_symmetry_loop = False
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("loop_"):
                in_symmetry_loop = False
                new_lines.append(line)
                continue
            if not in_symmetry_loop and "_symmetry_equiv_pos_as_xyz" in line:
                in_symmetry_loop = True
                new_lines.append(line)
                continue
            if in_symmetry_loop:
                if stripped == "" or stripped.startswith("_") or stripped.startswith("loop_"):
                    in_symmetry_loop = False
                    new_lines.append(line)
                else:
                    line = re.sub(r'"([^"]+)"', r"'\1'", line)
                    new_lines.append(line)
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    def parse_llm_structure(self, cif_content):
        sanitized = self.sanitize_cif(cif_content)
        try:
            return Structure.from_str(sanitized, fmt="cif")
        except Exception as e:
            print(f"Error parsing LLM‐generated structure: {e}")
            return None

    def compare_internal_energy(cif1, cif2):
        # uses ASE + MACE to get per‐atom potential energies
        atoms1 = read(StringIO(cif1), format='cif')
        atoms2 = read(StringIO(cif2), format='cif')
        calc = mace_mp(model="large", device='cuda')
        atoms1.calc = calc
        atoms2.calc = calc
        e1 = atoms1.get_potential_energy() / len(atoms1)
        e2 = atoms2.get_potential_energy() / len(atoms2)
        print("Original per‐atom energy:", e1)
        print("LLM per‐atom energy:", e2)
        if e1 < e2:
            return -4
        elif e1 > e2:
            return  1
        else:
            return -10

    def compute_internal_score(self, answer_cif, ground_truth_dict, alpha=5.0):
        gt_cif = ground_truth_dict.get("ground_truth", "")
        if not gt_cif:
            print("No ground truth CIF provided.")
            return -10

        # first, reformat / deserialize via tokenizer
        try:
            answer_cif = cif_tokenizer.deserialize(answer_cif, gt_cif)
        except Exception as e:
            print("Tokenization error:", e)
            return -10

        # quick gemmi checks
        try:
            for s in (gt_cif, answer_cif):
                doc = gemmi.cif.read_string(s)
                doc.check_for_missing_values()
                doc.check_for_duplicates()
        except Exception as e:
            print("CIF validation error:", e)
            return -10

        # parse Pymatgen structures
        try:
            dft_struct = Structure.from_str(gt_cif, fmt="cif")
        except Exception as e:
            print("Error parsing DFT structure:", e)
            return -10

        llm_struct = self.parse_llm_structure(answer_cif)
        if llm_struct is None:
            return -10

        # structure‐matching reward (–RMSD if match, else penalty)
        matcher = StructureMatcher(ltol=0.05, stol=0.05, angle_tol=1)
        try:
            if matcher.fit(dft_struct, llm_struct):
                rmsd = matcher.get_rms_dist(dft_struct, llm_struct)
                struct_reward = -rmsd
            else:
                return -10
        except Exception as e:
            print("Matcher error:", e)
            return -10

        # energy‐based reward
        energy_reward = self.compare_internal_energy(gt_cif, answer_cif)

        # choose which to return (here using energy check as original)
        return energy_reward
    
    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - check that completion is same as ground truth."""
        rewards = []
        # Here task is simple: check that the smiles is the same as the target smiles
        for content, sol in zip(completions, solution):
            print(f"\n\n=======<RESPONSE>=======\n"
                f"# answer_text: {content}\n"
                f"# ground_truth: {sol}\n"
            )
            content = self.preprocess_response(content)
            if content == "NONE":
                rewards.append(-10)
                continue

            # server_url = os.environ.get("SERVER_URL", "http://10.197.48.175:9001/compute_score")
            if content == sol:
                rewards.append(-10)
                continue
            
            try:
                reward = self.compute_internal_score(content, sol)
                rewards.append(reward)
            except Exception as e:
                rewards.append(-10)
        if self.log_custom_metrics:
            self.custom_metrics['val/rewards'].extend(rewards)
        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        pattern = r"<answer>(.*)<\/answer>"
        m = re.findall(pattern, response, re.DOTALL)
        if m:
            return m[-1].strip()
        else:
            return "NONE"

    def get_metrics(self) -> Dict:
        """
        Get task metrics to log in WANDB.
        This function takes no arguments and returns a dictionary of metrics {key[str]: value[float]}.
        """
        metrics = dict()
        if self.log_custom_metrics:
            rewards = self.custom_metrics['val/rewards']
            if rewards:
                correct_count = sum(1 for r in rewards if r == 1)
                total_count = len(rewards)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                metrics['val/accuracy'] = accuracy
                self.custom_metrics['val/rewards'] = []
        return metrics