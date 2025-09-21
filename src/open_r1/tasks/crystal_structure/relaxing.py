import os
import re
from random import random
from typing import Dict, Optional
from open_r1.download_data import download_data
import pandas as pd
from datasets import Dataset, DatasetDict
from open_r1.tasks.base import RLTask
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
            "system You are a seasoned crystallographic structure analysis expert. "
            "Your task is to relax a binary compound to a stable state.\n"
            "user Given a perturbed binary compound:\n"
            "{}\n, perform multiple steps of Structural Relaxation on the given perturbed binary compound "
            "and reduce the internal energy. Please document your thought process within <think> </think> tags, and provide "
            "the final corrected structure in <answer> </answer> tags using the proper m2s format as given in the example:\n"
            "serialized_cif formula Cd 1_int As 2_int \n"
            "space_group_symbol I4_122_sg\n"
            "lattice_parameters a 8.03811770 b 8.03811770 c 4.72563470 alpha 90.00000000 beta 90.00000000 gamma 90.00000000 \n"
            "Cd 4_int 0.00000000 0.00000000 0.00000000\n"
            "As 8_int 0.06170692 0.25000000 0.62500000\n"
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
    
    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - check that completion is same as ground truth."""
        def compute_internal_score(answer_cif, ground_truth_dict, alpha=5.0):
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
            def parse_llm_structure(cif_content):
                sanitized = sanitize_cif(cif_content)
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
                    return 0.5
                elif e1 > e2:
                    return  1
                else:
                    return 0
            gt_cif = ground_truth_dict
            if not gt_cif:
                print("No ground truth CIF provided.")
                return 0
            # first, reformat / deserialize via tokenizer
            try:
                answer_cif = cif_tokenizer.deserialize(answer_cif, gt_cif)
            except Exception as e:
                print("Tokenization error:", e)
                return 0

            # quick gemmi checks
            try:
                for s in (gt_cif, answer_cif):
                    doc = gemmi.cif.read_string(s)
                    doc.check_for_missing_values()
                    doc.check_for_duplicates()
            except Exception as e:
                print("CIF validation error:", e, '\n GT:', gt_cif, '\n answer_cif', answer_cif)
                return 0

            # parse Pymatgen structures
            try:
                dft_struct = Structure.from_str(gt_cif, fmt="cif")
            except Exception as e:
                print("Error parsing DFT structure:", e)
                return 0

            llm_struct = parse_llm_structure(answer_cif)
            if llm_struct is None:
                return 0

            # energy‐based reward
            energy_reward = compare_internal_energy(gt_cif, answer_cif)

            # choose which to return (here using energy check as original)
            return energy_reward
        rewards = []
        # Here task is simple: check that the smiles is the same as the target s
        for content, sol in zip(completions, solution):
            print(f"\n\n=======<RESPONSE>=======\n"
                f"#answer_text: {content}\n"
            )
            content = self.preprocess_response(content)
            print("#llm generated cif: ", content)
            print(f"\n# ground_truth: {sol}\n")
            if content == "NONE":
                rewards.append(0)
                print('content NONE reward 0')
                continue

            # server_url = os.environ.get("SERVER_URL", "http://10.197.48.175:9001/compute_score")
            if content == sol:
                print('content == init cif reward 0')
                rewards.append(0)
                continue
            
            try:
                reward = compute_internal_score(content, sol)
                print('all good reward: ', reward)
                rewards.append(reward)
            except Exception as e:
                print('compute_internal_score failed: ', e)
                rewards.append(0)
        if self.log_custom_metrics:
            self.custom_metrics['val/rewards'].extend(rewards)
        return rewards
    
    def format_reward(self, completions, **kwargs):
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs

        Returns:
            list[float]: Reward scores
        """
        rewards = []

        # detect malformed or missing tags
        tag_regex = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL)
        space_groups = [
            "P6/mmm", "Imma", "P4_32_12", "P4_2/mnm", "Fd-3m", "P3m1", "P-3", "P4mm", "P4_332", "P4/nnc", "P2_12_12", "Pnn2", "Pbcn", "P4_2/n", "Cm", "R3m", "Cmce", "Aea2", "P-42_1m", "P-42m", "P2_13", "R-3", "Fm-3", "Cmm2", "Pn-3n", "P6/mcc", "P-6m2", "P3_2", "P-3m1", "P3_212", "I23", "P-62m", "P4_2nm", "Pma2", "Pmma", "I-42m", "P-31c", "Pa-3", "Pmmn", "Pmmm", "P4_2/ncm", "I4/mcm", "I-4m2", "P3_1", "Pcc2", "Cmcm", "I222", "Fddd", "P312", "Cccm", "P6_1", "F-43c", "P6_322", "Pm-3", "P3_121", "P6_4", "Ia-3d", "Pm-3m", "P2_1/c", "C222_1", "Pc", "P4/n", "Pba2", "Ama2", "Pbcm", "P31m", "Pcca", "P222", "P-43n", "Pccm", "P6_422", "F23", "P42_12", "C222", "Pnnn", "P6_3cm", "P4_12_12", "P6/m", "Fmm2", "I4_1/a", "P4/mbm", "Pmn2_1", "P4_2bc", "P4_22_12", "I-43d", "I4/m", "P4bm", "Fdd2", "P3", "P6_122", "Pnc2", "P4_2/mcm", "P4_122", "Cmc2_1", "P-6c2", "R32", "P4_1", "P4_232", "Pnna", "P422", "Pban", "Cc", "I4_122", "P6_3/m", "P6_3mc", "I4_1/amd", "P4_2", "P4/nmm", "Pmna", "P4/m", "Fm-3m", "P4/mmm", "Imm2", "P4/ncc", "P-62c", "Ima2", "P6_5", "P2/c", "P4/nbm", "Ibam", "P6_522", "P6_3/mmc", "I4/mmm", "Fmmm", "P2/m", "P-4b2", "I-4", "C2/m", "P4_2/mmc", "P4", "Fd-3c", "P4_3", "P2_1/m", "I-43m", "P-42c", "F4_132", "Pm", "Pccn", "P-4n2", "P4_132", "P23", "I4cm", "R3c", "Amm2", "Immm", "Iba2", "I4", "Fd-3", "P1", "Pbam", "P4_2/nbc", "Im-3", "P4_2/nnm", "Pmc2_1", "P-31m", "R-3m", "Ia-3", "P622", "F222", "P2", "P-1", "Pmm2", "P-4", "Aem2", "P6_222", "P-3c1", "P4_322", "I422", "Pnma", "P6_3", "P3c1", "Pn-3", "P4nc", "P-6", "P4/mcc", "I2_12_12_1", "P4_2/mbc", "P31c", "Ccc2", "P4_2/nmc", "P6_3/mcm", "C2", "Pbca", "P-4c2", "I4_1cd", "P2_1", "P3_112", "P4_2mc", "Pn-3m", "C2/c", "R3", "P-43m", "I432", "P222_1", "I-42d", "I-4c2", "P6cc", "P6_2", "P3_221", "P321", "Pca2_1", "I4_1/acd", "I4_132", "F432", "Pna2_1", "Ccce", "Ibca", "P4/mnc", "I4_1md", "P2_12_12_1", "R-3c", "I2_13", "P-4m2", "Pm-3n", "I4mm", "F-43m", "Pnnm", "P-42_1c", "Cmmm", "P6mm", "P4_2cm", "P4_2/m", "Im-3m", "Fm-3c", "I4_1", "P4cc", "Cmme"
        ]
        escaped = [re.escape(sg) for sg in space_groups]
        pattern = r"\b(?:" + "|".join(escaped) + r")\b"

        symmetry_pattern = re.compile(pattern, re.IGNORECASE)
        # bonus patterns
        cif_pattern = re.compile(
            r"\b("
            r"cif|space\s+group|unit\s+cell|lattice|"
            r"symmetry|fractional\s+coordinates|cell\s+parameters|"
            r"bond\s+length|bond\s+angle|volume|Wyckoff|"
            r"atomic\s+positions|occupancy|site\s+multiplicity"
            r")\b",
            re.IGNORECASE
        )
        math_pattern = re.compile(
            # looks for sqrt(…), (…)^2, (…)=(…), or simple a±b/c etc.
            r"(?:sqrt\s*\(|\([^)]*\)\s*\^\s*2|[0-9\.\)]+\s*[\+\-\*/=]\s*[0-9\.\(])",
            re.IGNORECASE
        )
        position_pattern = re.compile(
            r"\b("
            r"position|pos\.?|coordinate|coord\.?|site|"
            r"atomic\s+position|fractional\s+coord(?:inate)?s?|"
            r"xyz|uvw"
            r")\b",
            re.IGNORECASE
        )
        lattice_angle_pattern = re.compile(
            r"\b(a=|b=|c=|c/a|gamma\s*=?\s*\d+(\.\d+)?°?)\b",
            re.IGNORECASE
        )
        crystallographic_pattern = re.compile(
            r"\b("
            r"Wyckoff|multiplicity|asymmetric unit|mirror plane|inversion center|"
            r"Bravais lattice|primitive cell|supercell"
            r")\b",
            re.IGNORECASE
        )
        energy_force_pattern = re.compile(
            r"\b("
            r"formation energy|total energy|enthalpy|residual force|stress|"
            r"converged energy|converged stress|force\s*<\s*0\.01\s*eV/Å"
            r")\b",
            re.IGNORECASE
        )
        dynamical_pattern = re.compile(
            r"\b("
            r"phonon dispersion|imaginary mode|soft mode|dynamical stability|"
            r"elastic constant|Born criteria"
            r")\b",
            re.IGNORECASE
        )
        classification_pattern = re.compile(
            r"\b("
            r"perovskite|spinel|rocksalt|layered oxide|phase transition|"
            r"Jahn[- ]Teller distortion|olivine|rutile"
            r")\b",
            re.IGNORECASE
        )
        chemical_pattern = re.compile(
            r"\b("
            r"bond length|bond angle|electronegativity|Bader charge|"
            r"electron localization|coordination number|coordination environment|ionic radius"
            r")\b",
            re.IGNORECASE
        )

        for completion in completions:
            try:
                # ensure it at least starts in the right place
                if not completion.startswith("<think>"):
                    completion = "<think>" + completion

                m = tag_regex.search(completion)
                if not m:
                    rewards.append(0.0)
                    continue

                bonus = 0.0
                if cif_pattern.search(completion):
                    bonus += 0.1
                if math_pattern.search(completion):
                    bonus += 0.2
                if position_pattern.search(completion):
                    bonus += 0.1
                if symmetry_pattern.search(completion):
                    bonus += 0.1
                if lattice_angle_pattern.search(completion):
                    bonus += 0.05
                if crystallographic_pattern.search(completion):
                    bonus += 0.1
                if energy_force_pattern.search(completion):
                    bonus += 0.1
                if dynamical_pattern.search(completion):
                    bonus += 0.1
                if classification_pattern.search(completion):
                    bonus += 0.1
                if chemical_pattern.search(completion):
                    bonus += 0.05

                rewards.append(bonus)

            except Exception:
                rewards.append(0.0)

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

Using the fractional coordinates, we can compute the distances between Ti and Ni atoms. Let's start with the first Ti at (0.335, 0.349, 0.316). Let's compute the distance to the nearest Ni. Let's assume Ni is at (0.0263, -0.0047, -0.0301). The distance formula in triclinic is:
    d = sqrt[(0.335 - 0.0263)Â² + (0.349 + 0.0047)Â² + (0.316 + 0.0301)Â²]
    d = sqrt[(0.335 - 0.0263)Â² + (0.349 + 0.0047)Â² + (0.316 + 0.0301)Â²]
             (0.335 - 0.0263) = 0.3087

longer/shorter than
[0.0617, 0.25, 0.625]
unit cell volume
Bond Length

Similarly, the other Fe-S bonds would need to adjust. Let's check another pair: Fe at (0.02498, -0.04894, 0.50441) and S at (0.6288, 0.3088, 0.7722). The dx here is 0.6288 - 0.02498 = 0.50382, dy = 0.3088 - (-0.04894) = 0.35774, dz = 0.7722 - 0.50441 = 0.2678. So distance â<89><88> sqrt(0.50382Â² + 0.35774Â² + 0.2678Â²) â<89><88> sqrt(0.254 + 0.128 + 0.0718) â<89><88> sqrt(0.4538) â<89><88> 0.674 Ã<85>. Still too short. Fe-S bond lengths in FeS are typically 2.3-2.4 Ã<85>, so these are under strain.