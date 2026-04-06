import math
import os
import re

from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(THIS_DIR, "spacegroups.txt"), "rt") as f:
    SPACE_GROUPS = [sg.strip() for sg in f.readlines()]


ATOMS = [
    "Si",
    "C",
    "Pb",
    "I",
    "Br",
    "Cl",
    "Eu",
    "O",
    "Fe",
    "Sb",
    "In",
    "S",
    "N",
    "U",
    "Mn",
    "Lu",
    "Se",
    "Tl",
    "Hf",
    "Ir",
    "Ca",
    "Ta",
    "Cr",
    "K",
    "Pm",
    "Mg",
    "Zn",
    "Cu",
    "Sn",
    "Ti",
    "B",
    "W",
    "P",
    "H",
    "Pd",
    "As",
    "Co",
    "Np",
    "Tc",
    "Hg",
    "Pu",
    "Al",
    "Tm",
    "Tb",
    "Ho",
    "Nb",
    "Ge",
    "Zr",
    "Cd",
    "V",
    "Sr",
    "Ni",
    "Rh",
    "Th",
    "Na",
    "Ru",
    "La",
    "Re",
    "Y",
    "Er",
    "Ce",
    "Pt",
    "Ga",
    "Li",
    "Cs",
    "F",
    "Ba",
    "Te",
    "Mo",
    "Gd",
    "Pr",
    "Bi",
    "Sc",
    "Ag",
    "Rb",
    "Dy",
    "Yb",
    "Nd",
    "Au",
    "Os",
    "Pa",
    "Sm",
    "Be",
    "Ac",
    "Xe",
    "Kr",
    "He",
    "Ne",
    "Ar",
]

DIGITS = [str(d) for d in list(range(10))]

INTS = [str(d) for d in list(range(300))]

KEYWORDS = ["space_group_symbol", "formula", "atoms", "lattice_parameters", "a", "b", "c", "alpha", "beta", "gamma"]

UNK_TOKEN = "<unk>"


def get_spacegroup_number(sg_symbol):
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        sg = SpaceGroup(sg_symbol)
        return sg
    except Exception as e:
        print("Err:", e)
        return None


def parse_formula(formula):
    formula = formula.replace("'", "").replace('"', "").strip()
    pattern = r"([A-Z][a-z]*)(\d*)"
    counts = {}
    for element, count in re.findall(pattern, formula):
        counts[element] = counts.get(element, 0) + (int(count) if count else 1)
    return counts


def compute_cell_formula_units_Z(formula_sum, formula_structural):
    counts_sum = parse_formula(formula_sum)
    counts_struct = parse_formula(formula_structural)

    ratios = []
    for element, count_struct in counts_struct.items():
        if element not in counts_sum:
            raise ValueError(f"{element}")
        ratio = counts_sum[element] / count_struct
        if ratio != int(ratio):
            raise ValueError(f"{element}, {ratio} not int")
        ratios.append(int(ratio))

    if len(set(ratios)) != 1:
        raise ValueError(f"{ratios} != 1")
    return ratios[0]


class CIFTokenizer:
    def __init__(self):
        self._tokens = ["<pad>"]
        self._tokens.extend(self.atoms())
        self._tokens.extend(self.digits())
        self._tokens.extend(self.keywords())
        self._tokens.extend(self.symbols())

        space_groups = list(self.space_groups())
        # Replace 'Pm' space group with 'Pm_sg' to disambiguate from atom 'Pm',
        #  or 'P1' with 'P1_sg' to disambiguate from atom 'P' and number '1'
        space_groups_sg = [sg + "_sg" for sg in space_groups]
        self._tokens.extend(space_groups_sg)

        digits_int = [v + "_int" for v in INTS]
        self._tokens.extend(digits_int)

        self._escaped_tokens = [re.escape(token) for token in self._tokens]
        self._escaped_tokens.sort(key=len, reverse=True)

        # a mapping from characters to integers
        self._token_to_id = {ch: i for i, ch in enumerate(self._tokens)}
        self._id_to_token = {i: ch for i, ch in enumerate(self._tokens)}
        # map the id of 'Pm_sg' back to 'Pm', or 'P1_sg' to 'P1',
        #  for decoding convenience
        for sg in space_groups_sg:
            self._id_to_token[self.token_to_id[sg]] = sg.replace("_sg", "")

        for v_int in digits_int:
            self._id_to_token[self.token_to_id[v_int]] = v_int.replace("_int", "")

    @staticmethod
    def atoms():
        return ATOMS

    @staticmethod
    def digits():
        return DIGITS

    @staticmethod
    def keywords():
        kws = list(KEYWORDS)
        return kws

    @staticmethod
    def symbols():
        # return ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]
        return [",", " ", ":", ".", "\n"]

    @staticmethod
    def space_groups():
        return SPACE_GROUPS

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    def prompt_tokenize(self, cif):
        token_pattern = "|".join(self._escaped_tokens)
        # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
        full_pattern = f"({token_pattern}|\\w+|[\\.,;!?])"
        # Tokenize the input string using the regex pattern
        cif = re.sub(r"[ \t]+", " ", cif)
        tokens = re.findall(full_pattern, cif)
        return tokens

    def encode(self, tokens):
        # encoder: take a list of tokens, output a list of integers
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids):
        # decoder: take a list of integers (i.e. encoded tokens), output a string
        return "".join([self._id_to_token[i] for i in ids])

    def serialize(self, cif_string):
        spacegroups = "|".join(SPACE_GROUPS)
        cif_string = re.sub(rf"(_symmetry_space_group_name_H-M *\b({spacegroups}))\n", r"\1_sg\n", cif_string)
        extracted_data = self.tokenize_cif_preprocess(cif_string)

        seq_res = ""
        # formula
        seq_res += "formula "
        formula = extracted_data["formula"]
        elements_counts = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
        for element, count in elements_counts:
            if not element:
                break
            if not count:
                count = "1"
            seq_res += element + " " + count + "_int "
        seq_res += "\n"
        # space group name
        seq_res += "space_group_symbol " + extracted_data["space_group_symbol"] + "\n"
        # lattice
        seq_res += "lattice_parameters " + "a " + extracted_data["lattice_parameters"]["a"] + " "
        seq_res += "b " + extracted_data["lattice_parameters"]["b"] + " "
        seq_res += "c " + extracted_data["lattice_parameters"]["c"] + " "
        seq_res += "alpha " + extracted_data["lattice_parameters"]["alpha"] + " "
        seq_res += "beta " + extracted_data["lattice_parameters"]["beta"] + " "
        seq_res += "gamma " + extracted_data["lattice_parameters"]["gamma"] + " "
        seq_res += "\n"
        # atoms
        for idx in range(len(extracted_data["atoms"])):
            tmp = extracted_data["atoms"][idx]
            seq_res += (
                tmp["type"]
                + " "
                + tmp["num"]
                + "_int "
                + tmp["coordinates"][0]
                + " "
                + tmp["coordinates"][1]
                + " "
                + tmp["coordinates"][2]
                + "\n"
            )
        seq_res += "\n"
        # Create a regex pattern by joining the escaped tokens with '|'
        token_pattern = "|".join(self._escaped_tokens)
        # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
        full_pattern = f"({token_pattern}|\\w+|[\\.,;!?])"
        # Tokenize the input string using the regex pattern
        seq_res = re.sub(r"[ \t]+", " ", seq_res)
        return seq_res

    def deserialize(self, custom_str, ground_truth=None):
        print("self", self)
        print("custom_str", custom_str)
        print("ground_truth", ground_truth)
        pattern_structural = re.compile(r"_chemical_formula_structural\s+['\"]?([^\n'\"]+)['\"]?")
        pattern_sum = re.compile(r"_chemical_formula_sum\s+['\"]?([^'\"]+)['\"]?")
        pattern_units = re.compile(r"_cell_formula_units_Z\s+(\d+)")

        structural_match = pattern_structural.search(ground_truth)
        sum_match = pattern_sum.search(ground_truth)
        units_match = pattern_units.search(ground_truth)

        symmetry_equiv_pos_pattern = re.compile(
            r"loop_\s*\n\s*_symmetry_equiv_pos_site_id\s*\n\s*_symmetry_equiv_pos_as_xyz\s*\n(.*?)(?:\nloop_|\Z)",
            re.DOTALL,
        )
        symmetry_equiv_pos_match = symmetry_equiv_pos_pattern.search(ground_truth)
        if symmetry_equiv_pos_match:
            sym_ops_block = symmetry_equiv_pos_match.group(1).strip()

        formula_structural = structural_match.group(1) if structural_match else None
        formula_sum = sum_match.group(1) if sum_match else None
        units_Z = int(units_match.group(1)) if units_match else None
        print("formula_structural", formula_structural)
        lines = custom_str.strip().splitlines()
        data = {}

        if lines:
            tokens = lines[0].split()
            if tokens[0] != "formula":
                raise ValueError("'formula' missing")
            formula = ""
            for i in range(1, len(tokens), 2):
                element = tokens[i]
                count_token = tokens[i + 1] if i + 1 < len(tokens) else ""
                if count_token.endswith("_int"):
                    count = count_token[:-4]
                else:
                    count = count_token
                formula += f"{element}{count}"
            data["formula"] = formula

        if len(lines) >= 2:
            tokens = lines[1].split()
            if tokens[0] != "space_group_symbol":
                raise ValueError("'space_group_symbol' missing")
            data["space_group_symbol"] = " ".join(tokens[1:])

        if len(lines) >= 3:
            tokens = lines[2].split()
            if tokens[0] != "lattice_parameters":
                raise ValueError("'lattice_parameters' missing")
            lattice = {}
            for i in range(1, len(tokens), 2):
                key = tokens[i]
                value = tokens[i + 1] if i + 1 < len(tokens) else ""
                lattice[key] = value
            data["lattice_parameters"] = lattice

        atoms = []
        for line in lines[3:]:
            if not line.strip():
                break
            tokens = line.split()
            if len(tokens) < 5:
                continue
            atom_type = tokens[0]
            num_token = tokens[1]
            if num_token.endswith("_int"):
                num = num_token[:-4]
            else:
                num = num_token
            coords = tokens[2:5]
            atoms.append({"type": atom_type, "num": num, "coordinates": coords})
        data["atoms"] = atoms

        cif_lines = []
        cif_lines.append(f"data_{formula_structural}")
        cif_lines.append(f"_symmetry_space_group_name_H-M {data['space_group_symbol'].split('_sg')[0]}")
        lattice = data["lattice_parameters"]
        cif_lines.append(f"_cell_length_a {lattice.get('a', '')}")
        cif_lines.append(f"_cell_length_b {lattice.get('b', '')}")
        cif_lines.append(f"_cell_length_c {lattice.get('c', '')}")
        cif_lines.append(f"_cell_angle_alpha {lattice.get('alpha', '')}")
        cif_lines.append(f"_cell_angle_beta {lattice.get('beta', '')}")
        cif_lines.append(f"_cell_angle_gamma {lattice.get('gamma', '')}")
        space_group_symbol = str(get_spacegroup_number(data["space_group_symbol"].split("_sg")[0].strip("'")))
        space_group_symbol = re.search(r"number\s+(\d+)", space_group_symbol).group(1)
        cif_lines.append(f"_symmetry_Int_Tables_number  {space_group_symbol}")
        cif_lines.append(f"_chemical_formula_structural  {formula_structural}")
        cif_lines.append(f"_chemical_formula_sum '{formula_sum}'")

        a = float(lattice.get("a", 0))
        b = float(lattice.get("b", 0))
        c = float(lattice.get("c", 0))
        alpha = float(lattice.get("alpha", 90))
        beta = float(lattice.get("beta", 90))
        gamma = float(lattice.get("gamma", 90))
        alpha_rad = math.radians(alpha)
        beta_rad = math.radians(beta)
        gamma_rad = math.radians(gamma)

        cos_alpha = math.cos(alpha_rad)
        cos_beta = math.cos(beta_rad)
        cos_gamma = math.cos(gamma_rad)
        cell_volume = (
            a * b * c * math.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
        )
        cif_lines.append(f"_cell_volume {cell_volume:.8f}")
        cif_lines.append(f"_cell_formula_units_Z '{units_Z}'")
        cif_lines.append("loop_")
        cif_lines.append(" _symmetry_equiv_pos_site_id")
        cif_lines.append(" _symmetry_equiv_pos_as_xyz")
        cif_lines.append(f"  {sym_ops_block}")
        cif_lines.append("loop_")
        cif_lines.append("_atom_site_type_symbol")
        cif_lines.append("_atom_site_label")
        cif_lines.append("_atom_site_symmetry_multiplicity")
        cif_lines.append("_atom_site_fract_x")
        cif_lines.append("_atom_site_fract_y")
        cif_lines.append("_atom_site_fract_z")
        cif_lines.append("_atom_site_occupancy")
        unique_counts = {}
        for atom in data["atoms"]:
            label = f"{atom['type']}"
            if label not in unique_counts:
                unique_counts[label] = len(unique_counts)
                label = label + str(unique_counts[label])
            else:
                label = label + str(unique_counts[label])
            cif_lines.append(
                f"{  atom['type']}  {label}  {atom['num']}  {atom['coordinates'][0]}  {atom['coordinates'][1]}  {atom['coordinates'][2]}  1"
            )
        cif_string_reconstructed = "\n".join(cif_lines)
        return cif_string_reconstructed

    def tokenize_cif(self, cif_string, max_length=1385):
        # Preprocessing step to replace '_symmetry_space_group_name_H-M Pm'
        #  with '_symmetry_space_group_name_H-M Pm_sg',to disambiguate from atom 'Pm',
        #  or any space group symbol to avoid problematic cases, like 'P1'
        spacegroups = "|".join(SPACE_GROUPS)
        cif_string = re.sub(rf"(_symmetry_space_group_name_H-M *\b({spacegroups}))\n", r"\1_sg\n", cif_string)

        extracted_data = self.tokenize_cif_preprocess(cif_string)

        seq_res = ""
        # formula
        seq_res += "formula "
        formula = extracted_data["formula"]
        elements_counts = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
        for element, count in elements_counts:
            if not element:
                break
            if not count:
                count = "1"
            seq_res += element + " " + count + "_int "
        seq_res += "\n"
        # space group name
        seq_res += "space_group_symbol " + extracted_data["space_group_symbol"] + "\n"
        # lattice
        seq_res += "lattice_parameters " + "a " + extracted_data["lattice_parameters"]["a"] + " "
        seq_res += "b " + extracted_data["lattice_parameters"]["b"] + " "
        seq_res += "c " + extracted_data["lattice_parameters"]["c"] + " "
        seq_res += "alpha " + extracted_data["lattice_parameters"]["alpha"] + " "
        seq_res += "beta " + extracted_data["lattice_parameters"]["beta"] + " "
        seq_res += "gamma " + extracted_data["lattice_parameters"]["gamma"] + " "
        seq_res += "\n"
        # atoms
        for idx in range(len(extracted_data["atoms"])):
            tmp = extracted_data["atoms"][idx]
            seq_res += (
                tmp["type"]
                + " "
                + tmp["num"]
                + "_int "
                + tmp["coordinates"][0]
                + " "
                + tmp["coordinates"][1]
                + " "
                + tmp["coordinates"][2]
                + "\n"
            )
        seq_res += "\n"
        # Create a regex pattern by joining the escaped tokens with '|'
        token_pattern = "|".join(self._escaped_tokens)
        # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
        full_pattern = f"({token_pattern}|\\w+|[\\.,;!?])"
        # Tokenize the input string using the regex pattern
        seq_res = re.sub(r"[ \t]+", " ", seq_res)
        # print(seq_res)
        tokens = re.findall(full_pattern, seq_res)
        # print(tokens)
        padding_length = max_length - len(tokens)
        if padding_length > 0:
            tokens.extend(["<pad>"] * padding_length)

        return tokens

    def tokenize_cif_preprocess(self, cif_string):
        # Re-initialize the dictionary to hold the extracted data
        extracted_data = {"space_group_symbol": "", "formula": "", "atoms": [], "lattice_parameters": {}}

        # Split the text into lines for processing
        lines = cif_string.split("\n")

        # Iterate through each line to extract the required information
        atom_line_idx = -1
        for line_idx in range(len(lines)):
            line = lines[line_idx]
            # Extract space group symbol
            if "_symmetry_space_group_name_H-M" in line:
                spacegroup_match = re.search(r"_symmetry_space_group_name_H-M\s+([^\n]+)", line)
                spacegroup = spacegroup_match.group(1).strip()
                extracted_data["space_group_symbol"] = spacegroup
            # Extract formula
            elif line.startswith("data_"):
                extracted_data["formula"] = line.split("_")[1]
            # Extract lattice parameters
            elif line.startswith("_cell_length_a"):
                extracted_data["lattice_parameters"]["a"] = line.split()[-1]
            elif line.startswith("_cell_length_b"):
                extracted_data["lattice_parameters"]["b"] = line.split()[-1]
            elif line.startswith("_cell_length_c"):
                extracted_data["lattice_parameters"]["c"] = line.split()[-1]
            elif line.startswith("_cell_angle_alpha"):
                extracted_data["lattice_parameters"]["alpha"] = line.split()[-1]
            elif line.startswith("_cell_angle_beta"):
                extracted_data["lattice_parameters"]["beta"] = line.split()[-1]
            elif line.startswith("_cell_angle_gamma"):
                extracted_data["lattice_parameters"]["gamma"] = line.split()[-1]
            elif "_atom_site_occupancy" in line:
                atom_line_idx = line_idx + 1
                break

        for line_idx in range(atom_line_idx, len(lines)):
            line = lines[line_idx]
            if len(line) < 2:
                continue
            atom_info = line.split()
            atom_type = atom_info[0]
            num_atoms = atom_info[2]
            x, y, z = atom_info[3], atom_info[4], atom_info[5]
            extracted_data["atoms"].append({"type": atom_type, "num": num_atoms, "coordinates": (x, y, z)})

        return extracted_data


class CinDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx][:1500]
        # if self.conditions is not None:
        #     raw_input_ids = raw_input_ids[1:]  # Remove the first token (<s>)
        input_ids = text[:-1]
        targets = text[1:]
        return input_ids, targets
