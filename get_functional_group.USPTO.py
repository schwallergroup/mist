import os
import exmol
import pandas as pd
from tqdm import tqdm

data_dir = "/home/vu/Documents/open-r1/data/USPTO_480k_clean_no_sft"

def get_fgs(file_path: str, cache: dict):
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines = [l.strip().replace(" ", "") for l in lines]
        for line in tqdm(lines, desc=f"Processing {file_path}"):
            reactants = line.split(".")
            for reactant in reactants:
                if reactant not in cache:
                    fgs = exmol.get_functional_groups(reactant, cutoff=500)
                    if fgs:
                        fgs = '.'.join(fgs)
                        cache[reactant] = fgs

def main():
    src_train_path = os.path.join(data_dir, data_dir, "src-train.txt")
    src_test_path = os.path.join(data_dir, data_dir, "src-test.txt")

    mol_to_fgs = {}
    get_fgs(src_train_path, mol_to_fgs)
    get_fgs(src_test_path, mol_to_fgs)
    
    mol_to_fgs_df = pd.DataFrame(mol_to_fgs.items(), columns=["smiles", "fgs"])
    mol_to_fgs_df.to_csv(os.path.join(data_dir, "fgs.csv"), index=False)

if __name__ == "__main__":
    main()