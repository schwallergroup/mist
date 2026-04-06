import pandas as pd
from tqdm import tqdm

import pubchempy

src_file = "/iopsstor/store/cscs/swissai/a05/LIAC/data/human_test_rxnpred/src-test.txt"


def main():
    smiles_to_iupac = {}
    with open(src_file, "r") as file:
        for line in tqdm(file):
            line = line.strip()
            smiles = line.split(".")
            for smi in smiles:
                if smi not in smiles_to_iupac:
                    matches = pubchempy.get_compounds(smi, namespace="smiles")
                    if not matches:
                        continue
                    iupac = matches[0].iupac_name
                    smiles_to_iupac[smi] = iupac

    df = pd.DataFrame(list(smiles_to_iupac.items()), columns=["smiles", "iupac"])
    df.to_csv("/iopsstor/store/cscs/swissai/a05/LIAC/data/human_test_rxnpred/iupac_names.csv", index=False)


if __name__ == "__main__":
    main()
