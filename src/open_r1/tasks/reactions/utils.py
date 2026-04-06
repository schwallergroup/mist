from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def tanimoto_sim(mol1, mol2):
    mol1 = Chem.MolFromSmiles(mol1)
    mol2 = Chem.MolFromSmiles(mol2)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, useChirality=True)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, useChirality=True)

    return DataStructs.TanimotoSimilarity(fp1, fp2)
