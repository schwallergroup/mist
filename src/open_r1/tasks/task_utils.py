"""
Task utilitary functions
"""

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def compute_lcs_length(s1, s2):
    """
    Compute the length of the longest common subsequence (LCS) between two strings (case-sensitive)
    Note: two totally dissimilar strings have an LCS length of 0 (worst)
    Note: the same strings have an LCS length equal to their length (best)
    Note: if a string is longer than the other, the best LCS length is the length of the shorter string (best)
    :param s1: string 1 [str]
    :param s2: string 2 [str]
    :return: LCS length [int]
    """
    # s1 should be longer than s2
    if len(s2) > len(s1):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    # Create a 2D DP table with (m+1) rows and (n+1) columns
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_levenshtein_distance(s1, s2):
    """
    Compute the levenshtein distance between two strings
    Note: levenshtein distance is the "edit distance" from one string to another (in terms of character insertions, deletions, substitutions)
    Note: the same strings have a levenshtein distance of 0 (best)
    Note: the maximum levenshtein distance is the length of the longest string (worst)
    :param s1: string 1 [str]
    :param s2: string 2 [str]
    :return: levenshtein distance [int]
    """
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_tanimoto_similarity(smiles1, smiles2):
    """
    Compute the Tanimoto similarity between two SMILES strings
    Note: range [0, 1] where 0 is completely different and 1 is identical
    :param smiles1: SMILES string 1 [str]
    :param smiles2: SMILES string 2 [str]
    :return: Tanimoto similarity [float] or None if invalid SMILES
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    if mol1 is None:
        return None  # invalid smiles1
    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol2 is None:
        return None  # invalid smiles2
    mol2_fp = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    # Calculate Tanimoto similarity
    tanimoto_similarity = DataStructs.TanimotoSimilarity(mol1_fp, mol2_fp)
    return tanimoto_similarity
