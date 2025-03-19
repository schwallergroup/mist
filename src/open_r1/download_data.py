import os

import gdown


def download_data(data_path="data/USPTO"):
    # links from https://github.com/coleygroup/Graph2SMILES/blob/main/scripts/download_raw_data.py
    USPTO_480k_links = [
        (
            "https://drive.google.com/uc?id=1RysNBvB2rsMP0Ap9XXi02XiiZkEXCrA8",
            "src-train.txt",
        ),
        (
            "https://drive.google.com/uc?id=1CxxcVqtmOmHE2nhmqPFA6bilavzpcIlb",
            "tgt-train.txt",
        ),
        (
            "https://drive.google.com/uc?id=1FFN1nz2yB4VwrpWaBuiBDzFzdX3ONBsy",
            "src-val.txt",
        ),
        (
            "https://drive.google.com/uc?id=1pYCjWkYvgp1ZQ78EKQBArOvt_2P1KnmI",
            "tgt-val.txt",
        ),
        (
            "https://drive.google.com/uc?id=10t6pHj9yR8Tp3kDvG0KMHl7Bt_TUbQ8W",
            "src-test.txt",
        ),
        (
            "https://drive.google.com/uc?id=1FeGuiGuz0chVBRgePMu0pGJA4FVReA-b",
            "tgt-test.txt",
        ),
    ]
    os.makedirs(data_path, exist_ok=True)
    for url, name in USPTO_480k_links:
        target_path = os.path.join(data_path, name)
        if not os.path.exists(target_path):
            gdown.download(url, target_path, quiet=False)
        else:
            print(f"{target_path} already exists")


if __name__ == "__main__":
    download_data()
