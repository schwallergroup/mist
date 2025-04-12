import os
from tqdm import tqdm
import argparse
import pandas as pd
import random
from verl.utils.hdfs_io import copy, makedirs
from AIRS_preporcess._tokenizer import CIFTokenizer

# Initialize the tokenizer
cif_tokenizer = CIFTokenizer()

def load_cif_dataset(binary_dir: str, perturbed_dir: str, size: int, local_dir: str) -> list:
    """
    Load the dataset:
    - Read a parquet dataframe from local_dir (assumed to be "perturbed_df_cif.parquet").
    - Load the ground truth CIF file and the perturbed CIF file based on material_id.
    - Serialize the loaded content using cif_tokenizer.serialize.
    """
    parquet_path = os.path.join(local_dir, "perturbed_df_cif.parquet")
    df = pd.read_parquet(parquet_path)
    df = df.head(size)
    samples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading CIF dataset"):
        material_id = row['material_id']
        # Extract the ground truth file name from material_id
        ground_truth_material_id = material_id.split("_random_")[0]
        gt_file = os.path.join(binary_dir, f"{ground_truth_material_id}.cif")
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt_content = f.read()
        except Exception as e:
            print(f"Error reading ground truth file {gt_file}: {e}")
            continue
        
        perturbed_file = os.path.join(perturbed_dir, f"{material_id}.cif")
        try:
            with open(perturbed_file, 'r', encoding='utf-8') as f:
                perturbed_content = f.read()
        except Exception as e:
            print(f"Error reading perturbed file {perturbed_file}: {e}")
            continue
        
        # Note: Both ground truth and perturbed content are serialized here.
        sample = {
            "compound_id": material_id,
            "ground_truth": cif_tokenizer.serialize(gt_content),
            "perturbed": cif_tokenizer.serialize(perturbed_content)
        }
        print("material_id:", material_id)
        print("ground_truth:", cif_tokenizer.serialize(gt_content))
        print("perturbed:", cif_tokenizer.serialize(perturbed_content))
        samples.append(sample)
    
    print(f"{len(samples)} samples loaded")
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_dir', 
        default='./binary_compounds_dataset', 
        help="Directory where the dataset is saved locally"
    )
    parser.add_argument(
        '--hdfs_dir', 
        default=None, 
        help="HDFS directory (optional)"
    )
    parser.add_argument(
        '--train_size', 
        type=int, 
        default=500, 
        help="Number of training set samples"
    )
    parser.add_argument(
        '--test_size', 
        type=int, 
        default=100, 
        help="Number of test set samples"
    )
    args = parser.parse_args()
    
    # Construct the paths for CIF files
    binary_dir = os.path.join(args.local_dir, 'binary_compounds_cifs')
    perturbed_dir = os.path.join(args.local_dir, 'perturbed_binary_compounds_cifs')
    
    samples = load_cif_dataset(binary_dir, perturbed_dir, args.train_size + args.test_size, args.local_dir)
    random.shuffle(samples)
    total_samples = len(samples)
    print(f"Total number of samples loaded: {total_samples}")
    
    # If the number of samples is insufficient, use all samples as the training set.
    if total_samples < (args.train_size + args.test_size):
        print("Warning: Insufficient samples, will use all samples as training set.")
        train_samples = samples
        test_samples = []
    else:
        train_samples = samples[:args.train_size]
        test_samples = samples[args.train_size:args.train_size + args.test_size]
    
    # Construct the file paths for generating the dataset required by BinaryCompoundRelaxing.
    src_train_path = os.path.join(args.local_dir, 'src-train.txt')
    tgt_train_path = os.path.join(args.local_dir, 'tgt-train.txt')
    src_test_path = os.path.join(args.local_dir, 'src-test.txt')
    tgt_test_path = os.path.join(args.local_dir, 'tgt-test.txt')
    
    # Write the training set text: each line of the question uses the 'perturbed' field, and the corresponding answer uses the 'ground_truth' field.
    with open(src_train_path, 'w', encoding='utf-8') as f_src, \
         open(tgt_train_path, 'w', encoding='utf-8') as f_tgt:
        for sample in train_samples:
            f_src.write(sample['perturbed'] + "\n")
            f_tgt.write(sample['ground_truth'] + "\n")
    
    # Write the test set text files.
    if test_samples:
        with open(src_test_path, 'w', encoding='utf-8') as f_src, \
             open(tgt_test_path, 'w', encoding='utf-8') as f_tgt:
            for sample in test_samples:
                f_src.write(sample['perturbed'] + "\n")
                f_tgt.write(sample['ground_truth'] + "\n")
    else:
        # If the test set is empty, create empty files.
        open(src_test_path, 'w', encoding='utf-8').close()
        open(tgt_test_path, 'w', encoding='utf-8').close()
    
    # If an HDFS directory is specified, copy the local_dir to HDFS.
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
