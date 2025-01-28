
import os
from datasets import Dataset
from typing import Dict

class ChemDataset(Dataset):
    def __init__(self, root: str, src_file: str, tgt_file: str):
        """
        Initialize the dataset with source and target file paths
        
        Args:
            src_file (str): Path to source text file
            tgt_file (str): Path to target text file
        """
        src_file = os.path.join(root, src_file)
        tgt_file = os.path.join(root, tgt_file)

        # Read the files
        with open(src_file, 'r', encoding='utf-8') as f:
            src_texts = [line.strip() for line in f.readlines()]
            
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_texts = [line.strip() for line in f.readlines()]
            
        # Create dictionary of features
        self.examples = {
            'source': src_texts,
            'target': tgt_texts
        }
        
        # Convert to HuggingFace Dataset
        self.dataset = Dataset.from_dict(self.examples)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, str]:
        return self.dataset[idx]

# Usage example:
if __name__ == "__main__":
    # Create dataset
    fpath = "data/USPTO"
    dataset = ChemDataset(
        root=fpath,
        src_file='src-train.txt',
        tgt_file='tgt-train.txt'
    )
    
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )
    
    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)

    for batch in dataloader:
        print(batch)
        break
    
