from typing import Literal

from datasets import load_from_disk
from torch.utils.data import Dataset as PTDataset

from config import PROCESSED_DATA_PATH


class HFDataset(PTDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

def get_dataset(type_:Literal['classify', 'summary'], split:Literal['train', 'test']):
    dataset = load_from_disk(PROCESSED_DATA_PATH[type_])[split]
    dataset = HFDataset(dataset)
    return dataset


if __name__ == "__main__":
    dataset = get_dataset('classify', 'train')
    print(dataset[0].keys())