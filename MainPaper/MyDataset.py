import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2

import sys 
import os
sys.path.append(os.path.abspath("../shared_util"))
from preprocess import PreProcess


DATA_CSV_HEADER = ("reference_image_path", "target_image_path", "pspi_score")


class MyDataset(Dataset):
    """Construct my own dataset"""

    def __init__(self, path_to_csv, ops, transform=None):
        """
        Args:
            path_to_csv (string): Path to the csv file
            preprocess (PainDetector): An object of PainDetector
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(path_to_csv)
        self.preprocess = PreProcess(ops)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        reference_image_path = self.df.iloc[idx][DATA_CSV_HEADER[0]]
        target_image_path = self.df.iloc[idx][DATA_CSV_HEADER[1]]
        reference_image = cv2.imread(reference_image_path)
        target_image = cv2.imread(target_image_path)

        reference_image_tensor = self.preprocess.prep_image(reference_image)
        target_image_tensor = self.preprocess.prep_image(target_image)

        input_tensor = torch.cat([reference_image_tensor, target_image_tensor], dim=1).squeeze(dim=0)

        pspi_score = self.df.iloc[idx, 2]
        output_tensor = torch.tensor(pspi_score) 

        return input_tensor, output_tensor