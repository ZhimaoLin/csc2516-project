import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sys 
import os
sys.path.append(os.path.abspath("../shared_util"))
from preprocess import PreProcess



DATA_SUMMARY_HEADER =  {"person":"person_name", "video":"video_name", "frame":"frame_number", "pspi":"pspi_score", "image":"image_path"}



class MyDataset(Dataset):
    def __init__(self, data_summary_path, ops, transform=None):
        self.df = pd.read_csv(data_summary_path)
        self.preprocess = PreProcess(ops)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pspi_score = self.df.iloc[idx][DATA_SUMMARY_HEADER['pspi']]
        output_tensor = torch.tensor(pspi_score)
        
        image_path = self.df.iloc[idx][DATA_SUMMARY_HEADER['image']]
        image = cv2.imread(image_path)
        image_face, _ = self.preprocess.detect_faces(image)

        if self.transform:
            image_tensor = self.transform(image_face) 
        
        return image_tensor, output_tensor

