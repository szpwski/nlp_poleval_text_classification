"""
Module containing PyTorch Dataset class used by DNN models
"""

#!pip install sacremoses
from transformers import AutoTokenizer
import torch
import numpy as np
import pandas as pd

# set tokenizer
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")

class Dataset(torch.utils.data.Dataset):

    """
    PyTorch Dataset class of our medical data
    """
    
    def __init__(self, df):

        # get labels
        self.labels = df.label.to_numpy()
        
        # get medical notes
        self.texts = [tokenizer.encode_plus(text,
                                # pad sentences with size lower than the max length
                                padding='max_length', 
                                # maximum length avalaible for BERT model
                                max_length = 512, 
                                # truncate sentences exceeding max length
                                truncation=True,
                                # return PyTorch tensors
                                return_tensors="pt"
                               ) for text in df['text']]

    def classes(self):
        # Return classes
        return self.labels

    def __len__(self):
        # Return the length of labels
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    

    def __getitem__(self, idx):

        """
        Get batches for DataLoader
        """

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y