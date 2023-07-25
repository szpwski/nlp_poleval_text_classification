"""
Package containing modules for modelling classification task
    - scoring <- module containing functions performing cross-validation and scoring results
    - pytorch_training <- module containing functions for training DNN using PyTorch framework
    - pytorch_evaluation <- module containing function to evaluate DNN results using PyTorch framework
    - pytorch_dataset <- module containing PyTorch Dataset class used by DNN models
    - pytorch_herbert <- module containing HerBert classifier class in PyTorch framework
    - pytorch_bilstm <- module containing BiLSTM classifier class in PyTorch framework
    - easy_data_augmentation <- module containing function for EDA (easy data augmentation)
    
"""
from .scoring import score, perform_cross_validation
from .pytorch_training import train_model
from .pytorch_evaluation import evaluate
from .pytorch_dataset import Dataset
from .pytorch_herbert import HerBertForSequenceClassification
from .easy_data_augmentation import random_swap, random_deletion, perform_eda
