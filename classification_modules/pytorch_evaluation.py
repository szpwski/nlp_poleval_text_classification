"""
Module containing function to evaluate DNN results using PyTorch framework
"""

from transformers import AutoTokenizer, AdamW
import torch
import numpy as np
import pandas as pd
from .pytorch_dataset import Dataset
from .scoring import score, create_result_df

# set tokenizer
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")

def evaluate(model, test_data, threshold = 0.5):

    print("Setting test data as Dataset...\n")
    test = Dataset(test_data)

    print("Setting up DataLoader...\n")
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    
    inputs = []
    labels, preds = torch.tensor([]), torch.tensor([])
    
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            
            inputs = inputs + [i for i in test_input['input_ids'].squeeze(1)]
            test_label = test_label.to(device)
            
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
                
            output_classes = torch.tensor([1 if prob >= threshold else 0 for prob in output]).float().to(device)
            
            labels = torch.cat((labels, test_label.to("cpu")))
            preds = torch.cat((preds, output_classes.to("cpu")))
            
    y_true = labels.tolist()
    y_pred = preds.detach().tolist()
    scores = score(y_true, y_pred)
    
    results = create_result_df(
        notes = [tokenizer.decode(i) for i in inputs],
        labels = y_true,
        preds = y_pred,
        step = 'test',
        epoch = 0,
        scores = scores
    )
                    
    for k in scores.keys():
        print(f"Test {k}: {scores.get(k)}")
    
    return results