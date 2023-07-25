"""
Module containing functions for training DNN using PyTorch framework
"""

#!pip install sacremoses
from transformers import AutoTokenizer, AdamW
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .pytorch_dataset import Dataset
from .scoring import score, create_result_df
from .pytorch_savebestmodel import SaveBestModel

# set tokenizer
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")

# init SaveBestModel
save_best_model = SaveBestModel()

def run_epoch(dataloader, model, criterion, optimizer, device, step):
    """
    Function runs single epoch
    """
    # initialize place for inputs and labels
    inputs = []
    labels, preds = torch.tensor([]), torch.tensor([])
            
    # initialize loss value  
    total_loss = 0

    # set model to training or evaluation phase
    model.train() if step == 'train' else model.eval()

    for inps, label in tqdm(dataloader):
        # store inputs
        inputs = inputs + [i for i in inps["input_ids"].squeeze(1)]

        # move to proper device
        label = label.to(device)
        
        mask = inps['attention_mask'].to(device)
        inp = inps['input_ids'].squeeze(1).to(device)
        output = model(inp, mask)
            
        output_classes = output.round().view(-1)
                
        labels = torch.cat((labels, label.to("cpu")))
        preds = torch.cat((preds, output_classes.to("cpu")))

        # get batch loss 
        batch_loss = criterion(output.view(-1).float(), label.view(-1).float())
        total_loss += batch_loss.item()
            
        if step == 'train':
            # zero gradients
            model.zero_grad()
            
            # calculate gradients
            batch_loss.backward()
            optimizer.step()
        
    return model, batch_loss, optimizer, inputs, labels, preds, total_loss

def train_model(model, train_data, val_data, learning_rate, epochs, batch_size, custom_model_name):
    """
    Function for training DL model
    """

    print("Setting Datasets...\n")
    # get training and validation Datasets
    train, val = Dataset(train_data), Dataset(val_data)
    
    print("Creating DataLoaders...\n")
    # use DataLoaders on specified Datasets
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    
    # use CUDA to train on GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Setting up optimizer and criterion...\n")
    # specify layers without weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    # decay the rest of layers
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # specify optimizer with learning rate
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # specify Binary Cross Entropy Loss
    criterion = nn.BCELoss()

    if use_cuda:
        print("Training on GPU...\n")
        # move model to GPU
        model = model.cuda()
        criterion = criterion.cuda()
    
    # create empty dataframe for results
    results = pd.DataFrame()
    
    for epoch_num in range(epochs):

        model, batch_loss, optimizer, inputs, labels, preds, total_loss_train = run_epoch(
            dataloader = train_dataloader, 
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            device = device, 
            step = 'train'
        )
        
        # transform tensors to list and get scores
        y_train_true = labels.tolist()
        y_train_pred = preds.detach().tolist()
        scores = score(y_train_true, y_train_pred)
        
        # create validation result dataframe
        results_train = create_result_df(
            notes = [tokenizer.decode(i) for i in inputs],
            labels = y_train_true,
            preds = y_train_pred,
            step = 'train',
            epoch = epoch_num + 1,
            scores = scores
        )
        
        with torch.no_grad():
            
            model, val_batch_loss, optimizer, inputs_val, labels_val, preds_val, total_loss_val = run_epoch(
                dataloader = val_dataloader, 
                model = model, 
                criterion = criterion, 
                optimizer = optimizer, 
                device = device, 
                step = 'val'
            )
        
            # transform validation tensors to list and get scores
            y_val_true = labels_val.tolist()
            y_val_pred = preds_val.tolist()
            val_scores = score(y_val_true, y_val_pred)

            # create validation result dataframe
            results_val = create_result_df(
                notes = [tokenizer.decode(i) for i in inputs_val],
                labels = y_val_true,
                preds = y_val_pred,
                step = 'val',
                epoch = epoch_num + 1,
                scores = val_scores
            )

        save_best_model(
            current_valid_loss = val_batch_loss,
            current_train_loss = batch_loss,
            epoch = epoch_num + 1,
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            train_scores = scores,
            val_scores = val_scores,
            custom_model_name = custom_model_name
        )
        
        results = pd.concat([results, results_train, results_val])

        print(f"Epochs: {epoch_num + 1}")

        print(f"Train Loss: {total_loss_train / len(train_data): .4f} | Validation Loss: {total_loss_val / len(val_data): .4f}")

        for k in scores.keys():
            print(f"Train {k}:{scores.get(k): .4f} | Validation {k}:{val_scores.get(k): .4f}")
                
    return results, model