"""
Module containing HerBert classifier class in PyTorch framework
"""

import torch.nn as nn
from transformers import RobertaModel

class HerBertForSequenceClassification(nn.Module):
    
    def __init__(self, num_classes, dropout_rate):
        super(HerBertForSequenceClassification, self).__init__()
        
        # Load the pre-trained HerBERT model
        self.herbert = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")
        
        # Create a sequence classification head
        self.classifier = nn.Linear(self.herbert.config.hidden_size, num_classes)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        # Pass the input through the HerBERT model
        outputs = self.herbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the pooled output
        pooled_output = outputs[1]
        
        # Apply dropout layer
        dropout_output = self.dropout(pooled_output)
        
        # Pass the pooled output through the sequence classification head
        logits = self.classifier(dropout_output)
        
        return self.sigmoid(logits)