import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class ABMIL(nn.Module):
    def __init__(self, input_dim):
        super(ABMIL, self).__init__()
        
        hidden_dim = 1 << (input_dim.bit_length() - 1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Linear(hidden_dim, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def attention_mil(self, x, y=None):
        A = self.attention(x)
        A = torch.transpose(A, 1, 0)
        A = torch.softmax(A, dim=1)
        x = torch.mm(A, x)

        return x

    def forward(self, x, y, mask):
        x = self.encoder(x)
        x = torch.concat([self.attention_mil(bag) for bag in torch.split(x, mask)])        
        x = self.classifier(x).squeeze(1)
        loss = self.bce_loss(x, y)

        return  torch.sigmoid(x), loss
