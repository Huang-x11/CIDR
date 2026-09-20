import torch
import torch.nn as nn

class MeanMIL(nn.Module):
    def __init__(self, input_dim):
        super(MeanMIL, self).__init__()

        hidden_dim = 1 << (input_dim.bit_length() - 1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def mean_bag(self, x):
        x = x.mean(dim=0, keepdims=True)
        return x

    def forward(self, x, y, mask):
        x = self.encoder(x)
        x = torch.concat([self.mean_bag(bag) for bag in torch.split(x, mask)])
        x = self.classifier(x).squeeze(1)
        loss = self.bce_loss(x, y)

        return  torch.sigmoid(x), loss

