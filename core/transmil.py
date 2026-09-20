import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_dim):
        super(TransMIL, self).__init__()
        
        hidden_dim = int(input_dim / 2)
        self.pos_layer = PPEG(dim=hidden_dim)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.layer1 = TransLayer(dim=hidden_dim)
        self.layer2 = TransLayer(dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, 1)
        self.bcewl_loss = nn.BCEWithLogitsLoss()

        
    def bag_forward(self, h):
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1) #.cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]
        
        return h
        

    def forward(self, h, y, mask):        
        h = self._fc1(h) #[B, n, 512]
        h = torch.concat([self.bag_forward(bag.unsqueeze(0)) for bag in torch.split(h, mask)])
        logits = self._fc2(h).squeeze(1)
        
        loss = self.bcewl_loss(logits, y)
        
        return  torch.sigmoid(logits), loss
        