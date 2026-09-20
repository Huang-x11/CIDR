import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention(L, D, K)
        self.classifier = Classifier_1fc(L, 1)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x
    
class Attention(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class DTFD(nn.Module):
    def __init__(self, input_dim, distill):
        super(DTFD, self).__init__()
        
        hidden_dim = int(input_dim / 2)
        
        self.distill = distill
        self.dimReduction = DimReduction(input_dim, hidden_dim)
        self.attention = Attention(hidden_dim)
        self.classifier = Classifier_1fc(hidden_dim, 1)
        self.UClassifier = Attention_with_Classifier(L=hidden_dim)
        self.bcewl_loss = nn.BCEWithLogitsLoss()
    
    
    def bag_forward(self, tfeat_tensor, tslideLabel):
        tslideLabel = tslideLabel.unsqueeze(0)
        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []
        
        feat_index = list(range(tfeat_tensor.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), 4)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        
        for tindex in index_chunk_list:
            slide_sub_labels.append(tslideLabel)
            subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).to(tfeat_tensor.device))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:1].long()
            topk_idx_min = sort_idx[-1:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if self.distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif self.distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif self.distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)

        ## optimization for the first tier
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0).squeeze() ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
        loss0 = self.bcewl_loss(slide_sub_preds, slide_sub_labels)
        
        ## optimization for the second tier
        gSlidePred = self.UClassifier(slide_pseudo_feat).flatten()
        loss1 = self.bcewl_loss(gSlidePred, tslideLabel)
        loss = loss0 * 0.5 + loss1
        
        return gSlidePred, loss
        
    def forward(self, x, y, mask):
        ops = [self.bag_forward(bag_x, y[idx]) for idx, bag_x in enumerate(torch.split(x, mask))]
        logits = torch.stack([item[0] for item in ops]).flatten()
        loss = torch.stack([item[1] for item in ops]).mean()
        
        return torch.sigmoid(logits), loss
