import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CLAM_SB(nn.Module):
    def __init__(self, embed_dim, gate = True, size_arg = "small", dropout = 0., k_sample=4, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.ce_loss = nn.CrossEntropyLoss()
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss

    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss


    def bag_forward(self, h, label, instance_eval=True):
        label = label.long()
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN

        A_raw = A
        A = F.softmax(A, dim=1)  
        
        total_inst_loss = 0.0
        all_preds = []
        all_targets = []
        inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
        for i in range(len(self.instance_classifiers)):
            inst_label = inst_labels[i].item()
            classifier = self.instance_classifiers[i]
            if inst_label == 1: #in-the-class:
                instance_loss = self.inst_eval(A, h, classifier)
            else: 
                instance_loss = self.inst_eval_out(A, h, classifier)
            total_inst_loss += instance_loss
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        loss = self.ce_loss(logits, label.unsqueeze(0)) + total_inst_loss

        return logits, loss
    
    def forward(self, x, y, mask):
        ops = [self.bag_forward(bag_x, y[idx]) for idx, bag_x in enumerate(torch.split(x, mask))]
        logits = torch.softmax(torch.concat([item[0] for item in ops]), dim=1)
        loss = torch.stack([item[1] for item in ops]).mean()
        
        return logits[:, 1], loss
    
    
    
class CLAM_MB(CLAM_SB):
    def __init__(self, embed_dim, gate = True, size_arg = "small", dropout = 0., k_sample=4, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def bag_forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        label = label.long()
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        
        total_inst_loss = 0.0
        all_preds = []
        all_targets = []
        inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
        for i in range(len(self.instance_classifiers)):
            inst_label = inst_labels[i].item()
            classifier = self.instance_classifiers[i]
            if inst_label == 1:
                instance_loss = self.inst_eval(A[i], h, classifier)
            else:
                instance_loss = self.inst_eval_out(A[i], h, classifier)
            total_inst_loss += instance_loss

        total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
            
        loss = self.ce_loss(logits, label.unsqueeze(0)) + total_inst_loss

        return logits, loss
    
    
    def forward(self, x, y, mask):
        ops = [self.bag_forward(bag_x, y[idx]) for idx, bag_x in enumerate(torch.split(x, mask))]
        logits = torch.softmax(torch.concat([item[0] for item in ops]), dim=1)
        loss = torch.stack([item[1] for item in ops]).mean()
        
        return logits[:, 1], loss