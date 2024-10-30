
from torch import nn
import torch
import torch.nn.functional as F

class Feature_extractor(nn.Module):
    def __init__(self, dims, act):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dims[0], dims[1]),
                                    nn.BatchNorm1d(dims[1]),
                                    act,
                                    nn.Linear(dims[1], dims[2]),
                                    nn.BatchNorm1d(dims[2]),
                                    act,)
    def forward(self, x):
        return self.layers(x)
    
class Domain_Classifier(nn.Module):
    def __init__(self, dim_in, dim_hid, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            nn.BatchNorm1d(dim_hid),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hid, 2))
    def forward(self, x):
        return self.layers(x)

class Source_Classifier(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim_in, dim_out),
                                    nn.Softmax(1))
    def forward(self, x):
        return self.layers(x)

class LETSmix(nn.Module):
    def __init__(self, dims, lr, d=0.5, act=nn.ELU()):
        super().__init__()
        self.fe = Feature_extractor(dims[:-2], act)
        self.sc = Source_Classifier(dims[-3], dims[-1])
        self.dc = Domain_Classifier(dims[-3], dims[-2], d)
        self.opt = torch.optim.Adam([{'params': self.fe.parameters()},
                                     {'params': self.sc.parameters()}], lr['sc'])
        self.opt_d = torch.optim.Adam(self.dc.parameters(), lr['dc'])
    def forward(self, x, y=None, yd=None, mode='train', w=2):
        z = self.fe(x)
        if mode == 'pretrain':  # ignore the domain classifier
            pred = self.sc(z)
            loss = F.kl_div(torch.log(pred), y, reduction='batchmean')
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return loss.item()
        elif mode == 'inf':   # get predictions for cell-type proportions
            return self.sc(z)
        elif mode == 'dc':   # binary classification training for the domain classifier
            pred = self.dc(z)
            loss = F.cross_entropy(pred, yd) * w
            self.opt_d.zero_grad()
            loss.backward()
            self.opt_d.step()
            pred = torch.argmax(pred.detach(), 1)
            acc = torch.sum(pred==yd) / len(yd)
            return acc.item()
        elif mode == 'train':   # adversarial training for the source classifier
            pred = self.dc(z)
            loss_d = F.cross_entropy(pred, yd)
            pred = self.sc(z[:len(y)])
            loss_s = F.kl_div(torch.log(pred), y, reduction='batchmean')
            loss = loss_s + w*loss_d   
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()






