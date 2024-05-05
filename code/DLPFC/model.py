
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

class SpaDA(nn.Module):
    def __init__(self, dims, lr, d=0.5, act=nn.ELU()):
        super().__init__()
        self.fe = Feature_extractor(dims[:-2], act)
        self.sc = Source_Classifier(dims[-3], dims[-1])
        self.dc = Domain_Classifier(dims[-3], dims[-2], d)
        self.opt = torch.optim.Adam([{'params': self.fe.parameters()},
                                     {'params': self.sc.parameters()}], lr['sc'])
        self.opt_d = torch.optim.Adam(self.dc.parameters(), lr['dc'])
    def forward(self, x, y=None, xt=None, xt_mix=None, yd=None, mode='train', w=None):
        z = self.fe(x)
        if mode == 'pretrain':  # ignore the domain classifier
            pred = self.sc(z)
            loss = F.kl_div(torch.log(pred), y, reduction='batchmean')
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return loss.item()
        elif mode == 'eval':   # calculate loss for the source classifier
            pred = self.sc(z)
            loss = F.kl_div(torch.log(pred), y, reduction='batchmean')
            return loss.item()
        elif mode == 'inf':   # get predictions for cell-type proportions
            return self.sc(z)
        elif mode == 'dc':   # binary classification training for the domain classifier
            pred = self.dc(self.fe(torch.cat([x,xt,xt_mix['spot_mix']])))
            pred_s = pred[:len(x)]
            pred_t = pred[len(x):len(x)+len(xt)]
            pred_tm = pred[len(x)+len(xt):]
            yds = torch.zeros(len(x), dtype=torch.int64, device=x.device)
            ydt = torch.ones(len(xt), dtype=torch.int64, device=x.device)
            ydtm = torch.ones(len(pred_tm), dtype=torch.int64, device=x.device)
            loss_s = F.cross_entropy(pred_s, yds)
            loss_t = F.cross_entropy(pred_t, ydt)
            loss_tm = F.cross_entropy(pred_tm, ydtm)
            loss = loss_s + w['target']*loss_t + w['target_mix']*loss_tm
            self.opt_d.zero_grad()
            loss.backward()
            self.opt_d.step()
            pred = torch.argmax(pred.detach(), 1)
            acc = torch.sum(pred==yd) / len(yd)
            return acc.item()
        elif mode == 'train':   # adversarial training for the source classifier
            pred = self.dc(self.fe(torch.cat([x,xt,xt_mix['spot_mix']])))
            pred_s = pred[:len(x)]
            pred_t = pred[len(x):len(x)+len(xt)]
            pred_tm = pred[len(x)+len(xt):]
            yds = torch.ones(len(x), dtype=torch.int64, device=x.device)
            ydt = torch.zeros(len(xt), dtype=torch.int64, device=x.device)
            ydtm = torch.zeros(len(pred_tm), dtype=torch.int64, device=x.device)
            loss_s = F.cross_entropy(pred_s, yds)
            loss_t = F.cross_entropy(pred_t, ydt)
            loss_tm = F.cross_entropy(pred_tm, ydtm)
            loss_d = loss_s + w['target']*loss_t + w['target_mix']*loss_tm

            pred = self.sc(z)
            loss_s = F.kl_div(torch.log(pred), y, reduction='batchmean')

            pred = self.sc(self.fe(torch.cat([xt_mix['spot1'], xt_mix['spot2'], xt_mix['spot_mix']])))
            pred_t1 = pred[:len(xt_mix['spot1'])]
            pred_t2 = pred[len(xt_mix['spot1']):len(xt_mix['spot1'])+len(xt_mix['spot2'])]
            pred_tm = pred[len(xt_mix['spot1'])+len(xt_mix['spot2']):]
            loss_r = F.kl_div(torch.log(pred_tm), xt_mix['r']*pred_t1 + (1-xt_mix['r'])*pred_t2, reduction='batchmean')

            loss = loss_s + w['domain']*loss_d + w['reg']*loss_r          
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return loss_r.item()






