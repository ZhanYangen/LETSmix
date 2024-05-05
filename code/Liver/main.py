
import os
import json
import torch
from time import time, ctime
import matplotlib.pyplot as plt 
from model import SpaDA
from data import load_data
import numpy as np
import scanpy as sc
from sklearn.metrics import roc_curve, auc
import pandas as pd
import shutil

class Args():
    def __init__(self):
        self.lr = {'sc':0.0001, 'dc':0.0001}
        self.dropout_dc = 0.5
        self.d = 10   # training iterations: domain classifier vs. source classifier
        self.bs = {'source':256, 'target':256, 'target_mix':256}
        self.beta = 2
        self.weight = {'target':1, 'target_mix':0, 'domain':1, 'reg':0}
        self.stage1 = {'interval':500, 'ite':6001}  # pretrain the source classifier
        self.stage2 = {'interval':500, 'ite':16001, 'reg':6000}  # adversarial training for both classifiers
        self.m = 100  # number of marker genes per cell type
        self.si = 0.5   # degree of spatial refinement
        self.k = 8   # number of cells per pseudo-spot
        self.n_samples = 200000   # total number of pseudo-spots
        self.spa = 1   # whether to use the spatial context information
        self.device = 'cuda:7'
        self.datadir = '/home/zhanyg/data/SpaDA/Liver/'
        self.st = [1,2,3]
        self.sc = 'inVivo'
        self.logdir = '/home/zhanyg/Code/LETSmix/code/Liver/log/test/draft/t1/'

def model_eval(loader, model, device, mode='val'):
    loss = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            loss.append(model(x, y, mode='eval'))
            if mode == 'train' and i+1 == len(loader)//4:
                break
    return np.mean(loss)

def model_test(loaders, adata_sts, sc_sub_dict, model, args):
    pred_all = []
    metric = {'AUC':[], 'ER':[]}
    for i in range(len(loaders)):  # load each ST sample
        loader = loaders[i]
        adata_st = adata_sts[i]
        pred = []
        with torch.no_grad():
            for x in loader:
                x = x.to(args.device)
                pred.append(model(x, mode='inf'))
        pred = torch.cat(pred).cpu().numpy()
        pred_all.append(pred)

        # calculate ER values
        for ct in adata_st.uns['cts_dom'].keys():
            pred_ct = pred[:,sc_sub_dict[ct]]
            idx = []
            for region in adata_st.uns['cts_dom'][ct]:
                idx.extend(np.nonzero(adata_st.obs.region.values==region)[0])
            er = np.sum(pred_ct[idx]) / np.sum(pred_ct)
            metric['ER'].append(er)

        # calculate AUC values
        for ct in adata_st.uns['cts_dom'].keys():
            pred_ct = pred[:,sc_sub_dict[ct]]
            target = np.zeros(len(pred_ct))
            for region in adata_st.uns['cts_dom'][ct]:
                idx = np.nonzero(adata_st.obs.region.values==region)[0]
                target[idx] = 1
            fpr, tpr, _ = roc_curve(target, pred_ct)
            roc_auc = auc(fpr, tpr)
            metric['AUC'].append(roc_auc)
    return {'ER':np.mean(metric['ER']), 'AUC':np.mean(metric['AUC'])}, pred_all

def evaluation(args, model, loader, adata_sts, sc_sub_dict, 
               res_epoch, t_best, ite, t0, stage):
    global pred_t_best, acc_d, loss_r
    model.eval()
    loss_s_train = model_eval(loader['s_train'], model, args.device, 'train')
    loss_s_val = model_eval(loader['s_val'], model, args.device)
    res, pred_t = model_test(loader['t_val'], adata_sts, sc_sub_dict, model, args)
    res_epoch['s_train'].append(loss_s_train)
    res_epoch['s_val'].append(loss_s_val)
    res_epoch['t_ER'].append(res['ER'])
    res_epoch['t_AUC'].append(res['AUC'])
    if res['ER'] > t_best:
        t_best = res['ER']
        pred_t_best = pred_t
    log = f'S{stage}_ite {ite}, {(time()-t0)/60:.0f}m, '
    if stage == 2:
        log += f'acc_d {np.mean(acc_d):.3f}, loss_r {np.mean(loss_r):.3f}, '
        acc_d = []
        loss_r = []
    log += f'klds_train {loss_s_train:.3f}, klds_val {loss_s_val:.3f}, '
    log += f'ER {res["ER"]:.3f}, AUC {res["AUC"]:.3f}, t_best {t_best:.3f}'
    print(log)
    with open(args.logdir+'log.txt', 'a') as f:
        f.write(log+'\n')
    model.train()
    return res_epoch, t_best

def main():
    # load data and the model
    global pred_t_best, acc_d, loss_r
    args = Args()
    log = [f't{n}' for n in range(5)]
    if not (args.logdir[-3:-1] in log):
        for i in log:
            if os.path.exists(args.logdir+i):
                continue
            args.logdir += f'{i}/'
            break
    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)
    os.makedirs(args.logdir+'figs/', exist_ok=1)
    with open(args.logdir+'args.txt','w') as f:
        json.dump(args.__dict__, f, indent=2)    
    loader, adata_sts, params, sc_sub_dict, ctr_sc = load_data(args)
    model = SpaDA([params['dim_in'], 1024, 64, 32, params['dim_out']], args.lr, args.dropout_dc)
    model.to(args.device)
    
    # pretrain the source classifier
    t0 = time()
    with open(args.logdir+'log.txt', 'w') as f:
        f.write(ctime()+'\n')
    res_epoch = {'s_train':[], 's_val':[], 't_JSD':[], 't_ER':[]}
    t_best = np.inf  # best performance on the target domain
    ite = 0
    while(1):
        for (xs, ys) in loader['s_train']:
            ite += 1
            if ite == args.stage1['ite']:
                break
            xs, ys = xs.to(args.device), ys.to(args.device)
            model(xs, ys, mode='pretrain')
            if ite % args.stage1['interval'] == 0:
                res_epoch, t_best = evaluation(args, model, loader, adata_sts, ctr_sc, 
                                               sc_sub_dict, res_epoch, t_best, ite, t0, 1)
        if ite == args.stage1['ite']:
            break

    # adversarial training for both classifiers
    t_best = np.inf
    ite = 0
    acc_d = []
    loss_r = []
    count = 0
    wr = args.weight['reg']
    args.weight['reg'] = 0
    while(1):
        for xs, ys, xt, xt_mix in loader['train']:
            count += 1
            xs, ys, xt = xs.to(args.device), ys.to(args.device), xt.to(args.device)
            for key in xt_mix.keys():
                xt_mix[key] = xt_mix[key].to(args.device)
            yd = torch.cat([torch.zeros(len(xs)), torch.ones(len(xt)+len(xt_mix['spot_mix']))]).to(torch.int64).to(args.device)
            acc_d.append(model(xs, ys, xt, xt_mix, yd, 'dc', args.weight))
            if count % args.d == 0:
                ite += 1
                if ite == args.stage2['ite']:
                    break
                if ite == args.stage2['reg']:
                    args.weight['reg'] = wr
                yd = (yd==0).long()
                loss_r.append(model(xs, ys, xt, xt_mix, yd, 'train', args.weight))
                if ite % args.stage2['interval'] == 0:
                    res_epoch, t_best = evaluation(args, model, loader, adata_sts, ctr_sc, 
                                                   sc_sub_dict, res_epoch, t_best, ite, t0, 2)
        if ite == args.stage2['ite']:
            break
    print('log saved: ' + args.logdir)
    logdir = args.logdir[:-4]
    fname = os.listdir(logdir)
    num = []
    for f in fname:
        with open(f'{logdir}/{f}/log.txt', 'r') as f:
            lines = f.readlines()
            if f"S2_ite {args.stage2['ite']-1}" in lines[-1].strip():
                n = lines[-2].strip()[-5:]  
                num.append(float(n))
    fnum = ["{:.3f}".format(n) for n in num]
    print(fnum, f'mean score {np.mean(num):.4f}')

    # plot loss and metric values
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # plot1 = ax1.plot(res_epoch['s_train'], 'r--', label='s_train')
    # plot2 = ax1.plot(res_epoch['s_val'], 'b--', label='s_val')
    # plt.xlabel('ite/interval')
    # plt.ylabel('kld')
    # ax2 = ax1.twinx()
    # plot3 = ax2.plot(res_epoch['t_ER'], 'g', label='t_ER_ax2')
    # plot4 = ax2.plot(res_epoch['t_AUC'], 'black', label='t_AUC_ax2')
    # plt.ylabel('rate')
    # lines = plot1 + plot2 + plot3 + plot4
    # ax1.legend(lines, [l.get_label() for l in lines])
    # fig.savefig(args.logdir+'acc.jpg', dpi=400, bbox_inches='tight')
    # plt.close(fig)
    
    # plot proportion heatmaps
    for j in range(len(adata_sts)):
        pred = pred_t_best[j]
        adata_st = adata_sts[j]
        # for i in sc_sub_dict.keys():
        for i in ['Central Vein Endothelial cells', 'Portal Vein Endothelial cells']:
            adata_st.obs.loc[:,'pred'] = pred[:,sc_sub_dict[i]]
            sc.pl.spatial(adata_st, img_key="hires", color='pred', cmap='jet',
                palette='Set1', size=1, legend_loc=None, title=i, show=False)
            plt.savefig(args.logdir+f'figs/JBO00{j+1}_{i}.jpg', dpi=400, bbox_inches='tight') 
            plt.close()
        pred = pd.DataFrame(pred, columns=list(sc_sub_dict.keys()))
        pred.to_csv(args.logdir+f'pred_JBO00{j+1}.csv', index=False)

if __name__ == "__main__":
    main()


