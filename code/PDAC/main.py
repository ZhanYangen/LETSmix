
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
from scipy.spatial.distance import jensenshannon

class Args():
    def __init__(self):
        self.lr = {'sc':0.0001, 'dc':0.0001}
        self.dropout_dc = 0.5
        self.d = 10   # training iterations: domain classifier vs. source classifier
        self.bs = {'source':128, 'target':128, 'target_mix':128}
        self.beta = 2
        self.r = 1
        self.weight = {'target':1, 'target_mix':0, 'domain':1, 'reg':0}
        self.stage1 = {'interval':1000, 'ite':10001}  # pretrain the source classifier
        self.stage2 = {'interval':1000, 'ite':20001, 'reg':5000}  # adversarial training for both classifiers
        self.m = 50  # number of marker genes per cell type
        self.si = 0.5   # degree of spatial refinement
        self.k = 8   # number of cells per pseudo-spot
        self.n_samples = 200000   # total number of pseudo-spots
        self.spa = 0   # whether to use the spatial context information
        self.device = 'cuda:6'
        self.datadir = '/home/zhanyg/data/SpaDA/PDAC/'
        self.st = 'A'  # 'A' or 'B'
        self.sc = 'Peng'  # 'A' or 'B' or 'Peng'
        self.logdir = f'/home/zhanyg/Code/SpaDA/PDAC/log2/{self.st}{self.sc}/test/tes4/'

def model_eval(loader, model, device, mode='val'):
    loss = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            loss.append(model(x, y, mode='eval'))
            if mode == 'train' and i+1 == len(loader)//4:
                break
    return np.mean(loss)

def model_test(loader, adata_st, sc_sub_dict, model, args, show=False):
    pred = []
    with torch.no_grad():
        for x in loader:
            x = x.to(args.device)
            pred.append(model(x, mode='inf'))
    pred = torch.cat(pred).cpu().numpy()

    # ER
    er_all = []
    if args.sc != 'Peng':
        for region in adata_st.uns[f'cts_dom_{args.sc[0]}'].keys():
            idx = np.nonzero(adata_st.obs.region.values==region)[0]
            for ct in adata_st.uns[f'cts_dom_{args.sc[0]}'][region]:
                pred_ct = pred[:,sc_sub_dict[ct]]
                rate = np.sum(pred_ct[idx]) / np.sum(pred_ct)
                er_all.append(rate)
    else:
        for ct in adata_st.uns[f'cts_dom_{args.sc[0]}'].keys():
            idx = []
            for region in adata_st.uns[f'cts_dom_{args.sc[0]}'][ct]:
                idx.extend(np.nonzero(adata_st.obs.region.values==region)[0])
            pred_ct = pred[:,sc_sub_dict[ct]]
            rate = np.sum(pred_ct[idx]) / np.sum(pred_ct)
            er_all.append(rate)

    # JSD
    if args.sc == args.st:
        ctr_st = np.sum(pred, 0) / len(pred)
        jsd = jensenshannon(ctr_st, adata_st.uns['ctr_sc'])
        return {'ER':np.mean(er_all), 'JSD':jsd}, pred
    else:
        return {'ER':np.mean(er_all)}, pred

def evaluation(args, model, loader, adata_sts, sc_sub_dict, 
               res_epoch, t_best, ite, t0, stage):
    global pred_t_best, acc_d, loss_r, res_best
    model.eval()
    loss_s_train = model_eval(loader['s_train'], model, args.device, 'train')
    loss_s_val = model_eval(loader['s_val'], model, args.device)
    res, pred_t = model_test(loader['t_val'], adata_sts, sc_sub_dict, model, args)
    res_epoch['s_train'].append(loss_s_train)
    res_epoch['s_val'].append(loss_s_val)
    res_epoch['t_ER'].append(res['ER'])
    if args.sc == args.st:
        res_epoch['t_JSD'].append(res['JSD'])
        score = res['JSD']+ (1-res['ER'])
    else:
        score = 1-res['ER']
    if score < t_best:
        res_best = res
        t_best = score
        pred_t_best = pred_t
    log = f'S{stage}_ite {ite}, {(time()-t0)/60:.0f}m, '
    if stage == 2:
        log += f'acc_d {np.mean(acc_d):.3f}, loss_r {np.mean(loss_r):.3f}, '
        acc_d = []
        loss_r = []
    log += f'klds_train {loss_s_train:.3f}, klds_val {loss_s_val:.3f}, '
    if args.sc == args.st:
        log += f'JSD {res["JSD"]:.3f}, ER {res["ER"]:.3f}, t_best {t_best:.3f}'
    else:
        log += f'ER {res["ER"]:.3f}, t_best {t_best:.3f}'
    print(log)
    with open(args.logdir+'log.txt', 'a') as f:
        f.write(log+'\n')
    model.train()
    return res_epoch, t_best

def main():
    # load data and the model
    global pred_t_best, acc_d, loss_r, res_best
    args = Args()
    log = [f't{n+1}' for n in range(5)]
    if not (args.logdir[-3:-1] in log):
        for i in log:
            if os.path.exists(args.logdir+i):
                continue
            args.logdir += f'{i}/'
            break
    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)
    os.makedirs(args.logdir+'res/best/', exist_ok=1)
    os.makedirs(args.logdir+'figs/', exist_ok=1)
    with open(args.logdir+'args.txt','w') as f:
        json.dump(args.__dict__, f, indent=2)    
    loader, adata_st, params, sc_sub_dict = load_data(args)
    model = SpaDA([params['dim_in'], 1024, 64, 32, params['dim_out']], args.lr, args.dropout_dc)
    model.to(args.device)
    
    # pretrain the source classifier
    t0 = time()
    with open(args.logdir+'log.txt', 'w') as f:
        f.write(ctime()+'\n')
    res_epoch = {'s_train':[], 's_val':[], 't_JSD':[], 't_ER':[]}
    t_best = np.inf
    ite = 0
    while(1):
        for (xs, ys) in loader['s_train']:
            ite += 1
            if ite == args.stage1['ite']:
                break
            xs, ys = xs.to(args.device), ys.to(args.device)
            model(xs, ys, mode='pretrain')
            if ite % args.stage1['interval'] == 0:
                res_epoch, t_best = evaluation(args, model, loader, adata_st, 
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
                    res_epoch, t_best = evaluation(args, model, loader, adata_st, 
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
                n = lines[-1].strip()[-5:]  
                num.append(float(n))
    fnum = ["{:.3f}".format(n) for n in num]
    print(fnum, f'mean score {np.mean(num):.4f}')

    # plot loss and metric values
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot1 = ax1.plot(res_epoch['s_train'], 'r--', label='s_train_kld', linewidth=0.8)
    plot2 = ax1.plot(res_epoch['s_val'], 'b--', label='s_val_kld', linewidth=0.8)
    plt.xlabel('ite/interval')
    ax2 = ax1.twinx()
    plot3 = ax2.plot(res_epoch['t_ER'], 'r', label='t_ER_ax2', linewidth=0.8)
    if args.sc == args.st:
        plot4 = ax2.plot(res_epoch['t_JSD'], 'b', label='t_JSD_ax2', linewidth=0.8)
        lines = plot1 + plot2 + plot3 + plot4
    else:
        lines = plot1 + plot2 + plot3
    ax1.legend(lines, [l.get_label() for l in lines])
    fig.savefig(args.logdir+'acc.jpg', dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    # plot proportion heatmaps
    pred = pred_t_best
    for i in sc_sub_dict.keys():
        adata_st.obs.loc[:,'pred'] = pred[:,sc_sub_dict[i]]
        sc.pl.spatial(adata_st, img_key="hires", color='pred', cmap='jet',
            palette='Set1', size=1, legend_loc=None, title=i, show=False)
        plt.savefig(args.logdir+f'figs/{i}.jpg', dpi=400, bbox_inches='tight') 
        plt.close()
    pred = pd.DataFrame(pred, columns=list(sc_sub_dict.keys()))
    pred.to_csv(args.logdir+'res/best/pred.csv', index=False)

if __name__ == "__main__":
    main()


