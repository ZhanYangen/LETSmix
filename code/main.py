
import os
import json
import torch
from time import time, ctime
import matplotlib.pyplot as plt 
from model import LETSmix
from data import load_data
import numpy as np
import scanpy as sc
import pandas as pd
import shutil
from scipy.spatial.distance import jensenshannon

class Args():
    def __init__(self):
        self.lr = {'sc':0.0001, 'dc':0.0001}
        self.dropout_dc = 0.5
        self.d = 10   # training iterations: domain classifier vs. source classifier
        self.bs = 128
        self.stage1 = {'interval':1000, 'ite':10001}  # pretrain the source classifier
        self.stage2 = {'interval':1000, 'ite':20001}  # adversarial training for both classifiers
        self.m = 50  # number of marker genes per cell type
        self.si = 0.5   # degree of spatial refinement
        self.k = 8   # number of cells per pseudo-spot
        self.device = 'cuda:4'
        self.datadir = '/data114_2/zhanyg/data/trans/PDAC/'
        self.st = 'A'  # 'A' or 'B'
        self.sc = 'A'  # 'A' or 'B' or 'Peng'
        self.logdir = '/data114_2/zhanyg/Code/LETSmix/log/exp1/'

def model_test(loader, adata_st, sc_sub_dict, model, args):
    pred = []
    with torch.no_grad():
        for x in loader:
            x = x.to(args.device)
            pred.append(model(x, mode='inf').detach())
    pred = torch.cat(pred).cpu().numpy()

    # ER
    er_all = []
    if args.sc != 'Peng':
        for region in adata_st.uns[f'cts_dom_{args.sc[0]}'].keys():
            idx = np.nonzero(adata_st.obs.region.values==region)[0]
            pred_ct = np.zeros(len(pred))
            for ct in adata_st.uns[f'cts_dom_{args.sc[0]}'][region]:
                pred_ct += pred[:,sc_sub_dict[ct]]
            if np.sum(pred_ct) == 0:
                er_all.append(0)
            else:
                er_all.append(np.sum(pred_ct[idx]) / np.sum(pred_ct))
    else:
        for ct in adata_st.uns[f'cts_dom_{args.sc[0]}'].keys():
            idx = []
            for region in adata_st.uns[f'cts_dom_{args.sc[0]}'][ct]:
                idx.extend(np.nonzero(adata_st.obs.region.values==region)[0])
            pred_ct = pred[:,sc_sub_dict[ct]]
            if np.sum(pred_ct) == 0:
                er_all.append(0)
            else:
                er_all.append(np.sum(pred_ct[idx]) / np.sum(pred_ct))
    # er_all = np.sqrt(np.array(er_all))

    # JSD
    if args.sc == args.st:
        ctr_st = np.sum(pred, 0) / len(pred)
        jsd = jensenshannon(ctr_st, adata_st.uns['ctr_sc'])
        return {'ER':np.mean(er_all), 'JSD':jsd}, pred
    else:
        return {'ER':np.mean(er_all)}, pred

def evaluation(args, model, loader, adata_st, sc_sub_dict, ite, t0, stage):
    global pred_t_best, acc_d, res_epoch, t_best
    model.eval()
    res, pred_t = model_test(loader['t_val'], adata_st, sc_sub_dict, model, args)
    res_epoch['ER'].append(res['ER'])
    if args.sc == args.st:
        res_epoch['JSD'].append(res['JSD'])
        score = res['JSD']+ (1-res['ER'])
    else:
        score = 1-res['ER']
    if score < t_best:
        t_best = score
        pred_t_best = pred_t
    log = f'S{stage}_ite {ite}, {(time()-t0)/60:.0f}m, '
    if stage == 2:
        log += f'acc_d {np.mean(acc_d):.3f}, '
        acc_d = []
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
    global pred_t_best, acc_d, res_epoch, t_best
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
    os.makedirs(args.logdir+'figs/', exist_ok=1)
    with open(args.logdir+'args.txt','w') as f:
        json.dump(args.__dict__, f, indent=2)    
    t0 = time()
    with open(args.logdir+'log.txt', 'w') as f:
        f.write(ctime()+'\n')
    loader, adata_st, params, sc_sub_dict = load_data(args)
    model = LETSmix([params['dim_in'], 1024, 64, 32, params['dim_out']], args.lr, args.dropout_dc)
    model.to(args.device)
    
    # pretrain the source classifier
    res_epoch = {'JSD':[], 'ER':[]}
    t_best = np.inf
    ite = 0
    while(1):
        for (xs, ys) in loader['s_train']:
            ite += 1
            if ite == args.stage1['ite']:
                break
            xs, ys = xs.to(args.device), ys.to(args.device)
            loss = model(xs, ys, mode='pretrain')
            if ite % args.stage1['interval'] == 0:
                evaluation(args, model, loader, adata_st, sc_sub_dict, ite, t0, 1)
        if ite == args.stage1['ite']:
            break

    # adversarial training for both classifiers
    t_best = np.inf
    ite = 0
    acc_d = []
    yd = torch.cat([torch.zeros(args.bs), torch.ones(args.bs)]).to(torch.int64).to(args.device)
    count = 0
    while(1):
        for xs, ys, xt in loader['t_train']:
            count += 1
            xs, ys, xt = xs.to(args.device), ys.to(args.device), xt.to(args.device)
            acc_d.append(model(torch.cat([xs, xt]), ys, yd, 'dc'))
            if count % args.d == 0:
                ite += 1
                if ite == args.stage2['ite']:
                    break
                yd = (yd==0).long()
                model(torch.cat([xs, xt]), ys, yd, 'train')
                yd = (yd==0).long()
                if ite % args.stage2['interval'] == 0:
                    evaluation(args, model, loader, adata_st, sc_sub_dict, ite, t0, 2)
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
    plt.plot(res_epoch['ER'], 'r', label='ER', linewidth=0.8)
    if args.sc == args.st:
        plt.plot(res_epoch['JSD'], 'b', label='JSD', linewidth=0.8)
    plt.xlabel('ite/interval')
    plt.legend()
    plt.savefig(args.logdir+'acc.jpg', dpi=400, bbox_inches='tight')
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
    pred.to_csv(args.logdir+'pred.csv', index=False)

if __name__ == "__main__":
    main()


