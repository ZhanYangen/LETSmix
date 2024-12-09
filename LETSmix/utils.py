import numpy as np
import torch
from time import time
import matplotlib.pyplot as plt

def cal_ctr_sc(adata_sc, cts_dict):
    """Calculate the general cell type compositioin in the scRNA-seq dataset for the JSD computation"""
    ctr_sc = []
    for ct in cts_dict.keys():
        idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
        ctr_sc.append(len(idx))
    ctr_sc /= np.sum(ctr_sc)
    return ctr_sc

def get_predictions(loader, model, device):
    """Get the deconvolution results for the ST data using LETSmix"""
    pred = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            pred.append(model(x, mode='inf').detach())
    pred = torch.cat(pred).cpu().numpy()
    return pred

def record_log(stage, ite, t0, res, t_best, logdir):
    """Record the training log"""
    log = f'S{stage}_ite {ite}, {(time()-t0)/60:.0f}m, '
    log += f'JSD {res["JSD"]:.3f}, ER {res["ER"]:.3f}, t_best {t_best:.3f}'
    print(log)
    with open(logdir+'log.txt', 'a') as f:
        f.write(log+'\n')

def plot_metric(res_epoch, logdir, show=False):
    """Plot the training metric"""
    fig = plt.figure()
    plt.plot(res_epoch['ER'], 'r', label='ER', linewidth=0.8)
    plt.plot(res_epoch['JSD'], 'b', label='JSD', linewidth=0.8)
    plt.xlabel('ite/interval')
    plt.legend()
    if show:
        plt.show()
    plt.savefig(logdir+'acc.jpg', dpi=400, bbox_inches='tight')


