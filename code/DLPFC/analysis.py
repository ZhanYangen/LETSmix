
import anndata as ad
import matplotlib.pyplot as plt 
import scanpy as sc
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import shutil

def cal_auc(adata_st, pred):
    aucs = []
    for ct in adata_st.uns['cts_dom'].keys():
        pred_ct = pred.loc[:,ct].values
        target = np.zeros(len(pred_ct))
        for region in adata_st.uns['cts_dom'][ct]:
            idx = np.nonzero(adata_st.obs.region.values==region)[0]
            target[idx] = 1
        fpr, tpr, _ = roc_curve(target, pred_ct)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    return aucs

def cal_er(adata_st, pred):
    rates = []
    for ct in adata_st.uns['cts_dom'].keys():
        pred_ct = pred.loc[:,ct].values
        idx = []
        for region in adata_st.uns['cts_dom'][ct]:
            idx.extend(np.nonzero(adata_st.obs.region.values==region)[0])
        rate = np.sum(pred_ct[idx]) / np.sum(pred_ct)
        rates.append(rate)
    return rates


#%% polt region labels
fdir = '/data112/zhanyg/data/SpaDA/DLPFC/'
flist = os.listdir(fdir)
flist = [fname for fname in flist if fname.endswith('.h5ad')]
flist.remove('sc.h5ad')
for f in flist:
    adata = ad.read(fdir+f)
    layers = adata.obs.loc[:,'region'].values.astype(str)
    layers_u = np.unique(layers)
    img = adata.uns['spatial']['histology']['images']['hires']
    scale = adata.uns['spatial']['histology']['scalefactors']['tissue_hires_scalef']
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    for j in layers_u:
        idx = np.nonzero(layers==j)
        plt.scatter(adata.obsm['spatial'][idx,0]*scale, 
                    adata.obsm['spatial'][idx,1]*scale, 
                    2, alpha=0.6, label=j)
    lgnd = plt.legend(bbox_to_anchor=(1.01, 1))
    for handle in lgnd.legendHandles:
        handle.set_sizes([20.0])
    plt.title(f'Sample_{f.split(".")[0]}')
    fig.savefig(f'/data112/zhanyg/data/SpaDA/DLPFC/draft/st_region/{f.split(".")[0]}.jpg', dpi=400, bbox_inches='tight')
    plt.close(fig)


#%% plot region label   split
ast = ad.read('/data112/zhanyg/data/SpaDA/DLPFC/151673.h5ad')
img = ast.uns['spatial']['histology']['images']['hires']
scale = ast.uns['spatial']['histology']['scalefactors']['tissue_hires_scalef']
for ct in ast.uns['cts_dom'].keys():
    idx = []
    for region in ast.uns['cts_dom'][ct]:
        idx.extend(np.nonzero(ast.obs.region.values==region)[0])
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.scatter(ast.obsm['spatial'][idx,0]*scale, 
                ast.obsm['spatial'][idx,1]*scale, 
                2, alpha=0.6, color='r')
    plt.title(f'{ct}')
    fig.savefig(f'/data112/zhanyg/data/SpaDA/DLPFC/draft/region151673/{ct}.jpg', dpi=400, bbox_inches='tight')
    plt.close(fig)


#%% plot sc umap
sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
adata_sc = ad.read('/data112/zhanyg/data/SpaDA/DLPFC/sc.h5ad')
tmp = adata_sc.copy()
sc.pp.normalize_total(tmp)
adata_sc = tmp
sc.tl.pca(adata_sc, svd_solver='arpack')  
sc.pp.neighbors(adata_sc, n_neighbors=10, n_pcs=40)  
sc.tl.umap(adata_sc)  
sc.pl.umap(adata_sc, color='label', show=False)
plt.savefig('/data112/zhanyg/data/SpaDA/DLPFC/draft/sc_umap.jpg', dpi=400, bbox_inches='tight')
plt.close('all')


#%% violin plot   ablation
adata_st = ad.read('/data112/zhanyg/data/SpaDA/DLPFC/151673.h5ad')
basedir = '/home/zhanyg/Code/SpaDA/DLPFC/log/s1/ablation/'
# models = os.listdir(basedir)
models = ['ajrbs', 'ajrb', 'ajrs', 'ajr', 'ajbs', 'wo_img', 'ajrbsl']
names = ['SpaDA', 'w/o sim', 'w/o block', 'w/o s&b', 'w/o l_vec', 'w/o img', 'louvain']
auc_all = {}
for model in models:
    auc_all[model] = []
    for i in range(5):
        pred = pd.read_csv(basedir+model+f'/t{i+1}/pred151673.csv')
        aucs = cal_auc(adata_st, pred)
        auc_all[model].append(np.mean(aucs))
fig = plt.figure()
ax = sns.violinplot(data=list(auc_all.values()))
ax.set_xticklabels(names)
fig.savefig('/data112/zhanyg/data/SpaDA/DLPFC/draft/s1_ablation.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)


#%% box plot model comparision (sample all)
datadir = '/data112/zhanyg/data/SpaDA/DLPFC/'
logdir = '/home/zhanyg/Code/SpaDA/DLPFC/log/sall/'
flist = os.listdir(datadir)
flist = [fname for fname in flist if fname.endswith('.h5ad')]
flist.remove('sc.h5ad')
models = ['SpaDA', 'CellDART', 'SpaDecon', 'Cell2location', 'CARD']
er_all = {}
for model in models:
    er_all[model] = []
for f in flist:
    adata_st = ad.read(datadir + f)
    for model in models:
        for i in range(5):
            pred = pd.read_csv(logdir+model+f'/t{i+1}/pred{f.split(".")[0]}.csv')
            er = cal_er(adata_st, pred)
            er_all[model].append(er)
fig = plt.figure()
ax = sns.boxplot(data=list(auc_all.values()))
ax.set_xticklabels(models)
ax.set_ylabel('ER')
fig.savefig(datadir + 'draft/sall_compare_ER_b50.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)


#%% s1 compare  auc
datadir = '/data112/zhanyg/data/SpaDA/DLPFC/'
adata_st = ad.read(datadir+'151673.h5ad')
logs = ['/home/zhanyg/Code/SpaDA/DLPFC/log/s1/ablation/ajrbs/',
        '/home/zhanyg/Code/SpaDA/DLPFC/log/s1/CellDART/',
        '/home/zhanyg/Code/SpaDA/DLPFC/log/sall/SpaDecon/',
        '/home/zhanyg/Code/SpaDA/DLPFC/log/sall/Cell2location/',
        '/home/zhanyg/Code/SpaDA/DLPFC/log/sall/CARD/']
models = ['SpaDA', 'CellDART', 'SpaDecon', 'Cell2location', 'CARD']
auc_all = {}
for i in range(len(logs)):
    auc_all[models[i]] = []
    for j in range(5):
        pred = pd.read_csv(logs[i]+f't{j+1}/pred151673.csv')
        aucs = cal_auc(adata_st, pred)
        auc_all[models[i]].append(aucs)
fig = plt.figure()
ax = sns.boxplot(data=list(auc_all.values()))
ax.set_xticklabels(models)
ax.set_ylabel('AUC')
fig.savefig(datadir+'draft/s1_compare_b50.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)


#%% s1 compare  3 conditions
datadir = '/data112/zhanyg/data/SpaDA/DLPFC/'
adata_st = ad.read(datadir+'151673.h5ad')
models = ['SpaDA', 'CellDART', 'SpaDecon', 'Cell2location', 'CARD']
conditions = ['original', 'merge', 'del inhib']
data = {}
for m in models:
    data[m] = {}
    for c in conditions:
        data[m][c] = []
logs = ['/home/zhanyg/Code/SpaDA/DLPFC/log/s1/ablation/ajrbs/',
        '/home/zhanyg/Code/SpaDA/DLPFC/log/s1/CellDART/',
        '/home/zhanyg/Code/SpaDA/DLPFC/log/sall/SpaDecon/',
        '/home/zhanyg/Code/SpaDA/DLPFC/log/sall/Cell2location/',
        '/home/zhanyg/Code/SpaDA/DLPFC/log/sall/CARD/']
for i in range(len(logs)):
    for j in range(5):
        pred = pd.read_csv(logs[i]+f't{j+1}/pred151673.csv')
        value = cal_auc(adata_st, pred)
        data[models[i]]['original'].extend(value)
for c in ['merge', 'del inhib']:
    for m in models:
        fdir = f'/home/zhanyg/Code/SpaDA/DLPFC/log2/s1/{c}/{m}/'
        for j in range(5):
            pred = pd.read_csv(fdir+f't{j+1}/pred151673.csv')
            value = cal_auc(adata_st, pred)
            data[m][c].extend(value)
tidy_data = []
for model in models:
    for condition in conditions:
        tidy_data.extend([(model, condition, value) for value in data[model][condition]])
df = pd.DataFrame(tidy_data, columns=['Model', 'Condition', 'AUC'])
# sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='AUC', hue='Condition', data=df, palette='Set3')
plt.legend(loc='upper right')
plt.xlabel('')
plt.savefig(f'/data112/zhanyg/data/SpaDA/DLPFC/draft/s1_c3_b50.jpg', dpi=400, bbox_inches='tight')


#%% plot pred heatmap
datadir = '/data112/zhanyg/data/SpaDA/DLPFC/'
adata_st = ad.read(datadir+'151673.h5ad')
log = '/home/zhanyg/Code/SpaDA/DLPFC/log/sall/SpaDecon/'
for i in range(5):
    if os.path.exists(log+f't{i+1}/figs/'):
        shutil.rmtree(log+f't{i+1}/figs/')
    os.makedirs(log+f't{i+1}/figs/', exist_ok=1)
    pred = pd.read_csv(log+f't{i+1}/pred151673.csv')
    pred = pred.loc[:, ~pred.columns.str.contains('^Unnamed')]
    for ct in pred.columns.values:
        adata_st.obs.loc[:,'pred'] = pred.loc[:,ct].values
        sc.pl.spatial(adata_st, img_key="hires", color='pred', cmap='jet',
            palette='Set1', size=1, legend_loc=None, title=ct, show=False)
        plt.savefig(log+f't{i+1}/figs/151673_{ct}.jpg', dpi=400, bbox_inches='tight') 
        plt.close()


#%% voilin plot for hypers
adata_st = ad.read('/data112/zhanyg/data/SpaDA/DLPFC/151673.h5ad')
log1 = '/home/zhanyg/Code/SpaDA/DLPFC/log/s1/ablation/ajrbs/'
log2 = '/home/zhanyg/Code/SpaDA/DLPFC/log/s1/hyper/'
hypers = ['nmix','nm','nd','aj']
values = [[4,6,8,12],[10,30,50,100],[1,5,10,20],['02','05','1','15']]
xlabels = [[4,6,8,12],[10,30,50,100],[1,5,10,20],[0.2,0.5,1,1.5]]
logs = [2,2,2,1]
titles = ['number of cells per pseudo-spot','number of marker gene per cell type',
          'domain / source','degree of spatial context']
for i in range(len(hypers)):
    hyper = hypers[i]
    value = values[i]
    xlabel = xlabels[i]
    log = logs[i]
    logdir = []
    for v in value:
        logdir.append(log2+hyper+f'/{hyper}{v}/')
    logdir[log] = log1
    auc_all = {}
    for basedir in logdir:
        auc_all[basedir] = []
        for j in range(5):
            pred = pd.read_csv(basedir+f't{j+1}/pred151673.csv')
            aucs = cal_auc(adata_st, pred)
            auc_all[basedir].append(np.mean(aucs))
    fig = plt.figure()
    ax = sns.violinplot(data=list(auc_all.values()))
    ax.set_xticklabels(xlabel)
    plt.ylabel('AUC')
    plt.xlabel(titles[i])
    fig.savefig(f'/data112/zhanyg/data/SpaDA/DLPFC/draft/hypers/{hyper}_v5.jpg', dpi=400, bbox_inches='tight')
    plt.close(fig)








