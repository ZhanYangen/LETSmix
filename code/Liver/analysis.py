
import anndata as ad
import matplotlib.pyplot as plt 
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import itertools
from tqdm import tqdm
import shutil
from scipy import stats
from scipy.spatial.distance import jensenshannon

#%%  discard outliers
n = 3
ast = ad.read(f'/home/zhanyg/data/SpaDA/Liver/st_JBO00{n}.h5ad')
# a = np.sum(ast.X, 1)
coord = ast.obsm['spatial']
dist = np.linalg.norm(coord[:, np.newaxis, :] - coord, axis=2)
np.fill_diagonal(dist, np.inf)
a = np.min(dist, axis=1)
b = np.mean(a)
c = np.std(a)
d = np.nonzero(a>(b+2*c))[0]
idx = set(np.arange(len(ast))) - set(d)

adata = ast[list(idx)]
img = adata.uns.data['spatial']['histology']['images']['hires']
scale = adata.uns.data['spatial']['histology']['scalefactors']['tissue_hires_scalef']
layers = adata.obs.loc[:,'region'].values.astype(str)
layers_u = np.unique(layers)
fig = plt.figure()
plt.imshow(img)
plt.axis('off')
for j in layers_u:
    idx = np.nonzero(layers==j)[0]
    plt.scatter(adata.obsm['spatial'][idx,0]*scale, 
                adata.obsm['spatial'][idx,1]*scale, 
                1, alpha=1, label=j)
lgnd = plt.legend(bbox_to_anchor=(1.01, 1))
for handle in lgnd.legendHandles:
    handle.set_sizes([20.0])
plt.title(f'Liver-JBO001')
fig.savefig(f'/home/zhanyg/Code/draft/fig.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)
adata.write(f'/home/zhanyg/data/SpaDA/Liver/st_JBO00{n}.h5ad')


#%% polt region label
for stn in range(1, 4):
    adata = ad.read(f'/home/zhanyg/data/SpaDA/Liver/st_JBO00{stn}.h5ad')
    img = adata.uns.data['spatial']['histology']['images']['hires']
    scale = adata.uns.data['spatial']['histology']['scalefactors']['tissue_hires_scalef']
    layers = adata.obs.loc[:,'region'].values.astype(str)
    layers_u = np.unique(layers)
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    for j in layers_u:
    # for j in ['Central']:
        idx = np.nonzero(layers==j)[0]
        plt.scatter(adata.obsm['spatial'][idx,0]*scale, 
                    adata.obsm['spatial'][idx,1]*scale, 
                    1, alpha=1, label=j)
    lgnd = plt.legend(bbox_to_anchor=(1.01, 1))
    for handle in lgnd.legendHandles:
        handle.set_sizes([20.0])
    plt.title(f'Liver-JBO00{stn}')
    fig.savefig(f'/home/zhanyg/data/SpaDA/Liver/draft/region/region_st_JBO00{stn}.jpg', dpi=400, bbox_inches='tight')
    plt.close(fig)


#%% polt region label   split
for stn in range(1, 2):
    adata = ad.read(f'/home/zhanyg/data/SpaDA/Liver/st_JBO00{stn}.h5ad')
    img = adata.uns.data['spatial']['histology']['images']['hires']
    scale = adata.uns.data['spatial']['histology']['scalefactors']['tissue_hires_scalef']
    layers = adata.obs.loc[:,'region'].values.astype(str)
    layers_u = ['Central', 'Portal']
    for j in layers_u:
        fig = plt.figure()
        plt.imshow(img)
        plt.axis('off')
        idx = np.nonzero(layers==j)[0]
        plt.scatter(adata.obsm['spatial'][idx,0]*scale, 
                    adata.obsm['spatial'][idx,1]*scale, 
                    1, 'r', alpha=1, label=j)
        plt.title(f'Liver-JBO00{stn}')
        fig.savefig(f'/home/zhanyg/data/SpaDA/Liver/draft/region/region_st_JBO00{stn}_{j}.jpg', dpi=400, bbox_inches='tight')
        plt.close(fig)


#%% plot sc umap
scn = 'inVivo'
sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
adata_sc = ad.read(f'/home/zhanyg/data/SpaDA/Liver/sc_{scn}.h5ad')
tmp = adata_sc.copy()
sc.pp.normalize_total(tmp)
adata_sc = tmp
sc.tl.pca(adata_sc, svd_solver='arpack')  
sc.pp.neighbors(adata_sc, n_neighbors=10, n_pcs=40)  
sc.tl.umap(adata_sc)  
sc.pl.umap(adata_sc, color='label', show=False)
plt.title(f'Liver-{scn}')
plt.savefig(f'/home/zhanyg/data/SpaDA/Liver/draft/region/sc_{scn}_umap.jpg', dpi=400, bbox_inches='tight')
# sc.tl.rank_genes_groups(adata_sc, 'label', method='wilcoxon')
# sc.pl.rank_genes_groups(adata_sc, n_genes=20, sharey=False, show=False)
# plt.savefig('/data112/zhanyg/data/SpaDA/DLPFC/draft/sc_gene.jpg', dpi=400, bbox_inches='tight')
plt.close('all')


#%% plot pred map
pred = pd.read_csv(f'/home/zhanyg/Code/SpaDA/Liver/log/inVivo/SpaDA2/t3/res/best/pred_JBO001.csv')
pred = pred.loc[:,'Central Vein Endothelial cells'].values
idx = pred > 0.03
pred[idx] = 0.03
adata_st = ad.read(f'/home/zhanyg/data/SpaDA/Liver/st_JBO001.h5ad')
# for ct in pred.columns.values[1:]:
    # adata_st.obs.loc[:,'pred'] = pred.loc[:,ct].values
adata_st.obs.loc[:,'pred'] = pred
sc.pl.spatial(adata_st, img_key="hires", color='pred', cmap='jet',
    palette='Set1', size=1, legend_loc=None, title='Central Vein Endothelial cells', show=False)
plt.savefig(f'/home/zhanyg/data/SpaDA/Liver/draft/prop/inVivo.jpg', dpi=400, bbox_inches='tight') 
plt.close()


#%% plot proportion barh
name = 'inVivo'
adata_sc = ad.read(f'/home/zhanyg/data/SpaDA/Liver/sc_nuclei.h5ad')
idx = []
for sample in ['ABU11', 'ABU13', 'ABU17', 'ABU20']:
    idx.extend(np.nonzero(adata_sc.obs.loc[:,'sample'].values==sample)[0])
adata_sc = adata_sc[idx].copy()
ctr_sc = []
cell_types = np.unique(adata_sc.obs.label.values)
for ct in cell_types:
    idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
    ctr_sc.append(len(idx))
ctr_sc /= np.sum(ctr_sc)
models = ['CARD', 'Cell2location', 'SpaDecon', 'CellDART', 'SpaDA']
prop = {}
for model in models:
    prop[model] = []
    ctr_st = np.zeros([5*3,len(ctr_sc)])
    for i in range(5):
        for j in range(3):
            pred = pd.read_csv(f'/home/zhanyg/Code/SpaDA/Liver/log2/{name}/{model}/t{i+1}/res/best/pred_JBO00{j+1}.csv')
            pred = pred.loc[:,cell_types]
            ctr_st[i*3+j] = np.sum(pred, 0) / len(pred)
    ctr_st = np.sum(ctr_st, 0) / len(ctr_st)
    prop[model].extend(ctr_st)
# jsds = []
# for model in prop.keys():
#     jsd = jensenshannon(prop[model], ctr_sc)
#     jsds.append(jsd)
models = ['CARD', 'Cell2location', 'SpaDecon', 'CellDART', 'LETSmix']
prop['LETSmix'] = prop['SpaDA']

fig, ax = plt.subplots()
# colors = plt.cm.tab20(np.linspace(0, 1, len(ctr_sc)))
colors_all = itertools.cycle(plt.cm.tab20.colors)
colors = [next(colors_all) for _ in cell_types]
position = 0
for model in models:
    for value, color in zip(prop[model], colors):
        width = value  # 每个部分的宽度正比于a中的值
        ax.barh(model, width, left=position, color=color)
        position += width
    position = 0
for value, color, ct in zip(ctr_sc, colors, cell_types):
    width = value  # 每个部分的宽度正比于a中的值
    ax.barh('scRNA-seq', width, left=position, color=color, label=ct)
    position += width
plt.xticks([])
plt.xlabel('proportion of cell types')
plt.legend(bbox_to_anchor=(1.01, 1))
plt.title(f'{name}')
plt.savefig(f'/home/zhanyg/data/SpaDA/Liver/draft/indc/barh_{name}.jpg', dpi=400, bbox_inches='tight')


#%% plot sc barh
adata = []
adata.append(ad.read(f'/home/zhanyg/data/SpaDA/Liver/sc_nuclei.h5ad'))
adata.append(ad.read(f'/home/zhanyg/data/SpaDA/Liver/sc_exVivo.h5ad'))
adata.append(ad.read(f'/home/zhanyg/data/SpaDA/Liver/sc_inVivo.h5ad'))
idx = []
for sample in ['ABU11', 'ABU13', 'ABU17', 'ABU20']:
    idx.extend(np.nonzero(adata[0].obs.loc[:,'sample'].values==sample)[0])
adata[0] = adata[0][idx].copy()
cell_types = np.unique(adata[0].obs.label.values)
prop = {'nuclei':[], 'exVivo':[], 'inVivo':[]}
for i,scn in enumerate(prop.keys()):
    ctr_sc = []
    for ct in cell_types:
        idx = np.nonzero(adata[i].obs.label.values==ct)[0]
        ctr_sc.append(len(idx))
    ctr_sc /= np.sum(ctr_sc)
    prop[scn].extend(ctr_sc)
fig, ax = plt.subplots()
colors_all = itertools.cycle(plt.cm.tab20.colors)
colors = [next(colors_all) for _ in cell_types]
position = 0
for scn in ['exVivo', 'inVivo']:
    for value, color in zip(prop[scn], colors):
        width = value  # 每个部分的宽度正比于a中的值
        ax.barh(scn, width, left=position, color=color)
        position += width
    position = 0
for value, color, ct in zip(prop['nuclei'], colors, cell_types):
    width = value  # 每个部分的宽度正比于a中的值
    ax.barh('nuclei', width, left=position, color=color, label=ct)
    position += width
plt.xticks([])
plt.xlabel('proportion of cell types')
plt.legend(bbox_to_anchor=(1.01, 1))
plt.savefig(f'/home/zhanyg/data/SpaDA/Liver/draft/barh_sc.jpg', dpi=400, bbox_inches='tight')


#%% plot indc bar
scns = ['nuclei', 'exVivo','inVivo']
for scn in scns:
    indc = ['JSD', 'ER']
    adata_sc = ad.read(f'/home/zhanyg/data/SpaDA/Liver/sc_nuclei.h5ad')
    idx = []
    for sample in ['ABU11', 'ABU13', 'ABU17', 'ABU20']:
        idx.extend(np.nonzero(adata_sc.obs.loc[:,'sample'].values==sample)[0])
    adata_sc = adata_sc[idx].copy()
    cts = np.unique(adata_sc.obs.label.values)
    ctr_sc = []
    for ct in cts:
        idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
        ctr_sc.append(len(idx))
    ctr_sc /= np.sum(ctr_sc)
    adata_sts = []
    for i in range(3):
        adata_st = ad.read(f'/home/zhanyg/data/SpaDA/Liver/st_JBO00{i+1}.h5ad')
        adata_sts.append(adata_st)
    models = {'SpaDA':[], 'CellDART':[], 'SpaDecon':[], 'Cell2location':[], 'CARD':[]}
    colors = ['rosybrown', 'darkseagreen', 'burlywood', 'slategray', 'thistle']  # 'lightcoral'
    for m in models.keys():
        fdir = f'/home/zhanyg/Code/SpaDA/Liver/log2/{scn}/{m}/'
        jsds = []
        ers = []
        for f in range(5):
            pred_all = []
            for i in range(3):
                fpath = f'{fdir}t{f+1}/res/best/pred_JBO00{i+1}.csv'
                pred = pd.read_csv(fpath)
                pred = pred.loc[:,cts]
                pred_all.append(pred)
            pred = pd.concat(pred_all)
            ctr_st = np.sum(pred, 0) / len(pred)
            jsd = jensenshannon(ctr_st, ctr_sc)
            jsds.append(jsd)
            er = []
            for i in range(3):
                for ct in adata_sts[i].uns['cts_dom'].keys():
                    pred_ct = pred_all[i].loc[:,ct]
                    region = adata_sts[i].uns['cts_dom'][ct]
                    idx = np.nonzero(adata_sts[i].obs.region.values==region)[0]
                    er.append(np.sum(pred_ct[idx]) / np.sum(pred_ct))
            ers.extend(er)
        models[m].append(np.mean(jsds))
        models[m].append(np.mean(ers))
    models['LETSmix'] = models['SpaDA']
    del models['SpaDA']
    ms = ['LETSmix', 'CellDART', 'SpaDecon', 'Cell2location', 'CARD']
    x = np.arange(len(indc))
    bar_width = 0.15
    for i,m in enumerate(ms):
        plt.bar(x - (len(models)-1)/2*bar_width + i*bar_width, models[m], bar_width, label=m, color=[colors[i]]*len(indc))
    for i in range(len(indc)):
        for j,m in enumerate(ms):
            plt.text(x[i] - (len(models)-1)/2*bar_width + j*bar_width, models[m][i] + 0.005, f'{models[m][i]:.2f}', ha='center')
    plt.xticks(x, indc)
    plt.legend(loc='upper right', bbox_to_anchor=(1.01, 1))
    plt.title(f'{scn}')
    plt.savefig(f'/home/zhanyg/data/SpaDA/Liver/draft/indc/bars_{scn}.jpg', dpi=400, bbox_inches='tight')
    plt.close('all')


#%% bar Central vs Portal   all in one
regions = ['Central', 'Portal']
scns = ['nuclei','exVivo','inVivo']
models = ['SpaDA', 'CellDART', 'SpaDecon', 'Cell2location', 'CARD']
idxr = {}
for i in range(3):
    adata_st = ad.read(f'/home/zhanyg/data/SpaDA/Liver/st_JBO00{i+1}.h5ad')
    idxr[str(i)] = {}
    for region in regions:
        idx = np.nonzero(adata_st.obs.region.values==region)[0]
        idxr[str(i)][region] = idx
for region in regions:
    cts = { 'Central cells - nuclei':[], 'Portal cells - nuclei':[],
            'Central cells - exVivo':[], 'Portal cells - exVivo':[],
            'Central cells - inVivo':[], 'Portal cells - inVivo':[],}
    colors_all = itertools.cycle(plt.cm.tab20.colors)
    colors = [next(colors_all) for _ in cts.keys()]
    for m in models:
        pred_all = {'Central cells - nuclei':[], 'Portal cells - nuclei':[],
                    'Central cells - exVivo':[], 'Portal cells - exVivo':[],
                    'Central cells - inVivo':[], 'Portal cells - inVivo':[],}
        for scn in scns:
            fdir = f'/home/zhanyg/Code/SpaDA/Liver/log2/{scn}/{m}/'
            for f in range(5):
                for i in range(3):
                    fpath = f'{fdir}t{f+1}/res/best/pred_JBO00{i+1}.csv'
                    pred = pd.read_csv(fpath)
                    for ct in ['Central Vein Endothelial cells', 'Portal Vein Endothelial cells']:
                        pred_ct = pred.loc[:,ct].values
                        pred_ct = np.mean(pred_ct[idxr[str(i)][region]])
                        pred_all[f'{ct.split(" ")[0]} cells - {scn}'].append(pred_ct)
        for ct in cts.keys():
            cts[ct].append(np.mean(pred_all[ct]))
    x = np.arange(len(models))
    bar_width = 0.13
    fig = plt.figure()
    for i,ct in enumerate(cts.keys()):
        plt.bar(x - (len(cts)-1)/2*bar_width + i*bar_width, cts[ct], bar_width, label=ct, color=[colors[i]]*len(models))
    models2 = ['LETSmix', 'CellDART', 'SpaDecon', 'Cell2location', 'CARD']
    plt.xticks(x, models2)
    plt.legend(bbox_to_anchor=(1.42, 1))
    plt.title(f'{region} region')
    plt.savefig(f'/home/zhanyg/data/SpaDA/Liver/draft/indc/cvp2_{region}.jpg', dpi=400, bbox_inches='tight')
    plt.close('all')




