
import anndata as ad
import matplotlib.pyplot as plt 
import scanpy as sc
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import shutil
from scipy.stats import pearsonr
import itertools
from scipy.spatial.distance import jensenshannon

#%% polt region labels
stn = 'B'
adata = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
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
                15, alpha=1, label=j)
lgnd = plt.legend(bbox_to_anchor=(1.01, 1))
for handle in lgnd.legendHandles:
    handle.set_sizes([20.0])
plt.title(f'PDAC-{stn}')
fig.savefig(f'/home/zhanyg/data/SpaDA/PDAC/draft/region_st_{stn}.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)


#%% plot region label   split
stn = 'A'
adata = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
img = adata.uns.data['spatial']['histology']['images']['hires']
scale = adata.uns.data['spatial']['histology']['scalefactors']['tissue_hires_scalef']
layers = adata.obs.loc[:,'region'].values.astype(str)
layers_u = ['Cancer & Duct']#np.unique(layers)
for j in layers_u:
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    idx = []
    for region in ['Cancer', 'Duct epithelium']:
        idx.extend(np.nonzero(layers==region)[0])
    # idx = np.nonzero(layers==j)[0]
    plt.scatter(adata.obsm['spatial'][idx,0]*scale, 
                adata.obsm['spatial'][idx,1]*scale, 
                15, 'r', alpha=1, label=j)
    plt.title(f'{j}')
    fig.savefig(f'/home/zhanyg/data/SpaDA/PDAC/draft/region/{stn}_{j}.jpg', dpi=400, bbox_inches='tight')
    plt.close(fig)


#%% plot sc umap
scn = 'Peng'
sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
adata_sc = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/sc_{scn}.h5ad')
tmp = adata_sc.copy()
sc.pp.normalize_total(tmp)
adata_sc = tmp
sc.tl.pca(adata_sc, svd_solver='arpack')  
sc.pp.neighbors(adata_sc, n_neighbors=10, n_pcs=40)  
sc.tl.umap(adata_sc)  
sc.pl.umap(adata_sc, color='label', show=False)
plt.title(f'PDAC-{scn}')
plt.savefig(f'/home/zhanyg/data/SpaDA/PDAC/draft/sc_{scn}_umap.jpg', dpi=400, bbox_inches='tight')
# sc.tl.rank_genes_groups(adata_sc, 'label', method='wilcoxon')
# sc.pl.rank_genes_groups(adata_sc, n_genes=20, sharey=False, show=False)
# plt.savefig('/home/zhanyg/data/SpaDA/DLPFC/draft/sc_gene.jpg', dpi=400, bbox_inches='tight')
plt.close('all')


#%% find marker gene
stn = 'B'
scn = 'B'
os.makedirs(f'/home/zhanyg/data/SpaDA/PDAC/draft/marker/{stn}{scn}', exist_ok=True)
scpath = f'/home/zhanyg/data/SpaDA/PDAC/sc_{scn}.h5ad'
dfpath = f'/home/zhanyg/data/SpaDA/PDAC/draft/gene_top_{scn}.csv'
adata_sc = ad.read(scpath)
if os.path.exists(dfpath):
    df_genelists = pd.read_csv(dfpath)
else:
    os.makedirs('/home/zhanyg/data/SpaDA/PDAC/draft/', exist_ok=1)
    adata_sc2 = adata_sc.copy()
    sc.pp.normalize_total(adata_sc2, target_sum=1e4)
    sc.pp.log1p(adata_sc2)
    sc.tl.rank_genes_groups(adata_sc2, 'label', method='wilcoxon')
    genelists=adata_sc2.uns['rank_genes_groups']['names']
    df_genelists = pd.DataFrame.from_records(genelists)  
    df_genelists.to_csv(dfpath, index=False)
gene_top = df_genelists
if stn == 'A':
    gene_top.Cancer_clone_B.values[0] = 'S100A4'
if 'Unnamed: 0' in gene_top.columns.values:
    gene_top.drop(columns='Unnamed: 0', inplace=True)
marker = gene_top.values[0]
# sc.pp.normalize_total(adata_sc, target_sum=1e4)
# sc.pp.log1p(adata_sc)

cor_mat = np.zeros([len(marker), len(marker)])
for i,g in enumerate(marker):
    gene = adata_sc[:,g].X.toarray().astype(np.float32).squeeze()
    for j in range(len(gene_top.columns)):
        ct = gene_top.columns[j]
        label = np.zeros(len(gene))
        idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
        label[idx] = 1
        cor, _ = pearsonr(gene, label)
        cor_mat[i,j] = cor
cors = np.diag(cor_mat)
fig = plt.figure()
sns.heatmap(cor_mat, cmap='RdYlBu_r')
plt.yticks(np.arange(len(gene_top.columns)) + 0.5, gene_top.columns, rotation='horizontal')
plt.xticks(np.arange(len(marker)) + 0.5, marker, rotation='vertical')
plt.title('Correlation Matrix')
plt.ylabel('Cell Types')
plt.xlabel('Marker Genes')
fig.savefig(f'/home/zhanyg/data/SpaDA/PDAC/draft/marker/{stn}{scn}/heat.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)

adata_st = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
# sc.pp.normalize_total(adata_st, target_sum=1e4)
sc.pp.log1p(adata_st)
for i,ct in enumerate(gene_top.columns):
    adata_st.obs.loc[:,'marker'] = adata_st[:,marker[i]].X.toarray()
    sc.pl.spatial(adata_st, img_key="hires", color='marker', cmap='jet',
        palette='Set1', size=1, legend_loc=None, title=f'{marker[i]}', show=False)
    plt.savefig(f'/home/zhanyg/data/SpaDA/PDAC/draft/marker/{stn}{scn}/{ct}.jpg', dpi=400, bbox_inches='tight') 
    plt.close()


#%% plot pie pred map
stn = 'A'
scn = 'Peng'
tes = 2
adata_st = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
pred = pd.read_csv(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{stn}{scn}/LETSmix/t{tes}/res/best/pred.csv')
cols = pred.columns.values
cols = np.sort(cols)
pred = pred.loc[:,cols]
coord = adata_st.obsm['spatial'] / 900   # 900 for A, 500 for B
coord[:,1] = -coord[:,1]
colors = itertools.cycle(plt.cm.tab20.colors)
current_colors = [next(colors) for _ in pred.columns.values]
fig, ax = plt.subplots()
for i in range(len(pred)):
    ax.pie(pred.values[i], startangle=90, counterclock=False, colors=current_colors,
           wedgeprops={'edgecolor': 'w'}, center=coord[i], radius=0.3, normalize=True)
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markersize=60,
                            markerfacecolor=current_colors[i], 
                            label=pred.columns.values[i]) for i in range(pred.shape[1])]
legend = ax.legend(handles=legend_labels, title='Cell Types', prop={'size': 60}, 
                   loc='right', bbox_to_anchor=(14,-3))   # (14,-3) for A, (3,6) for B
legend.get_title().set_fontsize(60)
fig.savefig(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{stn}{scn}/LETSmix/t{tes}/all.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)


#%% plot pred map
stn = 'A'
scn = 'A'
dir2 = 'SpaDecon'
dir1 = f'{stn}{scn}'
for i in range(5):
    pred = pd.read_csv(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{dir1}/{dir2}/t{i+1}/res/best/pred.csv')
    pred = pred.loc[:, ~pred.columns.str.contains('^Unnamed')]
    adata_st = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
    for ct in pred.columns.values:
        adata_st.obs.loc[:,'pred'] = pred.loc[:,ct].values
        sc.pl.spatial(adata_st, img_key="hires", color='pred', cmap='jet',
            palette='Set1', size=1, legend_loc=None, title=ct, show=False)
        plt.savefig(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{dir1}/{dir2}/t{i+1}/res/best/{ct}.jpg', dpi=400, bbox_inches='tight') 
        plt.close()


#%% plot proportion barh
name = 'B'
adata_sc = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/sc_{name}.h5ad')
ctr_sc = []
cell_types = np.unique(adata_sc.obs.label.values)
for ct in cell_types:
    idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
    ctr_sc.append(len(idx))
ctr_sc /= np.sum(ctr_sc)
models = ['CARD', 'Cell2location', 'SpaDecon', 'CellDART', 'LETSmix',]
prop = {}
for model in models:
    prop[model] = []
    ctr_st = np.zeros([5,len(ctr_sc)])
    for i in range(5):
        pred = pd.read_csv(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{name}{name}/{model}/t{i+1}/res/best/pred.csv')
        pred = pred.loc[:,cell_types]
        ctr_st[i] = np.sum(pred, 0) / len(pred)
    ctr_st = np.sum(ctr_st, 0) / len(ctr_st)
    prop[model].extend(ctr_st)

jsds = []
for m in prop.keys():
    jsd = jensenshannon(prop[m], ctr_sc)
    jsds.append(jsd)

fig, ax = plt.subplots()
colors_all = itertools.cycle(plt.cm.tab20.colors)
colors = [next(colors_all) for _ in cell_types]
position = 0
for model in models:
    for value, color in zip(prop[model], colors):
        width = value  
        ax.barh(model, width, left=position, color=color)
        position += width
    position = 0
for value, color, ct in zip(ctr_sc, colors, cell_types):
    width = value  
    ax.barh('scRNA-seq', width, left=position, color=color, label=ct)
    position += width
plt.xticks([])
plt.xlabel('proportion of cell types')
plt.legend(bbox_to_anchor=(1.01, 1.1))
plt.title(f'PDAC-{name}')
plt.savefig(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{name}{name}/barh.jpg', dpi=400, bbox_inches='tight')


#%% plot indc bar
stn = 'A'
scn = 'Peng'
adata_st = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
adata_sc = ad.read(f'/home/zhanyg/data/SpaDA/PDAC/sc_{scn}.h5ad')
models = {'LETSmix':[], 'CellDART':[], 'SpaDecon':[], 'Cell2location':[], 'CARD':[]}
colors = ['rosybrown', 'darkseagreen', 'burlywood', 'slategray', 'thistle']
ctr_sc = []
cts = np.unique(adata_sc.obs.label.values)
for ct in cts:
    idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
    ctr_sc.append(len(idx))
ctr_sc /= np.sum(ctr_sc)
for m in models.keys():
    fdir = f'/home/zhanyg/Code/SpaDA/PDAC/log2/{stn}{scn}/{m}/'
    jsds = []
    ers = []
    # aucs = []
    for i in range(5):
        fpath = f'{fdir}t{i+1}/res/best/pred.csv'
        pred = pd.read_csv(fpath)
        pred = pred.loc[:,cts]
        ctr_st = np.sum(pred, 0) / len(pred)
        jsd = jensenshannon(ctr_st, ctr_sc)
        jsds.append(jsd)
        er = []
        # au = []
        if scn != 'Peng':
            for region in adata_st.uns[f'cts_dom_{scn[0]}'].keys():
                idx = np.nonzero(adata_st.obs.region.values==region)[0]
                # target = np.zeros(len(pred))
                # target[idx] = 1
                for ct in adata_st.uns[f'cts_dom_{scn[0]}'][region]:
                    pred_ct = pred.loc[:,ct]
                    er.append(np.sum(pred_ct[idx]) / np.sum(pred_ct))
                    # fpr, tpr, _ = roc_curve(target, pred_ct)
                    # roc_auc = auc(fpr, tpr)
                    # au.append(roc_auc)
        else:
            for ct in adata_st.uns[f'cts_dom_{scn[0]}'].keys():
                idx = []
                for region in adata_st.uns[f'cts_dom_{scn[0]}'][ct]:
                    idx.extend(np.nonzero(adata_st.obs.region.values==region)[0])
                pred_ct = pred.loc[:,ct]
                er.append(np.sum(pred_ct[idx]) / np.sum(pred_ct))
        ers.extend(er)
        # aucs.extend(au)
    models[m].append(np.mean(jsds))
    models[m].append(np.mean(ers))
    # models[m].append(np.mean(aucs))
if stn == scn:
    indc = ['JSD', 'ER']
    x = np.arange(len(indc))
    bar_width = 0.16   
    for i,m in enumerate(models.keys()):
        plt.bar(x - (len(models)-1)/2*bar_width + i*bar_width, models[m], bar_width, label=m, color=[colors[i]]*len(indc))
    for i in range(len(indc)):
        for j,m in enumerate(models.keys()):
            plt.text(x[i] - (len(models)-1)/2*bar_width + j*bar_width, models[m][i] + 0.005, f'{models[m][i]:.2f}', ha='center')
    plt.xticks(x, indc)
    plt.legend(bbox_to_anchor=(1.01, 1))
else:
    x = np.arange(len(models))+0.1
    bar_width = 0.8  
    value = [i[1] for i in models.values()]
    plt.bar(x, value, bar_width, color=colors)
    for i,m in enumerate(models.keys()):
        plt.text(x[i], value[i]+0.005, f'{value[i]:.2f}', ha='center')
    plt.xticks(x, models.keys())
    plt.ylabel('ER')
plt.savefig(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{stn}{scn}/bars.jpg', dpi=400, bbox_inches='tight')


#%% boxplot hyper r
adata_st = ad.read('/home/zhanyg/data/SpaDA/PDAC/st_A.h5ad')
adata_sc = ad.read('/home/zhanyg/data/SpaDA/PDAC/sc_A.h5ad')
ctr_sc = []
cts = np.unique(adata_sc.obs.label.values)
for ct in cts:
    idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
    ctr_sc.append(len(idx))
ctr_sc /= np.sum(ctr_sc)
ers = {'LETSmix':[], '0.6':[], '0.2':[], '0.1':[], '0.05':[], 'LETSmix_base':[]}
jsds = {'LETSmix':[], '0.6':[], '0.2':[], '0.1':[], '0.05':[], 'LETSmix_base':[]}
logdir = ['/home/zhanyg/Code/SpaDA/PDAC/log2/AA/LETSmix',
          '/home/zhanyg/Code/SpaDA/PDAC/log2/AA/rt/r06/tes1',
          '/home/zhanyg/Code/SpaDA/PDAC/log2/AA/rt/r02/tes2',
          '/home/zhanyg/Code/SpaDA/PDAC/log2/AA/rt/r01/tes2',
          '/home/zhanyg/Code/SpaDA/PDAC/log2/AA/rt/r005/tes1',
          '/home/zhanyg/Code/SpaDA/PDAC/log2/AA/LETSmix_base']
for i in range(len(logdir)):
    m = list(ers.keys())[i]
    for j in range(5):
        fpath = f'{logdir[i]}/t{j+1}/res/best/pred.csv'
        pred = pd.read_csv(fpath)
        pred = pred.loc[:,cts]
        ctr_st = np.sum(pred, 0) / len(pred)
        jsds[m].append(jensenshannon(ctr_st, ctr_sc))
        er = []
        for region in adata_st.uns[f'cts_dom_A'].keys():
            idx = np.nonzero(adata_st.obs.region.values==region)[0]
            for ct in adata_st.uns[f'cts_dom_A'][region]:
                pred_ct = pred.loc[:,ct]
                er.append(np.sum(pred_ct[idx]) / np.sum(pred_ct))
        ers[m].append(np.mean(er))
fig = plt.figure()
ax = sns.boxplot(data=list(ers.values()))
ax.set_xticklabels(ers.keys())
ax.set_ylabel('ER')
ax.set_xlabel('proportion of the used real-spots')
fig.savefig('/home/zhanyg/Code/SpaDA/PDAC/log2/AA/r_ER.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)
fig = plt.figure()
ax = sns.boxplot(data=list(jsds.values()))
ax.set_xticklabels(ers.keys())
ax.set_ylabel('JSD')
ax.set_xlabel('proportion of the used real-spots')
fig.savefig('/home/zhanyg/Code/SpaDA/PDAC/log2/AA/r_JSD.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)


