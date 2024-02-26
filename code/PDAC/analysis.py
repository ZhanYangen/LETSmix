
import anndata as ad
import matplotlib.pyplot as plt 
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import seaborn as sns
import itertools
from scipy.spatial.distance import jensenshannon

#%% polt all region annotations on one image
stn = 'B'
adata = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
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
fig.savefig(f'/data112/zhanyg/data/SpaDA/PDAC/draft/region_st_{stn}.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)


#%% polt region annotations separately
stn = 'B'
adata = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
img = adata.uns.data['spatial']['histology']['images']['hires']
scale = adata.uns.data['spatial']['histology']['scalefactors']['tissue_hires_scalef']
layers = adata.obs.loc[:,'region'].values.astype(str)
layers_u = np.unique(layers)
for j in layers_u:
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    idx = np.nonzero(layers==j)[0]
    plt.scatter(adata.obsm['spatial'][idx,0]*scale, 
                adata.obsm['spatial'][idx,1]*scale, 
                15, 'r', alpha=1, label=j)
    plt.title(f'{j}')
    fig.savefig(f'/data112/zhanyg/data/SpaDA/PDAC/draft/region/{stn}_{j}.jpg', dpi=400, bbox_inches='tight')
    plt.close(fig)


#%% plot umap for scRNA-seq data
scn = 'Peng'
sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
adata_sc = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/sc_{scn}.h5ad')
tmp = adata_sc.copy()
sc.pp.normalize_total(tmp)
adata_sc = tmp
sc.tl.pca(adata_sc, svd_solver='arpack')  
sc.pp.neighbors(adata_sc, n_neighbors=10, n_pcs=40)  
sc.tl.umap(adata_sc)  
sc.pl.umap(adata_sc, color='label', show=False)
plt.title(f'PDAC-{scn}')
plt.savefig(f'/data112/zhanyg/data/SpaDA/PDAC/draft/sc_{scn}_umap.jpg', dpi=400, bbox_inches='tight')
plt.close('all')


#%% find marker gene and plot on the st sample
stn = 'B'
scn = 'B'

# find markers in scRNA-seq
os.makedirs(f'/data112/zhanyg/data/SpaDA/PDAC/draft/marker/{stn}{scn}', exist_ok=True)
scpath = f'/data112/zhanyg/data/SpaDA/PDAC/sc_{scn}.h5ad'
dfpath = f'/data112/zhanyg/data/SpaDA/PDAC/draft/gene_top_{scn}.csv'
adata_sc = ad.read(scpath)
if os.path.exists(dfpath):
    df_genelists = pd.read_csv(dfpath)
else:
    os.makedirs('/data112/zhanyg/data/SpaDA/PDAC/draft/', exist_ok=1)
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

# plot correlation matrix between markers and cell types
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
fig.savefig(f'/data112/zhanyg/data/SpaDA/PDAC/draft/marker/{stn}{scn}/heat.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)

# plot on the st sample
adata_st = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
sc.pp.log1p(adata_st)
for i,ct in enumerate(gene_top.columns):
    adata_st.obs.loc[:,'marker'] = adata_st[:,marker[i]].X.toarray()
    sc.pl.spatial(adata_st, img_key="hires", color='marker', cmap='jet',
        palette='Set1', size=1, legend_loc=None, title=f'{marker[i]}', show=False)
    plt.savefig(f'/data112/zhanyg/data/SpaDA/PDAC/draft/marker/{stn}{scn}/{ct}.jpg', dpi=400, bbox_inches='tight') 
    plt.close()


#%% plot pie pred map (contain estimated proportions of all cell types)
adata_st = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/st_B.h5ad')
pred = pd.read_csv(f'/home/zhanyg/Code/SpaDA/PDAC/log2/BB/SpaDA/t3/res/best/pred.csv')
cols = pred.columns.values
cols = np.sort(cols)
pred = pred.loc[:,cols]
coord = adata_st.obsm['spatial'] / 500
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
legend = ax.legend(handles=legend_labels, title='Cell Types', 
                   loc='upper left', prop={'size': 60}, bbox_to_anchor=(3, 6))
legend.get_title().set_fontsize(60)
fig.savefig(f'/home/zhanyg/Code/SpaDA/PDAC/log2/BB/SpaDA/t3/all.jpg', dpi=400, bbox_inches='tight')
plt.close(fig)


#%% plot pred proportion heatmaps
stn = 'A'
scn = 'A'
dir2 = 'SpaDecon'
dir1 = f'{stn}{scn}'
for i in range(5):
    pred = pd.read_csv(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{dir1}/{dir2}/t{i+1}/res/best/pred.csv')
    pred = pred.loc[:, ~pred.columns.str.contains('^Unnamed')]
    adata_st = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
    for ct in pred.columns.values:
        adata_st.obs.loc[:,'pred'] = pred.loc[:,ct].values
        sc.pl.spatial(adata_st, img_key="hires", color='pred', cmap='jet',
            palette='Set1', size=1, legend_loc=None, title=ct, show=False)
        plt.savefig(f'/home/zhanyg/Code/SpaDA/PDAC/log2/{dir1}/{dir2}/t{i+1}/res/best/{ct}.jpg', dpi=400, bbox_inches='tight') 
        plt.close()


#%% plot proportion barh (horizontal stacked bar plots)
name = 'B'
adata_sc = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/sc_{name}.h5ad')
ctr_sc = []
cell_types = np.unique(adata_sc.obs.label.values)
for ct in cell_types:
    idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
    ctr_sc.append(len(idx))
ctr_sc /= np.sum(ctr_sc)
models = ['CARD', 'Cell2location', 'SpaDecon', 'CellDART', 'SpaDA',]
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
plt.savefig(f'/data112/zhanyg/data/SpaDA/PDAC/draft/indc/{name}{name}_barh.jpg', dpi=400, bbox_inches='tight')


#%% bar polots of the metric values (JSD and ER)
scn = 'B'
stn = 'B'
indc = ['JSD', 'ER']  #, 'AUC'
adata_sc = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/sc_{scn}.h5ad')
adata_st = ad.read(f'/data112/zhanyg/data/SpaDA/PDAC/st_{stn}.h5ad')
models = {'SpaDA':[], 'CellDART':[], 'SpaDecon':[], 'Cell2location':[], 'CARD':[]}
colors = ['rosybrown', 'darkseagreen', 'burlywood', 'slategray', 'thistle']
ctr_sc = []
cts = np.unique(adata_sc.obs.label.values)
for ct in cts:
    idx = np.nonzero(adata_sc.obs.label.values==ct)[0]
    ctr_sc.append(len(idx))
ctr_sc /= np.sum(ctr_sc)
for m in models.keys():
    fdir = f'/home/zhanyg/Code/SpaDA/PDAC/log2/{scn}{stn}/{m}/'
    jsds = []
    ers = []
    for i in range(5):
        fpath = f'{fdir}t{i+1}/res/best/pred.csv'
        pred = pd.read_csv(fpath)
        pred = pred.loc[:,cts]
        ctr_st = np.sum(pred, 0) / len(pred)
        jsd = jensenshannon(ctr_st, ctr_sc)
        jsds.append(jsd)
        er = []
        for region in adata_st.uns['cts_dom'].keys():
            idx = np.nonzero(adata_st.obs.region.values==region)[0]
            for ct in adata_st.uns['cts_dom'][region]:
                pred_ct = pred.loc[:,ct]
                er.append(np.sum(pred_ct[idx]) / np.sum(pred_ct))
        ers.extend(er)
    models[m].append(np.mean(jsds))
    models[m].append(np.mean(ers))
x = np.arange(len(indc))
bar_width = 0.16
for i,m in enumerate(models.keys()):
    plt.bar(x - (len(models)-1)/2*bar_width + i*bar_width, models[m], bar_width, label=m, color=[colors[i]]*len(indc))
for i in range(len(indc)):
    for j,m in enumerate(models.keys()):
        plt.text(x[i] - (len(models)-1)/2*bar_width + j*bar_width, models[m][i] + 0.005, f'{models[m][i]:.2f}', ha='center')
plt.xticks(x, indc)
plt.legend(bbox_to_anchor=(1.01, 1))
plt.savefig(f'/data112/zhanyg/data/SpaDA/PDAC/draft/indc/{stn}{scn}_bars.jpg', dpi=400, bbox_inches='tight')

