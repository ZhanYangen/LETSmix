
import scanpy as sc
import pandas as pd
import numpy as np
from skimage import io 
import anndata as ad
import os
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def scale_adj(adj, l, sim):
    return np.exp(-1*adj/(2*(l**2))) * sim

def find_l(l, step, adj, si, sim):
    ll = np.zeros([len(adj),1])
    idx_done = []
    while(1):
        l += step
        adj_exp = scale_adj(adj, l, sim)
        tmp = np.sum(adj_exp, 1) - 1
        tmp = np.nonzero(tmp > si)[0]
        tmp = list(set(tmp) - set(idx_done))
        if len(tmp) > 0:
            ll[tmp] = l[tmp] - step
            idx_done.extend(tmp)
            if len(idx_done) == len(adj):
                break
    return ll

def calculate_adj_matrix(adata_st, args, img=None, r=None, mat=None):
    # spatial and histological distance
    pos = adata_st.obsm['spatial']
    pmean = np.zeros([len(pos),3])
    for i, (x,y) in enumerate(pos):
        spot = img[x-r:x+r, y-r:y+r]
        pmean[i] = np.mean(spot, (0,1))
    r,g,b = pmean.transpose()
    z = (r*np.var(r) + g*np.var(g) + b*np.var(b)) / (np.var(r) + np.var(g) + np.var(b))
    zs = (z-np.mean(z)) / np.std(z) * np.max([np.std(pos[:,0]), np.std(pos[:,1])])
    coord = np.vstack([pos[:,0], pos[:,1], zs]).transpose()
    adj = np.zeros([len(coord),len(coord)], np.float32)
    for i in range(len(coord)):
        adj[i,i+1:] = np.sqrt(np.sum((coord[i] - coord[i+1:])**2, 1))
    adj = adj + adj.transpose()

    # layer annotations mask
    block = np.ones_like(adj) * np.inf
    regions = list(set(adata_st.obs.region.values))
    for region in regions:
        idx = np.nonzero(adata_st.obs.region.values==region)[0]
        rows, cols = np.meshgrid(idx, idx)
        block[rows, cols] = 1
    adj *= block

    # expression similarity
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, 1)
    
    # approximate l_vec
    l = np.ones([len(adj),1])
    step = 0.1
    l = find_l(l, step, adj, args.si, sim)
    l = find_l(l, step/10, adj, args.si, sim)

    adj_exp = scale_adj(adj, l, sim)
    return adj_exp 

def log_minmaxscale(arr):  
    arrd = len(arr)
    arr = np.log1p(arr)
    return (arr-np.reshape(np.min(arr,axis=1), (arrd,1)))/np.reshape((np.max(arr, axis=1)-np.min(arr,axis=1)),(arrd,1))

def random_mix(Xs, ys, k, n_samples, seed=0):
    Xs_new, ys_new =[], []
    ys_ = np.zeros((len(ys), np.max(ys)+1))
    ys_[np.arange(len(ys)), ys] = 1
    rstate = np.random.RandomState(seed)
    fraction_all = rstate.rand(n_samples, k)  
    randindex_all = rstate.randint(len(Xs), size=(n_samples, k)) 
    for i in range(n_samples):
        # fraction: random fraction across the "k" number of sampled cells
        fraction = fraction_all[i]
        fraction = fraction/np.sum(fraction)
        fraction = np.reshape(fraction, (k, 1))
        # Random selection of the single cell data by the index
        randindex = randindex_all[i]
        ymix = ys_[randindex]
        # Calculate the fraction of cell types in the cell mixture
        yy = np.sum(ymix*fraction, axis=0)
        # Calculate weighted gene expression of the cell mixture
        XX = np.asarray(Xs[randindex])*fraction
        XX_ = np.sum(XX, axis=0)
        # Add cell type fraction & composite gene expression in the list
        ys_new.append(yy)
        Xs_new.append(XX_)
    Xs_new = np.asarray(Xs_new)
    ys_new = np.asarray(ys_new, np.float32)
    return Xs_new, ys_new

def load_data(args):
    # load scRNA-seq data
    adata_sc = ad.read(args.datadir+'sc.h5ad')
    if os.path.exists(args.datadir+'draft/gene_top.csv'):
        df_genelists = pd.read_csv(args.datadir+'draft/gene_top.csv')
    else:
        os.makedirs(args.datadir+'draft/', exist_ok=1)
        adata_sc2 = adata_sc.copy()
        sc.pp.normalize_total(adata_sc2, target_sum=1e4)
        sc.pp.log1p(adata_sc2)
        sc.tl.rank_genes_groups(adata_sc2, 'label', method='wilcoxon')
        genelists=adata_sc2.uns['rank_genes_groups']['names']
        df_genelists = pd.DataFrame.from_records(genelists)  
        df_genelists.to_csv(args.datadir+'draft/gene_top.csv', index=False)
    cts = list(set(adata_sc.obs.label))
    res_genes = []
    for column in cts: 
        res_genes.extend(df_genelists.head(args.m)[column].tolist())
    res_genes_ = list(set(res_genes))  
    inter_genes = [val for val in res_genes_ if val in adata_sc.uns['gene_st']]
    print('Selected Feature Gene number',len(inter_genes))
    adata_sc = adata_sc[:,inter_genes]
    tmp = adata_sc.copy()
    if args.spa:
        sc.pp.normalize_total(tmp, target_sum=1e4*(1+args.si))
    else:
        sc.pp.normalize_total(tmp, target_sum=1e4)
    adata_sc = tmp

    # generate pseudo-spots data
    mat_sc = adata_sc.X.toarray()
    lab_sc_sub = adata_sc.obs.loc[:,'label'].values
    sc_sub_dict = dict(zip(range(len(set(lab_sc_sub))), set(lab_sc_sub)))  #  num2str
    sc_sub_dict2 = dict((y,x) for x,y in sc_sub_dict.items())  # str2num
    lab_sc_num = [sc_sub_dict2[ii] for ii in lab_sc_sub]
    lab_sc_num = np.asarray(lab_sc_num, dtype=np.int32)
    mat_sc, lab_sc_num = random_mix(mat_sc, lab_sc_num, args.k, args.n_samples)
    mat_sc_s = log_minmaxscale(mat_sc).astype(np.float32)

    # load and refine ST data
    adata_sts = []
    mats_sp_s = []
    loader = {'t_val':[]}
    for st in args.st:
        adata_st = ad.read(args.datadir+st)
        adata_st = adata_st[:,inter_genes]
        tmp = adata_st.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4) 
        adata_st = tmp
        mat_sp = adata_st.X.toarray()
        if args.spa:
            img = io.imread(args.datadir+st.split('.')[0]+'_full_image.tif')
            r = int(adata_st.uns.data['spatial']['histology']['scalefactors']['spot_diameter_fullres']/2)
            adj = calculate_adj_matrix(adata_st, args, img, r, mat_sp)
            mat_sp = np.matmul(adj, mat_sp)
        mat_sp_s = log_minmaxscale(mat_sp).astype(np.float32)
        dataset = torch.from_numpy(mat_sp_s)
        loader['t_val'].append(DataLoader(dataset, args.bs, False, num_workers=8))
        adata_sts.append(adata_st)
        mats_sp_s.append(mat_sp_s)

    # prepare for training
    mat_sp_s = np.concatenate(mats_sp_s)
    idx = np.arange(len(lab_sc_num))
    np.random.shuffle(idx)
    idx_train = idx[:int(len(idx)*0.8)]
    idx_val = idx[int(len(idx)*0.8):]
    # loader for data from the source domain
    dataset = TensorDataset(torch.from_numpy(mat_sc_s[idx_val]), torch.from_numpy(lab_sc_num[idx_val]))
    loader['s_val'] = DataLoader(dataset, args.bs, False, num_workers=8)
    dataset = TensorDataset(torch.from_numpy(mat_sc_s[idx_train]), torch.from_numpy(lab_sc_num[idx_train]))
    loader['s_train'] = DataLoader(dataset, args.bs, True, num_workers=8, drop_last=True)
    # loader for data from both source and target domains
    def collate_fn(batch, data):
        xs, ys = zip(*batch)
        xs, ys = torch.stack(xs), torch.stack(ys)
        idx = np.random.choice(len(data), len(ys), False)
        xt = data[idx]
        yd = torch.cat([torch.zeros(len(ys)), torch.ones(len(ys))]).to(torch.int64)
        return xs, ys, xt, yd
    loader['train'] = DataLoader(dataset, args.bs, True, num_workers=8, drop_last=True,
                                collate_fn=lambda batch: collate_fn(batch, torch.from_numpy(mat_sp_s)))
    params = {'dim_in':mat_sc_s.shape[1], 'dim_out':len(sc_sub_dict2)}
    return loader, adata_sts, params, sc_sub_dict2


