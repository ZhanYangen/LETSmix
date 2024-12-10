
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
import gc
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def preprocess(
        adata_sc,
        adata_st,
        n_top_genes=100,
        smooth=0.5,
        tmp_dir=None,
):
    '''\

    Parameters
    ----------
    adata : anndata
        AnnData object of scRNA-seq data.
    gene_st : list
        List of gene names in spatial transcriptomics data.
    n_top_genes : int
        Number of top genes to select for scRNA-seq data.
    smooth : float
        Smoothing parameter in the LETS filter for spatial transcriptomics data.
    tmp_dir : str
        Directory to store temporary files.        

    Returns
    -------
    scRNA-seq and spatial transcriptomics data with selected genes and normalized expression values.

    '''
    if os.path.exists(tmp_dir+'/gene_top.csv'):
        df_genelists = pd.read_csv(tmp_dir+'/gene_top.csv')
    else:
        tmp = adata_sc.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.tl.rank_genes_groups(tmp, 'label', method='wilcoxon')
        genelists=tmp.uns['rank_genes_groups']['names']
        df_genelists = pd.DataFrame.from_records(genelists)  
        if tmp_dir is not None:
            os.makedirs(tmp_dir, exist_ok=1)
            df_genelists.to_csv(tmp_dir+'/gene_top.csv', index=False)
        del genelists
        gc.collect()
    cts = np.unique(adata_sc.obs.label.values)
    cts.sort()
    res_genes = []
    for column in cts: 
        res_genes.extend(df_genelists.head(n_top_genes)[column].tolist())
    res_genes = list(set(res_genes))
    res_genes = [val for val in res_genes if val in adata_st.var_names]
    print('Selected Feature Gene number', len(res_genes))

    adata_sc = adata_sc[:,res_genes]
    sc.pp.normalize_total(adata_sc, target_sum=1e4*(1+smooth))
    adata_st = adata_st[:,res_genes]
    sc.pp.normalize_total(adata_st, target_sum=1e4)

    mat_sc = adata_sc.X.toarray().astype(np.float32)
    lab_sc_sub = adata_sc.obs.loc[:,'label'].values
    sc_sub_dict = dict(zip(range(len(cts)), cts))  #  num2str
    sc_sub_dict = dict((y,x) for x,y in sc_sub_dict.items())  # str2num
    lab_sc_num = [sc_sub_dict[ii] for ii in lab_sc_sub]
    lab_sc_num = np.asarray(lab_sc_num, dtype=np.int32)
    lab_sc_num = np.eye(len(cts))[lab_sc_num].astype(np.int32)  # one-hot encoding
    return mat_sc, lab_sc_num, adata_st, sc_sub_dict

def scale_adj(adj, l, sim):
    return torch.exp(-1*adj/(2*(l**2))) * sim

def find_l(l, step, adj, si, sim):
    ll = torch.zeros_like(l)
    idx_done = set()
    while(1):
        l += step
        tmp = scale_adj(adj, l, sim)
        tmp = torch.sum(tmp, 1) - 1
        tmp = np.nonzero(tmp.cpu().numpy() > si)[0]
        tmp = list(set(tmp) - idx_done)
        if len(tmp) > 0:
            ll[tmp] = l[tmp] - step
            idx_done.update(tmp)
            if len(idx_done) == len(adj):
                break
    return ll

def LETS_filter(
        adata,
        smooth=0.5,
        img=None,
        regions=None,
        device='cpu',
        tmp_dir=None,
):
    '''\

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial transcriptomics data.
    smooth : float
        Smoothing parameter in the LETS filter for spatial transcriptomics data.
    img : numpy.ndarray
        Image data for spatial transcriptomics data.
    regions : str
        region annotations for spatial transcriptomics data.
    device : str
        Device to run the code.
    tmp_dir : str
        Directory to store temporary files.        

    Returns
    -------
    Spatial transcriptomics data with spatially refined expression values.

    '''
    # image and spot coordinates
    r = int(adata.uns.data['spatial']['histology']['scalefactors']['spot_diameter_fullres']/2)
    pos = np.round(adata.obsm['spatial']).astype(int)
    if img is None:
        coord = np.vstack([pos[:,0], pos[:,1]]).transpose().astype(np.float32)
        adj = np.linalg.norm(coord[:, np.newaxis, :] - coord[np.newaxis, :, :], axis=2)
    else:
        pmean = np.zeros([len(pos),3])
        for i, (x,y) in enumerate(pos):
            spot = img[y-r:y+r, x-r:x+r]
            pmean[i] = np.mean(spot, (0,1))
        del img
        gc.collect()
        r,g,b = pmean.transpose()
        z = (r*np.var(r) + g*np.var(g) + b*np.var(b)) / (np.var(r) + np.var(g) + np.var(b))
        z = (z-np.mean(z)) / np.std(z) * np.max([np.std(pos[:,0]), np.std(pos[:,1])])
        coord = np.vstack([pos[:,0], pos[:,1], z]).transpose().astype(np.float32)
        adj = np.linalg.norm(coord[:, np.newaxis, :] - coord[np.newaxis, :, :], axis=2)
    
    # layer annotations mask
    if regions is not None:
        region_mask = np.ones_like(adj) * np.inf
        for region in set(regions):
            idx = np.where(regions == region)[0]
            region_mask[np.ix_(idx, idx)] = 1
        adj *= region_mask
        del region_mask
        gc.collect()

    # expression similarity
    mat = adata.X.toarray().astype(np.float32)
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, 1)

    # approximate l_vec
    torch.cuda.set_device(device)
    adj, sim, mat = map(lambda x: torch.from_numpy(x).to(device), (adj, sim, mat))
    with torch.no_grad():
        if tmp_dir is not None:
            lpath = tmp_dir+'l_vec.npy'
            if os.path.exists(lpath):
                l = np.load(lpath)
                l = torch.from_numpy(l).to(device)
            else:
                l = torch.ones([len(adj),1]).to(device)
                step = 0.1
                l = find_l(l, step, adj, smooth, sim)
                l = find_l(l, step/10, adj, smooth, sim)
                np.save(lpath, l.cpu().numpy())
        else:
            l = torch.ones([len(adj),1]).to(device)
            step = 0.1
            l = find_l(l, step, adj, smooth, sim)
            l = find_l(l, step/10, adj, smooth, sim)
        adj = scale_adj(adj, l, sim)
        mat = torch.matmul(adj, mat)
    mat = mat.cpu().to(torch.float32)
    del adj, sim
    gc.collect()
    torch.cuda.empty_cache()
    return mat.numpy()

def log_minmaxscale(arr):
    arr = torch.log1p(arr)
    arr_min = torch.min(arr, dim=1, keepdim=True)[0]
    arr_max = torch.max(arr, dim=1, keepdim=True)[0]
    return (arr - arr_min) / (arr_max - arr_min)

def mix_spot(data, n, beta):
    idx = np.random.choice(len(data), [n*2,2])
    tmp = idx[:,0] - idx[:,1]
    tmp = np.nonzero(tmp)[0]
    idx = idx[tmp[:n]]
    x = {'s1':data[idx[:,0]], 's2':data[idx[:,1]]}
    x['r'] = torch.from_numpy(np.random.beta(beta, beta, [n,1])).to(torch.float32)
    x['smix'] = x['r'] * x['s1'] + (1-x['r']) * x['s2']
    return x

def collate(batch, bs, k, data=None):
    x, y = zip(*batch)
    x, y = torch.stack(x), torch.stack(y)
    x, y = x.reshape(bs, k, -1), y.reshape(bs, k, -1)
    w = torch.rand(bs, k, dtype=torch.float32)
    w = w / w.sum(dim=1, keepdim=True)
    w = w.unsqueeze(-1)
    x, y = torch.sum(x * w, dim=1), torch.sum(y * w, dim=1)
    x = log_minmaxscale(x)
    if data is not None:
        xt = mix_spot(data, bs, 2)
        xt = log_minmaxscale(xt['smix'])
        return x, y, xt
    return x, y

def load_data(
        mat_sc,
        lab_sc,
        mat_st,
        bs,
        k=8,
        num_aug=10,
        num_workers=8,
):
    '''\

    Parameters
    ----------
    mat_sc : numpy.ndarray
        scRNA-seq data with selected genes and normalized expression values.
    lab_sc : numpy.ndarray
        scRNA-seq data with one-hot encoding labels.
    mat_st : numpy.ndarray
        Spatial transcriptomics data with spatially refined expression values.
    bs : int
        Batch size.
    k : int
        Number of cells in each pseudo-spot.
    num_aug : int
        Number of augmentations for the dataset. Enabling larger batch size and faster training.
        Note that bs\*k shoule be less than num_aug\*min(len(mat_st), len(mat_sc)).
    num_workers : int
        Number of workers for data loading.        

    Returns
    -------
    Data loaders for training and evaluation. Pseudo-spots and mixed real-spots will be automatically generated when sampling batches \
    from the data loaders, except for the evaluation data loader['t_eval'].

    '''
    mat_sc, lab_sc, mat_st = map(lambda x: torch.from_numpy(x), (mat_sc, lab_sc, mat_st))
    loader = {}
    dataset = TensorDataset(mat_sc.repeat(num_aug, 1), lab_sc.repeat(num_aug, 1))
    loader['s_train'] = DataLoader(dataset, bs*k, True, num_workers=num_workers, pin_memory=True, drop_last=True,
                                   collate_fn=lambda batch: collate(batch, bs, k))
    loader['t_train'] = DataLoader(dataset, bs*k, True, num_workers=num_workers, pin_memory=True, drop_last=True,
                                   collate_fn=lambda batch: collate(batch, bs, k, mat_st.repeat(num_aug, 1)))
    bs2 = bs if bs < len(mat_st) else len(mat_st)
    loader['t_eval'] = DataLoader(log_minmaxscale(mat_st), bs2, False, num_workers=num_workers, pin_memory=True)
    return loader

