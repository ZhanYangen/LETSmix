
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

def calculate_adj_matrix(adata_st, args, img=None, r=None):
    # spatial and histological distance
    pos = adata_st.obsm['spatial']
    pos = np.round(pos).astype(int)
    pmean = np.zeros([len(pos),3])
    for i, (x,y) in enumerate(pos):
        spot = img[y-r:y+r, x-r:x+r]
        pmean[i] = np.mean(spot, (0,1))
    r,g,b = pmean.transpose()
    z = (r*np.var(r) + g*np.var(g) + b*np.var(b)) / (np.var(r) + np.var(g) + np.var(b))
    z = (z-np.mean(z)) / np.std(z) * np.max([np.std(pos[:,0]), np.std(pos[:,1])])
    coord = np.vstack([pos[:,0], pos[:,1], z]).transpose().astype(np.float32)
    adj = np.linalg.norm(coord[:, np.newaxis, :] - coord[np.newaxis, :, :], axis=2)

    # layer annotations mask
    region_mask = np.ones_like(adj) * np.inf
    for region in set(adata_st.obs['region']):
        idx = np.where(adata_st.obs['region'] == region)[0]
        region_mask[np.ix_(idx, idx)] = 1
    adj *= region_mask
    del region_mask
    gc.collect()

    # expression similarity
    mat = adata_st.X.toarray()
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, 1)
    
    # approximate l_vec
    adj, sim, mat = map(lambda x: torch.from_numpy(x).to(args.device), (adj, sim, mat))
    lpath = args.datadir+'draft/l_vec.npy'
    with torch.no_grad():
        if os.path.exists(lpath):
            l = np.load(lpath)
            l = torch.from_numpy(l).to(args.device)
        else:
            l = torch.ones([len(adj),1]).to(args.device)
            step = 0.1
            l = find_l(l, step, adj, args.si, sim)
            l = find_l(l, step/10, adj, args.si, sim)
            np.save(lpath, l.cpu().numpy())
        adj = scale_adj(adj, l, sim)
        mat = torch.matmul(adj, mat)
    mat = mat.cpu().to(torch.float32)
    del adj, sim
    gc.collect()
    torch.cuda.empty_cache()
    return mat

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

def collate(batch, args, data=None):
    x, y = zip(*batch)
    x, y = torch.stack(x), torch.stack(y)
    x, y = x.reshape(args.bs, args.k, -1), y.reshape(args.bs, args.k, -1)
    w = torch.rand(args.bs, args.k, dtype=torch.float32)
    w = w / w.sum(dim=1, keepdim=True)
    w = w.unsqueeze(-1)
    x, y = torch.sum(x * w, dim=1), torch.sum(y * w, dim=1)
    x = log_minmaxscale(x)
    if data is not None:
        xt = mix_spot(data, args.bs, 2)
        xt = log_minmaxscale(xt['smix'])
        return x, y, xt
    return x, y

def load_data(args):
    # Gene selection
    torch.cuda.set_device(args.device)
    adata_sc = ad.read(args.datadir+f'sc_{args.sc}.h5ad')
    adata_st = ad.read(args.datadir+f'st_{args.st}.h5ad')
    if os.path.exists(args.datadir+f'draft/gene_top_{args.sc}.csv'):
        df_genelists = pd.read_csv(args.datadir+f'draft/gene_top_{args.sc}.csv')
    else:
        os.makedirs(args.datadir+'draft/', exist_ok=1)
        tmp = adata_sc.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.tl.rank_genes_groups(tmp, 'label', method='wilcoxon')
        genelists=tmp.uns['rank_genes_groups']['names']
        df_genelists = pd.DataFrame.from_records(genelists)  
        df_genelists.to_csv(args.datadir+f'draft/gene_top_{args.sc}.csv', index=False)
        del genelists
        gc.collect()
    cts = np.unique(adata_sc.obs.label.values)
    res_genes = []
    for column in cts: 
        res_genes.extend(df_genelists.head(args.m)[column].tolist())
    res_genes = list(set(res_genes))
    res_genes = [val for val in res_genes if val in adata_st.var_names]
    print('Selected Feature Gene number',len(res_genes))

    adata_sc = adata_sc[:,res_genes]
    sc.pp.normalize_total(adata_sc, target_sum=1e4*(1+args.si))
    mat_sc = adata_sc.X.toarray()
    lab_sc_sub = adata_sc.obs.loc[:,'label'].values
    sc_sub_dict = dict(zip(range(len(cts)), cts))  #  num2str
    sc_sub_dict = dict((y,x) for x,y in sc_sub_dict.items())  # str2num
    lab_sc_num = [sc_sub_dict[ii] for ii in lab_sc_sub]
    lab_sc_num = np.asarray(lab_sc_num, dtype=np.int32)
    lab_sc_num = np.eye(len(cts))[lab_sc_num].astype(np.int32)  # one-hot encoding
    mat_sc, lab_sc_num = torch.from_numpy(mat_sc), torch.from_numpy(lab_sc_num)

    # calculate the general cell type compositioin label for the JSD computation
    ctr_sc = []
    for ct in cts:
        idx = np.nonzero(lab_sc_sub==ct)[0]
        ctr_sc.append(len(idx))
    ctr_sc /= np.sum(ctr_sc)
    adata_st.uns['ctr_sc'] = ctr_sc

    # load and refine ST data
    adata_st = adata_st[:,res_genes]
    sc.pp.normalize_total(adata_st, target_sum=1e4) 
    img = io.imread(args.datadir+f'st_{args.st}_full_image.tif')
    r = int(adata_st.uns.data['spatial']['histology']['scalefactors']['spot_diameter_fullres']/2)
    mat_sp = calculate_adj_matrix(adata_st, args, img, r)
    del img
    gc.collect()

    # prepare for training
    loader = {}
    num_aug = 40 # dataset augmentation if needed
    dataset = TensorDataset(mat_sc.repeat(num_aug, 1), lab_sc_num.repeat(num_aug, 1))
    mat_sp_aug = mat_sp.repeat(num_aug, 1)
    loader['s_train'] = DataLoader(dataset, args.bs*args.k, True, num_workers=8, pin_memory=True, drop_last=True,
                                   collate_fn=lambda batch: collate(batch, args))
    loader['t_train'] = DataLoader(dataset, args.bs*args.k, True, num_workers=8, pin_memory=True, drop_last=True,
                                   collate_fn=lambda batch: collate(batch, args, mat_sp_aug))
    bs = args.bs if args.bs < len(mat_sp) else len(mat_sp)
    loader['t_val'] = DataLoader(log_minmaxscale(mat_sp), bs, False, num_workers=8, pin_memory=True)
    params = {'dim_in':mat_sc.shape[1], 'dim_out':len(cts)}
    return loader, adata_st, params, sc_sub_dict


