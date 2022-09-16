from jittor import nn
import jittor as jt 
from sklearn.neighbors import NearestNeighbors
import math 
import numpy as np


class LRScheduler:
    def __init__(self, optimizer, base_lr):
        self.optimizer = optimizer

        self.basic_lr = base_lr
        self.lr_decay = 0.6
        self.decay_step = 15000

    def step(self, step):
        lr_decay = self.lr_decay ** int(step / self.decay_step)
        lr_decay = max(lr_decay, 2e-5)
        self.optimizer.lr = lr_decay * self.basic_lr


def knn_indices_func_cpu(rep_pts,  # (N, pts, dim)
                         pts,      # (N, x, dim)
                         K : int,
                         D : int):
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    rep_pts = rep_pts.data
    pts = pts.data
    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(D*K + 1, algorithm = "ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:,1::D])

    region_idx = jt.array(np.stack(region_idx, axis = 0))
    return region_idx

def knn_indices_func_gpu(rep_pts,  # (N, pts, dim)
                         pts,      # (N, x, dim)
                         k : int, d : int ): # (N, pts, K)
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Very memory intensive, and thus unoptimal for large numbers of points.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    region_idx = []
    batch_size = rep_pts.shape[0]
    for idx in range (batch_size):
        qry = rep_pts[idx]
        ref = pts[idx]
        n, d = ref.shape
        m, d = qry.shape
        mref = ref.view(1, n, d).repeat(m, 1, 1)
        mqry = qry.view(m, 1, d).repeat(1, n, 1)
        
        dist2 = jt.sum((mqry - mref)**2, 2) # pytorch has squeeze 
        _, inds = topk(dist2, k*d + 1, dim = 1, largest = False)
        
        region_idx.append(inds[:,1::d])

    region_idx = jt.stack(region_idx, dim = 0) 

    return region_idx



def expand(x,shape):
    r'''
    Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
    Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front.
    Args:
       x-the input tensor.
       shape-the shape of expanded tensor.
    '''
    x_shape = x.shape
    x_l = len(x_shape)
    rest_shape=shape[:-x_l]
    expand_shape = shape[-x_l:]
    indexs=[]
    ii = len(rest_shape)
    for i,j in zip(expand_shape,x_shape):
        if i!=j:
            assert j==1
        indexs.append(f'i{ii}' if j>1 else f'0')
        ii+=1
    return x.reindex(shape,indexs)


def topk(input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim

    transpose_dims = [i for i in range(input.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    input = input.transpose(transpose_dims)
    index,values = jt.argsort(input,dim=0,descending=largest)
    indices = index[:k]
    values = values[:k]
    indices = indices.transpose(transpose_dims)
    values = values.transpose(transpose_dims)
    return [values,indices]
