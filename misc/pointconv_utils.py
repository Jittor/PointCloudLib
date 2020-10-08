
"""
Utility function for PointConv
Originally from : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/utils.py
Modify by Wenxuan Wu
Date: September 2019
"""

from time import time
import numpy as np
import random
#from sklearn.neighbors.kde import KernelDensity

import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

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
    return values,indices

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * jt.matmul(src, dst.permute(0, 2, 1))
    dist += jt.sum(src ** 2, -1).view(B, N, 1)
    dist += jt.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    #device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.arange(B, dtype='l')
    batch_indices = jt.array(batch_indices).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    #device = xyz.device
    B, N, C = xyz.shape
    centroids = jt.zeros((B, npoint))
    distance = jt.ones((B, N)) * 1e10
    
    farthest = np.random.randint(0, N, B, dtype='l')
    batch_indices = np.arange(B, dtype='l')
    farthest = jt.array(farthest)
    batch_indices = jt.array(batch_indices) 
    # jt.sync_all(True)
    # print (xyz.shape, farthest.shape, batch_indices.shape, centroids.shape, distance.shape)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :]
        centroid = centroid.view(B, 1, 3) 

        dist = jt.sum((xyz - centroid.repeat(1, N, 1)) ** 2, 2) 
        mask = dist < distance
        # distance = mask.ternary(distance, dist)
        # print (mask.size())

        if mask.sum().data[0] > 0: 
            distance[mask] = dist[mask] # bug if mask.sum() == 0 

        farthest = jt.argmax(distance, 1)[0]
        # print (farthest)
        # print (farthest.shape)
    # B, N, C = xyz.size() 
    # sample_list = random.sample(range(0, N), npoint)
    # centroids = jt.zeros((1, npoint)) 
    # centroids[0,:] = jt.array(sample_list)
    # centroids = centroids.view(1, -1).repeat(B, 1)
    # x_center = x[:,sample_list, :]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(npoint, nsample, xyz, points, density_scale = None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    # jt.sync_all(True)
    # print ('11111111111111111')
    new_xyz = index_points(xyz, fps_idx)
    # jt.sync_all(True)
    # print ('2222222222222222222')
    idx = knn_point(nsample, xyz, new_xyz)
    # jt.sync_all(True)
    # print ('333333333333333333')
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = concat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    # jt.sync_all(True)
    # print ('44444444444444444444444')
    
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density

def sample_and_group_all(xyz, points, density_scale = None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    #device = xyz.device
    B, N, C = xyz.shape
    #new_xyz = torch.zeros(B, 1, C).to(device)
    new_xyz = xyz.mean(dim = 1, keepdims = True)
    grouped_xyz = xyz.reshape(B, 1, N, C) - new_xyz.reshape(B, 1, 1, C)
    if points is not None:
        new_points = concat([grouped_xyz, points.view(B, 1, N, -1)], dim=3)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.reshape(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density

def group(nsample, xyz, points):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = jt.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density

class DensityNet(nn.Module):
    def __init__(self, hidden_unit = [8, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList() 

        self.mlp_convs.append(nn.Conv1d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv1d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm1d(1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU() 
    def execute(self, xyz_density):
        B, N = xyz_density.shape 
        density_scale = xyz_density.unsqueeze(1) 

        # for i, conv in enumerate(self.mlp_convs):  
        for i in range(len(self.mlp_convs)):
            # print ('xxxxxxxxxxx', i, len(self.mlp_bns), len(self.mlp_convs))
            bn = self.mlp_bns[i]  
            conv = self.mlp_convs[i]
            # print ('after get bn')
            density_scale = bn(conv(density_scale))
            # print ('after get desity scale')
            if i == len(self.mlp_convs):
                density_scale = self.sigmoid(density_scale) + 0.5
            else:
                density_scale = self.relu(density_scale)
        
        return density_scale

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.relu = nn.ReLU() 
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm(out_channel))
        else:
            self.mlp_convs.append(nn.Conv(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm(out_channel))
        
    def execute(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz

        for i in range (len(self.mlp_convs)):
            conv = self.mlp_convs[i]
            bn = self.mlp_bns[i]
            weights = self.relu(bn(conv(weights)))

        return weights


class PointConvDensitySetInterpolation(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bandwidth):
        super(PointConvDensitySetInterpolation, self).__init__()
        self.bandwidth = bandwidth 
        self.nsample = nsample 
        self.in_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.relu = nn.ReLU()
        last_channel = in_channel
        self.weightnet = WeightNet(3, 16)
        self.densitynet = DensityNet()

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])

    def execute(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N] 
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # print ('xyz1.shape, xyz2.shape')
        # print (xyz1.shape, xyz2.shape, points1.shape, points2.shape)

        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        # points2 = points2.permute(0, 2, 1)
        # print (xyz1.shape, xyz2.shape)
        dists = square_distance(xyz1, xyz2)
        idx, dists = jt.argsort(dists, dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = jt.sum(dist_recip, dim=2, keepdims=True)
        weight = dist_recip / norm
        interpolated_points = jt.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        
        # print ('interpolated_points shape', interpolated_points.shape)

        xyz_density = compute_density(xyz1, self.bandwidth)
        density_scale = self.densitynet(xyz_density)

        new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(N, self.nsample, xyz1, interpolated_points, density_scale.reshape(B, N, 1))

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]

        for i in range(len(self.mlp_convs)):
            conv = self.mlp_convs[i]
            bn = self.mlp_bns[i]
            # print ('new new new point shape', new_points.shape)
            new_points = self.relu(bn(conv(new_points)))

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = new_points * grouped_density.permute(0, 3, 2, 1)
        new_points = jt.matmul(new_points.permute(0, 3, 1, 2), weights.permute(0, 3, 2, 1)).reshape(B, N, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = self.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_points

        # new_points = new_points.permute(0, 2, 1)
        # # l = len(self.mlp_convs)
        # for i, conv in self.mlp_convs.layers.items():
        #     # conv = self.mlp_convs[i]
        #     bn = self.mlp_bns[i]
        #     new_points = self.relu(bn(conv(new_points))) 
        # return new_points.permute(0, 2, 1)


class PointConvDensitySetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.densitynet = DensityNet()

        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.group_all = group_all 
        self.bandwidth = bandwidth
        self.relu = nn.ReLU()
    def execute(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        density_scale = self.densitynet(xyz_density)

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(xyz, points, density_scale.reshape(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(self.npoint, self.nsample, xyz, points, density_scale.reshape(B, N, 1))

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i in range(len(self.mlp_convs)):
            # print ('new_point shape', new_points.shape)
            conv = self.mlp_convs[i]
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = new_points * grouped_density.permute(0, 3, 2, 1)
        new_points = jt.matmul(new_points.permute(0, 3, 1, 2), weights.permute(0, 3, 2, 1)).reshape(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = self.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points




