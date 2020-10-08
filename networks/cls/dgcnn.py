
import numpy as np
import jittor as jt 
from jittor import nn
from jittor.contrib import concat 

from misc.ops import KNN

import time

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


def get_graph_feature(x, knn=None, k=None, idx=None):
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.reshape(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x,x)   # (batch_size, num_points, k)
        idx = idx.permute(0, 2, 1)
    idx_base = jt.array(np.arange(0, batch_size)).reshape(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.reshape(-1)

    _, num_dims, _ = x.shape

    x = x.transpose(0, 2, 1)   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.reshape(batch_size*num_points, -1)[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims)
    x = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = concat((feature-x, x), dim=3).transpose(0, 3, 1, 2)
    return feature

def knn (x, k):
    inner = -2 * jt.nn.bmm(x.transpose(0, 2, 1), x)
    xx = jt.sum(x ** 2, dim = 1, keepdims=True)
    distance = -xx - inner - xx.transpose(0, 2, 1)
    idx = topk(distance ,k=k, dim=-1)[1]
    return idx



class DGCNN(nn.Module):
    def __init__(self, n_classes=40):
        super(DGCNN, self).__init__()
        self.k = 20
        self.knn = KNN(self.k)
        self.bn1 = nn.BatchNorm(64)
        self.bn2 = nn.BatchNorm(64)
        self.bn3 = nn.BatchNorm(128)
        self.bn4 = nn.BatchNorm(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(scale=0.2))
        self.conv2 = nn.Sequential(nn.Conv(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(scale=0.2))
        self.conv3 = nn.Sequential(nn.Conv(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(scale=0.2))
        self.conv4 = nn.Sequential(nn.Conv(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(scale=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(scale=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, n_classes)

    def execute(self, x):
        #print (x.shape)
        # x = x.transpose(0, 2, 1) # b, n, c -> b, c, n
        batch_size = x.shape[0]
        
        x = get_graph_feature(x, knn=self.knn, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdims=False)
        x = get_graph_feature(x1, knn=self.knn, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdims=False)
        x = get_graph_feature(x2, knn=self.knn, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdims=False)
        x = get_graph_feature(x3, knn=self.knn, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdims=False)
        x = concat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x) #  ############ this line  
        x1 = x.max(dim=2).reshape(batch_size, -1) 
        x2 = x.mean(dim=2).reshape(batch_size, -1)
        x = concat((x1, x2), 1)
        x = nn.leaky_relu(self.bn6(self.linear1(x)), scale=0.2)
        x = self.dp1(x)
        x = nn.leaky_relu(self.bn7(self.linear2(x)), scale=0.2)
        x = self.dp2(x)
        x = self.linear3(x) 
        return x


def main():
    net = DGCNN()
    x = jt.array(np.random.random((32, 3, 256))) 
    print (x.shape)
    out = net(x)
    print (out.shape)
if __name__ == '__main__' :
    main()
