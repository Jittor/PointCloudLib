import numpy as np
import jittor as jt 
from jittor import nn
from jittor.contrib import concat

from misc.pointconv_utils import PointConvDensitySetAbstraction

class PointConvDensityClsSsg(nn.Module):
    def __init__(self, n_classes = 40):
        super(PointConvDensityClsSsg, self).__init__()
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU() 
    
    def execute(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        B, _, _ = xyz.shape
        
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.reshape(B, 1024) # to reshape 
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


