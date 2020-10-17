"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import numpy as np
import jittor as jt 
from jittor import nn
from jittor.contrib import concat

from misc.pointconv_utils import PointConvDensitySetAbstraction, PointConvDensitySetInterpolation

class PointConvDensity_partseg(nn.Module):
    def __init__(self, part_num=50):
        super(PointConvDensity_partseg, self).__init__()
        self.part_num = part_num 

        self.sa0 = PointConvDensitySetAbstraction(npoint=1024, nsample=32, in_channel=3, mlp=[32,32,64], bandwidth = 0.1, group_all=False)
        self.sa1 = PointConvDensitySetAbstraction(npoint=256, nsample=32, in_channel=64 + 3, mlp=[64,64,128], bandwidth = 0.2, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=32, in_channel=128 + 3, mlp=[128,128,256], bandwidth = 0.4, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=36, nsample=32, in_channel=256 + 3, mlp=[256,256,512], bandwidth = 0.8, group_all=False)
        

        # TODO upsample  
        # upsampling 
        # def __init__(self, nsample, in_channel, mlp, bandwidth):

        self.in0 = PointConvDensitySetInterpolation(nsample=16, in_channel=512 + 3, mlp=[512,512], bandwidth=0.8)
        self.in1 = PointConvDensitySetInterpolation(nsample=16, in_channel=512 + 3, mlp=[256,256], bandwidth=0.4)
        self.in2 = PointConvDensitySetInterpolation(nsample=16, in_channel=256 + 3, mlp=[128,128], bandwidth=0.2)
        self.in3 = PointConvDensitySetInterpolation(nsample=16, in_channel=128 + 3, mlp=[128,128, 128], bandwidth=0.1)
        
        # self.fp0 = PointConvDensitySetAbstraction(npoint=1024, nsample=32, in_channel=3, mlp=[32,32,64], bandwidth = 0.1, group_all=False)
        # self.fp1 = PointConvDensitySetAbstraction(npoint=256, nsample=32, in_channel=64 + 3, mlp=[64,64,128], bandwidth = 0.2, group_all=False)
        # self.fp2 = PointConvDensitySetAbstraction(npoint=64, nsample=32, in_channel=128 + 3, mlp=[128,128,256], bandwidth = 0.4, group_all=False)
        # self.fp3 = PointConvDensitySetAbstraction(npoint=36, nsample=32, in_channel=256 + 3, mlp=[256,256,512], bandwidth = 0.8, group_all=False)
        
        self.fc1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc3 = nn.Conv1d(128, self.part_num, 1)
        self.relu = nn.ReLU() 
    
    def execute(self, xyz, cls_label):
        xyz = xyz.permute(0, 2, 1)
        B, _, _ = xyz.shape 
        
        l1_xyz, l1_points = self.sa0(xyz, None)
        l2_xyz, l2_points = self.sa1(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa2(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa3(l3_xyz, l3_points)
        # print ('after encoder shape =',l4_xyz.shape, l4_points.shape)
        # def execute(self, xyz1, xyz2, points1, points2):

        l3_points = self.in0(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.in1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.in2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.in3(xyz, l1_xyz, xyz, l1_points)

        # print ('after decoder shape =', l0_points.shape)
        
        x = self.drop1(self.relu(self.bn1(self.fc1(l0_points))))
        x = self.fc3(x)
        x = x.permute(0, 2, 1)
        return x


