import jittor as jt 
from jittor import nn 
from jittor.contrib import concat 

import numpy as np 
import math 

from misc.layers import STN3d, STNkd





class PointNet_partseg(nn.Module):
    def __init__(self, part_num=50):
        super(PointNet_partseg, self).__init__()
        self.part_num = part_num
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = nn.Conv1d(4944, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
    def execute(self, point_cloud, label):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(0, 2, 1)
        point_cloud = nn.bmm(point_cloud, trans)

        point_cloud = point_cloud.transpose(0, 2, 1)

        out1 = self.relu(self.bn1(self.conv1(point_cloud)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(0, 2, 1)
        net_transformed = nn.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(0, 2, 1)

        out4 = self.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = jt.argmax(out5, 2, keepdims=True)[1]
        out_max = out_max.view(-1, 2048)

        out_max = concat((out_max, label),1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat_feature = concat([expand, out1, out2, out3, out4, out5], 1)
        net = self.relu(self.bns1(self.convs1(concat_feature)))
        net = self.relu(self.bns2(self.convs2(net)))
        net = self.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        return net

if __name__ == '__main__':
    from jittor import init 
    x_input = init.invariant_uniform([16, 3, 1024], dtype='float')
    cls_input = init.invariant_uniform([16, 16], dtype='float')
    model = PointNet_partseg()
    out = model(x_input, cls_input)
    print (out.size())

