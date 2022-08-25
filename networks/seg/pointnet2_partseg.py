from typing import List, Optional, Tuple

import jittor as jt
import jittor.nn as nn
from jittor import init
from jittor.contrib import concat 


from misc.ops import FurthestPointSampler
from misc.ops import BallQueryGrouper
from misc.ops import GroupAll
from misc.ops import PointNetFeaturePropagation







class PointNetModuleBase(nn.Module):
    def __init__(self):
        self.n_points = None
        self.sampler = None
        self.groupers = None
        self.mlps = None

    def build_mlps(self, mlp_spec: List[int], use_xyz: bool=True, 
                   bn: bool = True) -> nn.Sequential:
        layers = []
        
        if use_xyz:
            mlp_spec[0] += 3

        for i in range(1, len(mlp_spec)):
            layers.append(nn.Conv(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm(mlp_spec[i]))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def execute(self, xyz: jt.Var, feature: Optional[jt.Var]) -> Tuple[jt.Var, jt.Var]:
        '''
        Parameters
        ----------
        xyz: jt.Var, (B, N, 3)
        feature: jt.Var, (B, N, C)

        Returns
        -------
        new_xyz: jt.Var, (B, n_points, 3)
        new_feature: jt.Var, (B, n_points, C')
        '''
        B, _, _ = xyz.shape 
        new_xyz = self.sampler(xyz) if self.n_points is not None else jt.zeros((B, 1, 3))

        new_feature_list = []
        # print (self.groupers) 
        # lens = 
        for i, grouper in self.groupers.layers.items():
            new_feature = grouper(new_xyz, xyz, feature)
            # [B, n_points, n_samples, C] -> [B, C, n_points, n_samples]
            new_feature = new_feature.transpose(0, 3, 1, 2)
            new_feature = self.mlps[i](new_feature)
            # [B, C, n_points, n_samples] -> [B, n_points, n_samples, C]
            new_feature = new_feature.transpose(0, 2, 3, 1)
            new_feature = new_feature.argmax(dim=2)[1]

            new_feature_list.append(new_feature)

        new_feature = jt.contrib.concat(new_feature_list, dim=-1)
        return new_xyz, new_feature


class PointnetModule(PointNetModuleBase):
    def __init__(self, mlp: List[int], n_points=None, radius=None, 
                 n_samples=None, bn=True, use_xyz=True):
        super().__init__()

        self.n_points = n_points
        
        self.groupers = nn.ModuleList()
        if self.n_points is not None:
            self.sampler = FurthestPointSampler(n_points)
            self.groupers.append(BallQueryGrouper(radius, n_samples, use_xyz))
        else:
            self.groupers.append(GroupAll(use_xyz))

        self.mlps = nn.ModuleList()
        self.mlps.append(self.build_mlps(mlp, use_xyz))


class PointnetModuleMSG(PointNetModuleBase):
    def __init__(self, n_points: int, radius: List[float], n_samples: List[int],
                 mlps: List[List[int]], bn=True, use_xyz=True):
        super().__init__()

        self.n_points = n_points
        self.sampler = FurthestPointSampler(n_points)

        self.groupers = nn.ModuleList()
        for r, s in zip(radius, n_samples):
            self.groupers.append(BallQueryGrouper(r, s, use_xyz))

        self.mlps = nn.ModuleList()
        for mlp in mlps:
            self.mlps.append(self.build_mlps(mlp, use_xyz))


class PointNet2_partseg(nn.Module):
    def __init__(self, part_num=50, use_xyz=True):
        super().__init__()
        self.part_num = part_num
        self.use_xyz = use_xyz
        self.build_model()

    def build_model(self):
        self.pointnet_modules = nn.ModuleList()
        self.pointnet_modules.append(
            PointnetModule(
                n_points=512,
                radius=0.2,
                n_samples=64,
                mlp=[3, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )

        self.pointnet_modules.append(
            PointnetModule(
                n_points=128,
                radius=0.4,
                n_samples=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.use_xyz,
            )
        )

        self.pointnet_modules.append(
            PointnetModule(
                mlp=[256, 256, 512, 1024],
                use_xyz=self.use_xyz,
            )
        )

        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6, mlp=[128, 128, 128])
        

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.part_num, 1)
        )

    def execute(self, xyz, feature, cls_label):
        # for module in self.pointnet_modules:
        #     xyz, feature = module(xyz, feature)
        
        B, N, _ = xyz.shape
        l1_xyz, l1_feature = self.pointnet_modules[0](xyz, feature)
        l2_xyz, l2_feature = self.pointnet_modules[1](l1_xyz, l1_feature)
        l3_xyz, l3_feature = self.pointnet_modules[2](l2_xyz, l2_feature) 
        # print ('before interpolate shape')
        # print (l2_xyz.shape, l2_feature.shape, l3_xyz.shape, l3_feature.shape)
        l2_feature = self.fp3(l2_xyz, l3_xyz, l2_feature, l3_feature)
        l1_feature = self.fp2(l1_xyz, l2_xyz, l1_feature, l2_feature)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N).permute(0, 2, 1)
        # print ('before concat size ')
        # print (cls_label_one_hot.size(),xyz.size(),feature.size())
        feature = self.fp1(xyz, l1_xyz, concat([cls_label_one_hot,xyz,feature],2), l1_feature)
        feature = feature.permute(0, 2, 1)
        # print (feature.shape)
        return self.fc_layer(feature)


class PointNetMSG(PointNet2_partseg):
    def build_model(self):
        super().build_model()

        self.pointnet_modules = nn.ModuleList()
        self.pointnet_modules.append(
            PointnetModuleMSG(
                n_points=512,
                radius=[0.1, 0.2, 0.4],
                n_samples=[16, 32, 128],
                mlps=[[3, 32, 32, 64], [3, 64, 64, 128], [3, 64, 96, 128]],
                use_xyz=self.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.pointnet_modules.append(
            PointnetModuleMSG(
                n_points=128,
                radius=[0.2, 0.4, 0.8],
                n_samples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.use_xyz,
            )
        )

        self.pointnet_modules.append(
            PointnetModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.use_xyz,
            )
        )

def main():
    model = PointNet2_partseg()
    input_point = init.gauss([2, 1024, 3], 'float', mean=0.0)
    input_feature = init.gauss([2, 1024, 3], 'float', mean=0.0)
    cls_label = init.gauss([2, 16], 'float', mean=0.0)
    
    print (input_point.shape)
    print (input_feature.shape)
    print (cls_label.shape)
    outputs = model(input_point, input_feature, cls_label)
    print (outputs.shape)


if __name__ == '__main__' :
    main()
