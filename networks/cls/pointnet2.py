from typing import List, Optional, Tuple

import jittor as jt
import jittor.nn as nn
from jittor import init

from misc.ops import FurthestPointSampler
from misc.ops import BallQueryGrouper
from misc.ops import GroupAll

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
        new_xyz = self.sampler(xyz) if self.n_points is not None else None

        new_feature_list = []
        l = len (self.groupers)
        for i in range (l):
            grouper = self.groupers[i]
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
        for mlp in mlps.layers.items():
            self.mlps.append(self.build_mlps(mlp, use_xyz))


class PointNet2_cls(nn.Module):
    def __init__(self, n_classes=40, use_xyz=True):
        super().__init__()

        self.n_classes = n_classes
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

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.n_classes),
        )

    def execute(self, xyz, feature):
        l = len(self.pointnet_modules)
        for i in range (l):
            module = self.pointnet_modules[i]
            xyz, feature = module(xyz, feature)
        # for module in self.pointnet_modules:
        #     xyz, feature = module(xyz, feature)

        feature = feature.squeeze(dim=1)
        return self.fc_layer(feature)


class PointNetMSG(PointNet2_cls):
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
    model = PointNet(n_classes=40)
    input_point = init.gauss([2, 1024, 3], 'float', mean=0.0)
    input_feature = init.gauss([2, 1024, 3], 'float', mean=0.0)
    print (input_point.shape)
    print (input_feature.shape)
    outputs = model(input_point, input_feature)
    print (outputs.shape)


if __name__ == '__main__' :
    main()
