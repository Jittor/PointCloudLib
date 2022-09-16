import numpy as np
import jittor as jt 
from jittor import nn
from jittor.contrib import concat 
import math 
from typing import Tuple, Callable, Optional, Union
from misc.ops import FurthestPointSampler, KNN
import time 


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def execute(self, x):
        batchsize = x.shape[0]
        # print ('x shape =', x.shape)
        x = self.conv1(x)
        # print ('x shape =', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # print ('before max shape is', x.shape)
        x = jt.max(x, 2)

        # print ('after max shape is', x.shape)
        x = x.reshape(-1, 1024)

        #x = self.relu(self.bn4(self.fc1(x)))
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = ((jt.array(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).reshape(1,9)).repeat(batchsize, 1)
        # print (iden.shape)
        x = x + iden
        # print (x.shape)
        x = x.reshape(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def execute(self, x):
        batchsize = x.size()[0] 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = jt.max(x, 2)
        x = x.reshape(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = jt.array(np.eye(self.k).flatten().astype(np.float32)).reshape(1,self.k*self.k)
        x = x + iden
        x = x.reshape(-1, self.k, self.k)
        return x




def EndChannels(f, make_contiguous=False):
    """ Class decorator to apply 2D convolution along end channels. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def execute(self, x):
            x = x.permute(0,3,1,2)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()


def EndChannels1d(f, make_contiguous=False):
    """ Class decorator to apply 2D convolution along end channels. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def execute(self, x):
            x = x.permute(0,2,1)
            x = self.f(x)
            x = x.permute(0,2,1)
            return x

    return WrappedLayer()


class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]],
                 depth_multiplier = 1, with_bn = True,
                 activation = nn.ReLU()) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
            nn.Conv(in_channels * depth_multiplier, out_channels, 1, bias = not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm(out_channels, momentum = 0.9) if with_bn else None

    def execute(self, x):
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x



class Conv(nn.Module):
    """
    2D convolutional layer with optional activation and batch normalization.
    """

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size, with_bn = True,
                 activation = nn.ReLU()) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv(in_channels, out_channels, kernel_size, bias = not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm(out_channels, momentum = 0.9) if with_bn else None

    def execute(self, x):
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x



class Dense_Conv1d(nn.Module):
    def __init__(self, in_features : int, out_features : int,
                drop_rate : int = 0, with_bn : bool = True,
                activation = nn.ReLU()
            ) -> None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense_Conv1d, self).__init__()

        self.linear = nn.Conv1d(in_features, out_features, 1)
        self.activation = activation
        self.with_bn = with_bn
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.bn = nn.BatchNorm1d(out_features) if with_bn else None

    def execute(self, x): 
        x = self.linear(x)

        if self.with_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.drop:
            x = self.drop(x)
        return x 


class Dense_Conv2d(nn.Module):
    def __init__(self, in_features : int, out_features : int,
                drop_rate : int = 0, with_bn : bool = True,
                activation = nn.ReLU(), groups=1) -> None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense_Conv2d, self).__init__()

        self.linear = nn.Conv(in_features, out_features, 1, groups=groups)
        self.activation = activation
        self.with_bn = with_bn
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.bn = nn.BatchNorm(out_features) if with_bn else None

    def execute(self, x): 
        x = self.linear(x)

        # print ('before bn shape', x.shape)
        if self.with_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.drop:
            x = self.drop(x)

        return x 


class RandPointCNN_Decoder(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in : int, C_out : int, C_last : int, dims : int, K : int, D : int, P : int) -> None:
        """ See documentation for PointCNN. """
        super(RandPointCNN_Decoder, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P)
        self.P = P
        self.conv_fuse = EndChannels1d(Dense_Conv1d(C_out + C_last, C_out))
        # if self.P > 0:
        # self.sampler = FurthestPointSampler(self.P)

    def execute(self, x_l, x_h): # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts_l, fts_l = x_l
        pts_h, fts_h = x_h
        rep_pts_fts = self.pointcnn((pts_h, pts_l, fts_l))
        concat_feature = concat ((rep_pts_fts, fts_h), dim=2)
        rep_pts_fts = self.conv_fuse(concat_feature)
        return pts_h, rep_pts_fts


class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int) -> None:
        """ See documentation for PointCNN. """
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P)
        self.P = P
        if self.P > 0:
            self.sampler = FurthestPointSampler(self.P)

    def execute(self, x): # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            rep_pts = self.sampler(pts)
            # idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
            # rep_pts = pts[:,idx,:]

        else:
            rep_pts = pts
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        return rep_pts, rep_pts_fts



class PointCNN(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        :param r_indices_func: Selector function of the type,
          INPUTS
          rep_pts : Representative points.
          pts  : Point cloud.
          K : Number of points for each region.
          D : "Spread" of neighboring points.

          OUTPUT
          pts_idx : Array of indices into pts such that pts[pts_idx] is the set
          of points in the "region" around rep_pt.
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4

        if C_in == 0:
            depth_multiplier = 4
        else:
            # depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)
            depth_multiplier = int(np.ceil(C_out / C_in))

        self.knn = KNN(K * D)
        self.dense = EndChannels1d(Dense_Conv1d(C_in, C_out // 2)) if C_in != 0 else None

        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        
        self.D = D
        self.K = K

    def select_region(self, pts,  # (N, x, dims)
                      pts_idx): # (P, K, dims)

        regions = jt.stack([
            pts[n][idx,:] for n, idx in enumerate(jt.misc.unbind(pts_idx, dim = 0))
        ], dim = 0) 

        return regions

    def execute(self, x):              # (N, P, C_out)
        
        rep_pts, pts, fts = x         
        # print ('input size =', rep_pts.shape, fts.shape, fts.shape)
        fts = self.dense(fts) if fts is not None else fts # B, N, D
        tmp_rep_pts = rep_pts.permute(0, 2, 1)
        tmp_pts = pts.permute(0, 2, 1)
        # pts_idx = knn_indices_func_gpu(rep_pts, pts, self.K, self.D)
        pts_idx = self.knn(tmp_rep_pts, tmp_pts) # b, d, n 
        pts_idx = pts_idx[:,0::self.D, :]
        pts_idx = pts_idx.permute(0, 2, 1)
        # jt.sync_all(True)
        # start_time = time.time()
        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        # jt.sync_all(True)
        # end_time = time.time()
        # print ('select region run time =', end_time - start_time)
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))
        return fts_p



class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.  """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int,
                 P : int, C_mid : int, depth_multiplier : int) -> None:
        super(XConv, self).__init__()

        self.C_in = C_in
        self.C_mid = C_mid
        self.dims = dims
        self.K = K

        self.P = P

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers
        self.dense1 = Dense_Conv2d(dims, C_mid)
        self.dense2 = Dense_Conv2d(C_mid, C_mid)

        # Layers to generate X
        self.x_trans_0 = Conv(
                in_channels = dims,
                out_channels = K*K,
                kernel_size = (1, K),
                with_bn = True)
        self.x_trans_1 = Dense_Conv2d(K*K, K*K, with_bn = True, groups=1)
        self.x_trans_2 = Dense_Conv2d(K*K, K*K, with_bn = False, activation = None, groups=1)

        self.end_conv = EndChannels(SepConv(
            in_channels = C_mid + C_in,
            out_channels = C_out,
            kernel_size = (1, K),
            depth_multiplier = depth_multiplier
        ))
        
    def execute(self, x):                        # (N, K, C_out)
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x  # b, n, c // b ,n k, c // b, n, k, d
        if fts is not None:
            assert(rep_pt.size()[0] == pts.size()[0] == fts.size()[0])  # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1] == fts.size()[1])  # Check P is equal.
            assert(pts.size()[2] == fts.size()[2] == self.K)            # Check K is equal.
            assert(fts.size()[3] == self.C_in)                          # Check C_in is equal.
        else:
            assert(rep_pt.size()[0] == pts.size()[0])                   # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1])                   # Check P is equal.
            assert(pts.size()[2] == self.K)                             # Check K is equal.
        assert(rep_pt.size()[2] == pts.size()[3] == self.dims)          # Check dims is equal.

        N = pts.size()[0]
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = jt.unsqueeze(rep_pt, dim = 2)  # (N, P, 1, dims)
        # print (p_center.size()) # 
        # Move pts to local coordinate system of rep_pt.
        pts_local = pts - p_center.repeat(1, 1, self.K, 1)  # (N, P, K, dims)
        # pts_local = self.pts_layernorm(pts - p_center)

        # Individually lift each point into C_mid space.
        # print (pts_local.size(), 'before size')  
        pts_local = pts_local.permute(0, 3, 1, 2) # N, dim, P, K
        fts_lifted0 = self.dense1(pts_local) # ?  
        # print (.size(), 'after size')  
        fts_lifted  = self.dense2(fts_lifted0)  # N, C_mid, P, K

        fts = fts.permute(0, 3, 1, 2)
        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = concat((fts_lifted, fts), 1)  # (N, C_mid + C_in, P, K)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, P, self.K, self.K) 
        # X = self.x_trans(pts_local)  # N, K*K, 1, P
        x = self.x_trans_0(pts_local)
        x = self.x_trans_1(x)
        X = self.x_trans_2(x)
        
        # print ('X size ', X.size())
        X = X.permute(0, 2, 3, 1)  # n p 1 k
        X = X.view(X_shape) # N, P, K, K

        
        # print (fts_cat.shape)
        fts_cat = fts_cat.permute(0, 2, 3, 1)
        fts_X = jt.matmul(X, fts_cat) # 
        
        # print ('fts X size =', fts_X.shape)
        
        fts_p = self.end_conv(fts_X).squeeze(dim = 2) 
        # print ('xxxxxxxxxxx')
        # print ('result size')
        # print (fts_X.size(), fts_p.size())

        return fts_p
