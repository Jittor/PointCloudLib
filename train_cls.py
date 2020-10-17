import numpy as np
from tqdm import tqdm
import jittor as jt
from jittor import nn
from jittor.contrib import concat 

jt.flags.use_cuda = 1

from networks.cls.pointnet2 import PointNet2_cls
from networks.cls.pointnet import PointNet as  PointNet_cls
from networks.cls.dgcnn import DGCNN
from networks.cls.pointcnn import PointCNNcls
from networks.cls.pointconv import PointConvDensityClsSsg

import math 

from data_utils.modelnet40_loader import ModelNet40
from misc.utils import LRScheduler
import argparse


import time 

def freeze_random_seed():
    np.random.seed(0)


def soft_cross_entropy_loss(output, target, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    target = target.view(-1)
    softmax = nn.Softmax(dim=1)
    if smoothing:
        eps = 0.2
        b, n_class = output.shape

        one_hot = jt.zeros(output.shape)
        for i in range (b):
            one_hot[i, target[i].data] = 1

        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        # print (one_hot[0].data)
        log_prb = jt.log(softmax(output))
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = nn.cross_entropy_loss(output, target)

    return loss


def train(net, optimizer, epoch, dataloader, args):
    net.train()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')
    for pts, normals, labels in pbar:
        
        # #output = net(pts, normals) 
        
        
        if args.model == 'pointnet' or args.model == 'dgcnn' :
            pts = pts.transpose(0, 2, 1)

        if args.model == 'pointnet2':
            output = net(pts, normals)
        else :
            output = net(pts)

        # loss = nn.cross_entropy_loss(output, labels)
        loss = soft_cross_entropy_loss(output, labels)
        optimizer.step(loss) 
        pred = np.argmax(output.data, axis=1)
        acc = np.mean(pred == labels.data) * 100
        pbar.set_description(f'Epoch {epoch} [TRAIN] loss = {loss.data[0]:.2f}, acc = {acc:.2f}')

def evaluate(net, epoch, dataloader, args):
    total_acc = 0
    total_num = 0

    net.eval()
    total_time = 0.0
    for pts, normals, labels in tqdm(dataloader, desc=f'Epoch {epoch} [Val]'):
        # pts = jt.float32(pts.numpy())
        # normals = jt.float32(normals.numpy())
        # labels = jt.int32(labels.numpy())
        # feature = concat((pts, normals), 2)
        if args.model == 'pointnet' or args.model == 'dgcnn' :
            pts = pts.transpose(0, 2, 1)

        # pts = pts.transpose(0, 2, 1) # for pointnet DGCNN

        # output = net(pts, feature)
        if args.model == 'pointnet2':
            output = net(pts, normals)
        else :
            output = net(pts)
        # output = net()
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0] 
    acc = 0.0
    acc = total_acc / total_num
    return acc


if __name__ == '__main__':
    freeze_random_seed()
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='[pointnet]', metavar='N',
                        choices=['pointnet', 'pointnet2', 'pointcnn', 'dgcnn', 'pointconv'],
                        help='Model to use')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 0.02)')    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train ')

    args = parser.parse_args()


    if args.model == 'pointnet':
        net = PointNet_cls()
    elif args.model == 'pointnet2':
        net = PointNet2_cls()
    elif args.model == 'pointcnn':
        net = PointCNNcls()
    elif args.model == 'dgcnn':
        net = DGCNN()
    elif args.model == 'pointconv':
        net = PointConvDensityClsSsg()
    else:
        raise Exception("Not implemented")

    base_lr = args.lr
    optimizer = nn.SGD(net.parameters(), lr = base_lr, momentum = args.momentum)

    lr_scheduler = LRScheduler(optimizer, base_lr)

    batch_size = args.batch_size
    n_points = args.num_points
    train_dataloader = ModelNet40(n_points=n_points, batch_size=batch_size, train=True, shuffle=True)
    val_dataloader = ModelNet40(n_points=n_points, batch_size=batch_size, train=False, shuffle=False)

    step = 0
    best_acc = 0
    for epoch in range(args.epochs):
        lr_scheduler.step(len(train_dataloader) * batch_size)

        train(net, optimizer, epoch, train_dataloader, args)
        acc = evaluate(net, epoch, val_dataloader, args)

        best_acc = max(best_acc, acc)
        print(f'val acc={acc:.4f}, best={best_acc:.4f}')
