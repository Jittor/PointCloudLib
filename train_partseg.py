

import os
import sys 
from data_utils.shapenet_loader import ShapeNetPart
import numpy as np
import sklearn.metrics as metrics
from misc.utils import LRScheduler
from networks.seg.pointnet_partseg import PointNet_partseg
from networks.seg.pointnet2_partseg import PointNet2_partseg
from networks.seg.pointconv_partseg import PointConvDensity_partseg
from networks.seg.dgcnn_partseg import DGCNN_partseg
from networks.seg.pointcnn_partseg import PointCNN_partseg

import time 

# jittor related 
import jittor as jt 
from jittor import nn 

import argparse

jt.flags.use_cuda = 1

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    # label = label.squeeze(-1)
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            # print (label[shape_idx][0])
            idx = label[shape_idx][0]
            start_index = index_start[idx]
            num = seg_num[idx]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            # print ('iou ', part, iou)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    # cal average class iou 
    id2cat = ['airplane', 'bag', 'cap', 'car', 'chair', 
                'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                'motor', 'mug', 'pistol', 'rocket', 'skateboard','table']

    # for cat in range (len(id2cat)):
    #     class_avg_iou = []
    #     for idx, iou in enumerate(shape_ious):
    #         if label[idx] == cat:
    #             class_avg_iou.append(iou)
    #     print (id2cat[cat], 'iou =', np.mean(class_avg_iou))

    return shape_ious


def train(model, args):
    batch_size = 16
    train_loader = ShapeNetPart(partition='trainval', num_points=2048, class_choice=None, batch_size=batch_size, shuffle=True)
    test_loader = ShapeNetPart(partition='test', num_points=2048, class_choice=None, batch_size=batch_size, shuffle=False)
    
    seg_num_all = 50
    seg_start_index = 0

    print(str(model))
    base_lr = 0.01
    optimizer = nn.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = LRScheduler(optimizer, base_lr)

    # criterion = nn.cross_entropy_loss() # here

    best_test_iou = 0
    for epoch in range(200):
        ####################
        # Train
        ####################
        lr_scheduler.step(len(train_loader) * batch_size)
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []

        # debug = 0
        for data, label, seg in train_loader:
            # with jt.profile_scope() as report:
            
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            # print (label.size())
            for idx in range(label.shape[0]):
                label_one_hot[idx, label.numpy()[idx,0]] = 1
            label_one_hot = jt.array(label_one_hot.astype(np.float32))
            if args.model == 'pointnet' or args.model == 'dgcnn':            
                data = data.permute(0, 2, 1) # for pointnet it should not be committed 
            batch_size = data.size()[0]
            if args.model == 'pointnet2':
                seg_pred = model(data, data, label_one_hot)
            else :
                seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1)
            # print (seg_pred.size())
            # print (seg_pred.size(), seg.size())
            loss = nn.cross_entropy_loss(seg_pred.view(-1, seg_num_all), seg.view(-1))
            # print (loss.data)
            optimizer.step(loss)

            pred = jt.argmax(seg_pred, dim=2)[0]               # (batch_size, num_points)
            # print ('pred size =', pred.size(), seg.size())
            count += batch_size
            train_loss += loss.numpy() * batch_size
            seg_np = seg.numpy()                  # (batch_size, num_points)
            pred_np = pred.numpy()    # (batch_size, num_points)
            # print (type(label))

            label = label.numpy() # added 
            
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            temp_label = label.reshape(-1, 1)
        
            train_label_seg.append(temp_label)
        
            # print(report)
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        # print (train_true_cls.shape ,train_pred_cls.shape)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)

        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls.data, train_pred_cls.data)
        # print ('train acc =',train_acc, 'avg_per_class_acc', avg_per_class_acc)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        # print (len(train_pred_seg), train_pred_seg[0].shape)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        # print (len(train_label_seg), train_label_seg[0].size())
        # print (train_label_seg[0])
        train_label_seg = np.concatenate(train_label_seg, axis=0)
        # print (train_pred_seg.shape, train_true_seg.shape, train_label_seg.shape)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, None)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        # io.cprint(outstr)
        print (outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in test_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label.numpy()[idx,0]] = 1
            label_one_hot = jt.array(label_one_hot.astype(np.float32))
            if args.model == 'pointnet' or args.model == 'dgcnn':            
                data = data.permute(0, 2, 1) # for pointnet it should not be committed 
            batch_size = data.size()[0]
            if args.model == 'pointnet2':
                seg_pred = model(data, data, label_one_hot)
            else :
                seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1)
            loss = nn.cross_entropy_loss(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze(-1))
            pred = jt.argmax(seg_pred, dim=2)[0]
            count += batch_size
            test_loss += loss.numpy() * batch_size
            seg_np = seg.numpy()
            pred_np = pred.numpy()
            label = label.numpy() # added 

            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1, 1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, None)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        print (outstr)
        # io.cprint(outstr)
        # if np.mean(test_ious) >= best_test_iou:
        #     best_test_iou = np.mean(test_ious)
        #     torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)



if __name__ == "__main__":
    # Training settings    
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
        model = PointNet_partseg(part_num=50)
    elif args.model == 'pointnet2':
        model = PointNet2_partseg (part_num=50)
    elif args.model == 'pointcnn':
        model = PointCNN_partseg(part_num=50)
    elif args.model == 'dgcnn':
        model = DGCNN_partseg(part_num=50)
    elif args.model == 'pointconv':
        model = PointConvDensity_partseg(part_num=50)
    else:
        raise Exception("Not implemented")

    train(model, args)
