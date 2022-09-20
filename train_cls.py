import numpy as np
from tqdm import tqdm
import jittor as jt
from jittor import nn
from jittor.contrib import concat 
import torch
jt.flags.use_cuda = 1

from networks.cls.pointnet2 import PointNet2_cls
from networks.cls.pointnet import PointNet as  PointNet_cls
from networks.cls.dgcnn import DGCNN
from networks.cls.pointcnn import PointCNNcls
from networks.cls.pointconv import PointConvDensityClsSsg
from networks.cls.kpconv import KPCNN
import math 
from os.path import exists, join
from data_utils.modelnet40_loader import ModelNet40
from data_utils.kpconv_loader import KPConvLoader
from misc.utils import LRScheduler
import argparse
from datasets.ModelNet40 import ModelNet40Dataset, ModelNet40Sampler, ModelNet40CustomBatch, Modelnet40Config

import time 
import pickle
from jittor_utils import auto_diff

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

        loss = soft_cross_entropy_loss(output, labels)
        optimizer.step(loss) 
        pred = np.argmax(output.data, axis=1)
        acc = np.mean(pred == labels.data) * 100
        pbar.set_description(f'Epoch {epoch} [TRAIN] loss = {loss.data[0]:.2f}, acc = {acc:.2f}')

def train_kpconv(net, optimizer, epoch, dataloader: KPConvLoader):
    net.train()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')
    jt.sync_all(True)
    for input_list in pbar:
        L = (len(input_list) - 5) // 4
        labels = jt.array(input_list[4 * L + 1]).squeeze(0)
        output = net(input_list)
        loss = soft_cross_entropy_loss(output, labels)
        optimizer.step(loss)
        pred = np.argmax(output.data, axis=1)
        acc = np.mean(pred == labels.data) * 100
        pbar.set_description(f'Epoch {epoch} [TRAIN] loss = {loss.data[0]:.2f}, acc = {acc:.2f}')
        # jt.display_memory_info()

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

def evaluate_kpconv(net, epoch, dataloader: KPConvLoader):
    total_acc = 0
    total_num = 0

    net.eval()
    total_time = 0.0
    for input_list in tqdm(dataloader, desc=f'Epoch {epoch} [Val]'):
        L = (len(input_list) - 5) // 4
        labels = jt.array(input_list[4 * L + 1]).squeeze(0)
        output = net(input_list)

        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0] 
    acc = 0.0
    acc = total_acc / total_num
    return acc


def fast_confusion(true, pred, label_values=None): # used only by kpconv test
    """
    Fast confusion matrix (100x faster than Scikit learn). But only works if labels are la
    :param true:
    :param false:
    :param num_classes:
    :return:
    """

    # Ensure data is in the right format
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    if len(true.shape) != 1:
        raise ValueError('Truth values are stored in a {:d}D array instead of 1D array'. format(len(true.shape)))
    if len(pred.shape) != 1:
        raise ValueError('Prediction values are stored in a {:d}D array instead of 1D array'. format(len(pred.shape)))
    if true.dtype not in [np.int32, np.int64]:
        raise ValueError('Truth values are {:s} instead of int32 or int64'.format(true.dtype))
    if pred.dtype not in [np.int32, np.int64]:
        raise ValueError('Prediction values are {:s} instead of int32 or int64'.format(pred.dtype))
    true = true.astype(np.int32)
    pred = pred.astype(np.int32)

    # Get the label values
    if label_values is None:
        # From data if they are not given
        label_values = np.unique(np.hstack((true, pred)))
    else:
        # Ensure they are good if given
        if label_values.dtype not in [np.int32, np.int64]:
            raise ValueError('label values are {:s} instead of int32 or int64'.format(label_values.dtype))
        if len(np.unique(label_values)) < len(label_values):
            raise ValueError('Given labels are not unique')

    # Sort labels
    label_values = np.sort(label_values)

    # Get the number of classes
    num_classes = len(label_values)

    #print(num_classes)
    #print(label_values)
    #print(np.max(true))
    #print(np.max(pred))
    #print(np.max(true * num_classes + pred))

    # Start confusion computations
    if label_values[0] == 0 and label_values[-1] == num_classes - 1:

        # Vectorized confusion
        vec_conf = np.bincount(true * num_classes + pred)

        # Add possible missing values due to classes not being in pred or true
        #print(vec_conf.shape)
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')
        #print(vec_conf.shape)

        # Reshape confusion in a matrix
        return vec_conf.reshape((num_classes, num_classes))


    else:

        # Ensure no negative classes
        if label_values[0] < 0:
            raise ValueError('Unsupported negative classes')

        # Get the data in [0,num_classes[
        label_map = np.zeros((label_values[-1] + 1,), dtype=np.int32)
        for k, v in enumerate(label_values):
            label_map[v] = k

        pred = label_map[pred]
        true = label_map[true]

        # Vectorized confusion
        vec_conf = np.bincount(true * num_classes + pred)

        # Add possible missing values due to classes not being in pred or true
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')

        # Reshape confusion in a matrix
        return vec_conf.reshape((num_classes, num_classes))


def classification_test(net, test_loader: KPConvLoader, config, num_votes=100): # used only by kpconv test
    print("validation size:", config.validation_size, "batch_num:", config.batch_num)
    ############
    # Initialize
    ############    
    # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    softmax = jt.nn.Softmax(1)

    # Number of classes including ignored labels
    nc_tot = test_loader.num_classes

    # Number of classes predicted by the model
    nc_model = config.num_classes

    # Initiate global prediction over test clouds
    test_probs = np.zeros((test_loader.num_models, nc_model))
    test_counts = np.zeros((test_loader.num_models, nc_model))
    print("probs shape:", test_probs.shape)

    t = [time.time()]
    mean_dt = np.zeros(1)
    last_display = time.time()
    while np.min(test_counts) < num_votes:

        # Run model on all test examples
        # ******************************

        # Initiate result containers
        probs = []
        targets = []
        obj_inds = []
        idx = 0
        # Start validation loop
        test_loader.prepare_batch_indices()
        for input_list in test_loader:
            # print("test", idx)
            idx += 1
            # batch = ModelNet40CustomBatch([input_list])
            # labels, model_inds = batch.labels, batch.model_inds
            L = (len(input_list) - 5) // 4
            labels = jt.array(input_list[4 * L + 1]).squeeze(0)
            model_inds = jt.array(input_list[4 * L + 4]).squeeze(0)
            # print(model_inds)
            # New time
            t = t[-1:]
            t += [time.time()]

            # Forward pass
            outputs = net(input_list)

            # Get probs and labels
            probs += [softmax(outputs).numpy()]
            targets += [labels.numpy()]
            obj_inds += [model_inds.numpy()]
            # print("probs: ", probs)
            # print("targets: ", targets)
            # print("obj_inds: ", obj_inds)

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 5.0:
                last_display = t[-1]
                message = 'Test vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(np.min(test_counts),
                                        100 * len(obj_inds) / config.validation_size,
                                        1000 * (mean_dt[0]),
                                        1000 * (mean_dt[1])))
        # Stack all validation predictions
        probs = np.vstack(probs)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)
        # print(obj_inds.shape)

        if np.any(test_loader.input_labels[obj_inds] != targets):
            raise ValueError('wrong object indices')

        # Compute incremental average (predictions are always ordered)
        test_counts[obj_inds] += 1
        # print(test_counts.shape)
        # print(test_counts)
        test_probs[obj_inds] += (probs - test_probs[obj_inds]) / (test_counts[obj_inds])

        # Save/Display temporary results
        # ******************************

        test_labels = np.array(test_loader.label_values)

        # Compute classification results
        C1 = fast_confusion(test_loader.input_labels,
                            np.argmax(test_probs, axis=1),
                            test_labels)

        ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        print('Test Accuracy = {:.1f}%'.format(ACC), flush=True)

    return

def hook():
    cfg = Modelnet40Config()
    net = KPCNN(cfg)
    chkp_path = "/mnt/disk1/chentuo/PointNet/KPConv-PyTorch/results/Log_2022-08-04_15-17-48/checkpoints/current_chkp.tar"
    checkpoint = torch.load(chkp_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer不控制
    print("Model and training state restored.")
    other_params = [v for k, v in net.named_parameters() if 'offset' not in k and 'running' not in k]
    optimizer = nn.SGD(other_params, lr = cfg.learning_rate, momentum = cfg.momentum, weight_decay=cfg.weight_decay)
    f = open('/mnt/disk1/chentuo/PointNet/PointCloudLib/networks/cls/data.txt', 'rb')
    data = pickle.load(f)
    f.close()
    batch = ModelNet40CustomBatch([data])
    hook = auto_diff.Hook("KPCNN")
    hook.hook_module(net)
    # hook = auto_diff.Hook("KPCNN_optim")
    # hook.hook_optimizer(optimizer)
    outputs = net(batch)
    loss = net.loss(outputs, batch.labels)
    acc = net.accuracy(outputs, batch.labels)
    # jt.display_memory_info()
    # Backward + optimize
    # idx = 0
    # for k, v in net.named_parameters():
    #     if 'offset' not in k and 'running' not in k and 'output_loss' not in k:
    #         print(" [", idx, "] ", k, " ###### ", jt.grad(loss, v))
    #         idx += 1
    optimizer.step(loss)
    outputs = net(batch)
    # jt.display_memory_info()
    exit(0)

if __name__ == '__main__':
    
    # hook()
    freeze_random_seed()
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--eval', action='store_true', default=False) # only used by kpconv
    parser.add_argument('--model', type=str, default='[pointnet]', metavar='N',
                        choices=['pointnet', 'pointnet2', 'pointcnn', 'dgcnn', 'pointconv', 'kpconv'],
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
    elif args.model == 'kpconv':
        cfg = Modelnet40Config()
        net = KPCNN(cfg)
    else:
        raise Exception("Not implemented")

    base_lr = args.lr
    if args.model != 'kpconv':
        optimizer = nn.SGD(net.parameters(), lr = base_lr, momentum = args.momentum)
    else:
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k and 'running' not in k]
        # idx = 0
        # for k, v in net.named_parameters():
        #     if 'offset' not in k and 'running' not in k:
        #         print(" [", idx, "] ", k, " ###### ", v.shape)
        #         idx += 1
        # exit(0)
        optimizer = nn.SGD(other_params, lr = cfg.learning_rate, momentum = cfg.momentum, weight_decay=cfg.weight_decay)
    lr_scheduler = LRScheduler(optimizer, base_lr)

    batch_size = args.batch_size
    n_points = args.num_points
    if args.model != 'kpconv':
        train_dataloader = ModelNet40(n_points=n_points, batch_size=batch_size, train=True, shuffle=True)
        val_dataloader = ModelNet40(n_points=n_points, batch_size=batch_size, train=False, shuffle=False)
    else:
        if not args.eval:
            train_dataloader = KPConvLoader(cfg, train=True, num_workers=0) # you can change num_workers to speed up
        cfg.validation_size = 250
        cfg.val_batch_num = 10
        val_dataloader = KPConvLoader(cfg, train=False, num_workers=4)
        #### load model ####
        if args.eval:
            chkp_path = "/mnt/disk1/chentuo/PointNet/PointCloudLib/checkpoints/kpconv/best_chkp.tar"
            checkpoint = jt.load(chkp_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # optimizer.load_state_dict({"defaults": checkpoint['optimizer_state_dict']['param_groups'][0]})
            net.eval()
            print("Model and training state restored.")
        # print(optimizer.state_dict())
        # print("#####")
        # print(checkpoint['optimizer_state_dict']['param_groups'][0])
        #####
    step = 0
    best_acc = 0
    
    
    for epoch in range(args.epochs):
        if args.model == 'kpconv':
            if not args.eval:
                train_kpconv(net, optimizer, epoch, train_dataloader)
                acc = evaluate_kpconv(net, epoch, val_dataloader)
                train_dataloader.prepare_batch_indices()
                val_dataloader.prepare_batch_indices()
                if epoch in cfg.lr_decays:
                    optimizer.lr *= cfg.lr_decays[epoch]
                if cfg.saving:
                    # Get current state dict
                    checkpoint_directory = 'checkpoints/kpconv'
                    save_dict = {'epoch': epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'saving_path': cfg.saving_path}
                    checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                    if acc > best_acc:
                        checkpoint_path = join(checkpoint_directory, 'best_chkp.tar')

                    # Save current state of the network (for restoring purposes)
                    jt.save(save_dict, checkpoint_path)

                    # Save checkpoints occasionally
                    if (epoch + 1) % cfg.checkpoint_gap == 0:
                        checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(epoch + 1))
                        jt.save(save_dict, checkpoint_path)
            else: # kpconv eval
                classification_test(net, val_dataloader, cfg)
                exit(0)
        else:
            lr_scheduler.step(len(train_dataloader) * batch_size)
            train(net, optimizer, epoch, train_dataloader, args)
            acc = evaluate(net, epoch, val_dataloader, args)

        best_acc = max(best_acc, acc)
        print(f'[Epoch {epoch}] val acc={acc:.4f}, best={best_acc:.4f}')
