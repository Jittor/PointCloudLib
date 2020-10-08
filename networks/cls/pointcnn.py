import numpy as np
import jittor as jt 
from jittor import nn
from jittor.contrib import concat 
from jittor import init 

from misc.layers import Dense_Conv1d, Dense_Conv2d, RandPointCNN

import time 


jt.flags.use_cuda = 1



# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
# Abbreviated PointCNN constructor.

AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e)


class PointCNNcls(nn.Module):
    def __init__(self, n_classes=40):
        super(PointCNNcls, self).__init__()
        self.pcnn1 = AbbPointCNN(3, 48, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(48, 96, 12, 2, 384),
            AbbPointCNN(96, 192, 16, 2, 128),
            AbbPointCNN(192, 384, 16, 3, 128),
        )

        self.fcn = nn.Sequential(
            Dense_Conv1d(384, 192),
            Dense_Conv1d(192, 128, drop_rate=0.5),
            Dense_Conv1d(128, n_classes, with_bn=False, activation=None)
        )

    
    def execute(self, x, normal=None):
        if normal is None:
            x = (x, x)
        else :
            x = (x, normal)
        
        x = self.pcnn1(x)
        x = self.pcnn2(x)[1] # grab features 
        
        x = x.permute(0, 2, 1) # b, dim, n
        logits = self.fcn(x) 
        logits = jt.mean(logits, dim=2)
        return logits


def main():
    x_input = init.invariant_uniform([16, 1024, 3], dtype='float')
    x_ = x_input
    x = (x_, x_input)
    model = PointCNN()
    y = model(x)
    _ = y.data  
    print (y.shape)

if __name__ == '__main__':
    main()
