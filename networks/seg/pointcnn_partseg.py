import numpy as np
import jittor as jt 
from jittor import nn
from jittor.contrib import concat 
from jittor import init 
from misc.layers import Dense_Conv1d, Dense_Conv2d, RandPointCNN, RandPointCNN_Decoder
import time 
jt.flags.use_cuda = 1


#  (C_in : int, C_out : int, dims : int, K : int, D : int, P : int) -> None:

EncoderCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e)

DecoderCNN = lambda a, b, last_c, c, d, e: RandPointCNN_Decoder(a, b, last_c, 3, c, d, e) 

class PointCNN_partseg(nn.Module):
    def __init__(self, part_num=50):
        super(PointCNN_partseg, self).__init__()
        self.encoder_0 = EncoderCNN(3, 256, 8, 1, -1)
        self.encoder_1 = EncoderCNN(256, 256, 12, 1, 768)
        self.encoder_2 = EncoderCNN(256, 512, 16, 1, 384)
        self.encoder_3 = EncoderCNN(512, 1024, 16, 1, 128)


        self.decoder_0 = DecoderCNN(1024, 1024, 1024,  16, 1, 128)
        self.decoder_1 = DecoderCNN(1024, 512, 512, 16, 1, 385)
        self.decoder_2 = DecoderCNN(512, 256, 256, 12, 1, 768)
        self.decoder_3 = DecoderCNN(256, part_num, 256, 8, 1, 2048)
        # self.decoder_4 = DecoderCNN(256, part_num, 8, 4, 2048)
        


    def execute(self, x, normal=None):
        x = (x, x)        
        # jt.sync_all(True)
        # start_time = time.time()
        x_0 = self.encoder_0(x)
        x_1 = self.encoder_1(x_0)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_3 = self.decoder_0(x_3, x_3)
        x_2 = self.decoder_1(x_3, x_2)
        x_1 = self.decoder_2(x_2, x_1)
        x_0 = self.decoder_3(x_1, x_0)
        
        return x_0[1].permute(0, 2, 1)


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
