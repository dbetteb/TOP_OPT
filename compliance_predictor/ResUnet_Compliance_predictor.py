# source : https://github.com/hsinyilin19/ResNetVAE/blob/82a8bf16b1ee0bec95e53498bf0fddde4d4f2d28/modules.py#L54
#### one difference from the original Cond_Generator_ResNet152_VAE.py : z = embeddings + the volume fraction value

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models

class Residual_convolution_block(nn.Module):
    def __init__(self, kernel, padding, in_channels, out_channels):
        super(Residual_convolution_block, self).__init__()
        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=2, padding=padding, bias=False) 
#         self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel, stride=2, padding=padding, bias=False) or try this
        self.residual_block = nn.Sequential( 
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.001, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.001, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.identity_map = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, input):
        input = self.downsample(input)
        out = self.residual_block(input) + self.identity_map(input)
        return out
    
    
class First_residual_convolution_block(nn.Module):
    def __init__(self, kernel, padding, in_channels, out_channels):
        super(First_residual_convolution_block, self).__init__()
        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=2, padding=padding, bias=False)
        self.residual_block = nn.Sequential( 
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.001, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.identity_map = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, input):
        input = self.downsample(input)
        out = self.residual_block(input) + self.identity_map(input)
        return out
    
class Residual_Transpose_convolution_block(nn.Module):
    def __init__(self, kernel, padding, in_channels, out_channels):
        super(Residual_Transpose_convolution_block, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=2, padding=padding, bias=False)
        self.residual_block = nn.Sequential( 
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.001, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.001, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.identity_map = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, input):
        input = self.upsample(input)
        out = self.residual_block(input) + self.identity_map(input)
        return out
    
# class compliance_predictor(nn.Module):
#     def __init__(self, nc = 1, nf = 32):
#         super(compliance_predictor, self).__init__()
#         if (nf<2):
#             print('min value for nf is 2')
            
        
#         ## encoder ##
#         self.b1 = First_residual_convolution_block( kernel=3, padding=0, in_channels=nc, out_channels=nf)# 50x50xnf
        
#         self.convolution1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False)

#         self.b2 = Residual_convolution_block( kernel=4, padding=1, in_channels=nf, out_channels=nf*2)    # 25x25xnf*2
        
#         self.convolution2 = nn.Conv2d(in_channels=nf*2, out_channels=nf*2, kernel_size=3, stride=1, padding=1, bias=False)
        
#         self.b3 = Residual_convolution_block( kernel=3, padding=1, in_channels=nf*2, out_channels=nf*4)  # 13x13xnf*4
        
#         self.convolution3 = nn.Conv2d(in_channels=nf*4, out_channels=nf*4, kernel_size=3, stride=1, padding=1, bias=False)
        
#         self.b4 = Residual_convolution_block( kernel=3, padding=1, in_channels=nf*4, out_channels=nf*8)  # 7x7xnf*8
        
        
#         ## bridge ##
#         self.bridge = Residual_convolution_block( kernel=3, padding=1, in_channels=nf*8, out_channels=nf*16) # 4x4xnf*16
        
#         ## decoder ##
#         self.b6 = Residual_Transpose_convolution_block( kernel=3, padding=1, in_channels=nf*16, out_channels=nf*8) # 7x7xnf*8
#         self.b7 = Residual_Transpose_convolution_block( kernel=3, padding=1, in_channels=nf*8*2, out_channels=nf*4) # 13x13xnf*4
#         self.b8 = Residual_Transpose_convolution_block( kernel=3, padding=1, in_channels=nf*4*3, out_channels=nf*2) # 25x25xnf*2
#         self.b9 = Residual_Transpose_convolution_block( kernel=4, padding=1, in_channels=nf*2*3, out_channels=nf)   # 50x50xnf
#         self.b10 = Residual_Transpose_convolution_block( kernel=4, padding=1, in_channels=nf*3, out_channels=int(nf*0.5)) # 100x100xnf/2
        
#         self.last_layer = nn.Conv2d(in_channels=int(nf*0.5), out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)# 100x100x1
        
#         self.lastReLU = nn.LeakyReLU(negative_slope=0.001, inplace=True)

#     def forward(self, input):
#         out1 = self.b1(input)
#         out1_1 = self.convolution1(out1)
#         for i in range(10):
#             out1_1 = self.convolution1(out1_1)
#         out2 = self.b2(out1_1)
#         out2_2 = self.convolution2(out2)
#         for i in range(10):
#             out2_2 = self.convolution2(out2_2)
#         out3 = self.b3(out2_2)
#         out3_3 = self.convolution3(out3)
#         for i in range(10):
#             out3_3 = self.convolution3(out3_3)
#         out4 = self.b4(out3_3)
# #         print("Change activation function from sigmoid to something else, check values in compliance matrices for that!")
#         return self.lastReLU(self.last_layer(
#             self.b10(torch.cat( (self.b9(
#                 torch.cat( (self.b8(
#                     torch.cat( (self.b7(
#                         torch.cat( (self.b6(self.bridge(out4)), out4), 1)), out3, out3_3), 1)), out2, out2_2), 1)), out1, out1_1), 1)) ))       

class compliance_predictor(nn.Module):
    def __init__(self, nc = 1, nf = 32):
        super(compliance_predictor, self).__init__()
        if (nf<2):
            print('min value for nf is 2')
            
        
        ## encoder ##
        self.b1 = First_residual_convolution_block( kernel=3, padding=0, in_channels=nc, out_channels=nf)# 50x50xnf
        self.b2 = Residual_convolution_block( kernel=4, padding=1, in_channels=nf, out_channels=nf*2)    # 25x25xnf*2
        self.b3 = Residual_convolution_block( kernel=3, padding=1, in_channels=nf*2, out_channels=nf*4)  # 13x13xnf*4
        self.b4 = Residual_convolution_block( kernel=3, padding=1, in_channels=nf*4, out_channels=nf*8)  # 7x7xnf*8
        
        
        ## bridge ##
        self.bridge = Residual_convolution_block( kernel=3, padding=1, in_channels=nf*8, out_channels=nf*16) # 4x4xnf*16
        
        ## decoder ##
        self.b6 = Residual_Transpose_convolution_block( kernel=3, padding=1, in_channels=nf*16, out_channels=nf*8) # 7x7xnf*8
        self.b7 = Residual_Transpose_convolution_block( kernel=3, padding=1, in_channels=nf*8*2, out_channels=nf*4) # 13x13xnf*4
        self.b8 = Residual_Transpose_convolution_block( kernel=3, padding=1, in_channels=nf*4*2, out_channels=nf*2) # 25x25xnf*2
        self.b9 = Residual_Transpose_convolution_block( kernel=4, padding=1, in_channels=nf*2*2, out_channels=nf)   # 50x50xnf
        self.b10 = Residual_Transpose_convolution_block( kernel=4, padding=1, in_channels=nf*2, out_channels=int(nf*0.5)) # 100x100xnf/2
        
        self.last_layer = nn.Conv2d(in_channels=int(nf*0.5), out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)# 100x100x1
        
        self.lastReLU = nn.LeakyReLU(negative_slope=0.001, inplace=True)

    def forward(self, input):
        out1 = self.b1(input)
        out2 = self.b2(out1)
        out3 = self.b3(out2)
        out4 = self.b4(out3)
#         print("Change activation function from sigmoid to something else, check values in compliance matrices for that!")
        return self.lastReLU(self.last_layer(self.b10(torch.cat( (self.b9(torch.cat( (self.b8(torch.cat( (self.b7(torch.cat( (self.b6(self.bridge(out4)), out4), 1)), out3), 1)), out2), 1)), out1), 1)) ))