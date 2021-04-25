
# Python packages
import numpy as np
from matplotlib import pyplot as plt

# Pytorch imports
import torch
import torch.nn.modules.loss as Loss 
import torch.nn.functional as F
#class Operators: 
   
sobel_dx = torch.Tensor( [[-1,  0,  1],
                          [-2,  0,  2],
                          [-1,  0,  1]] ) * (1/8)
sobel_dy = torch.Tensor( [[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]] ) * (1/8)

kernel_gaussian_blur = torch.Tensor( [[ 1, 2, 1],
                                      [ 2, 4, 2],
                                      [ 1, 2, 1]] ) * (1/16)

kernel_gaussian_blur2 = torch.Tensor( [[ 1, 4, 7, 4, 1],
                                       [ 4,16,26,16, 4],
                                       [ 7,26,41,26, 7],
                                       [ 4,16,26,16, 4],
                                       [ 1, 4, 7, 4, 1]] ) * (1/273)

central_dx = torch.Tensor( [[ 0,  0,  0],
                            [-1,  0,  1],
                            [ 0,  0,  0]] ) * (1/2)

central_dy = torch.Tensor( [[ 0, -1,  0],
                            [ 0,  0,  0],
                            [ 0,  1,  0]] ) * (1/2)
# 1st order upwind
upwPlus_dx = torch.Tensor( [[ 0,  0,  0],
                            [ 0, -1,  1],
                            [ 0,  0,  0]] )

upwMinus_dx = torch.Tensor( [[ 0,  0,  0],
                             [-1,  1,  0],
                             [ 0,  0,  0]] )

upwPlus_dy = torch.Tensor( [[ 0,  0,  0],
                            [ 0, -1,  0],
                            [ 0,  1,  0]] ) 

upwMinus_dy = torch.Tensor( [[ 0, -1,  0],
                             [ 0,  1,  0],
                             [ 0,  0,  0]] ) 

# 2nd order upwind
upw2ndPlus_dx = torch.Tensor( [[ 0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0], 
                               [ 0,  0, -3,  4, -1],
                               [ 0,  0,  0,  0,  0], 
                               [ 0,  0,  0,  0,  0]] ) * (1/2)

upw2ndMinus_dx = torch.Tensor( [[ 0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0], 
                                [ 1, -4,  3,  0,  0],
                                [ 0,  0,  0,  0,  0], 
                                [ 0,  0,  0,  0,  0]] ) * (1/2)

upw2ndPlus_dy = upw2ndPlus_dx.transpose(0,1)

upw2ndMinus_dy = upw2ndMinus_dx.transpose(0,1)

# might have a bug


laplacian = torch.Tensor( [[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]] ) 

#laplacian = (1-gamma)*laplacian + gamma*torch.Tensor( [[0.5, 0, 0.5],
#                           [0,  -2,   0],
#                           [0.5, 0, 0.5]] ) 
#
sobel_dx  = sobel_dx.view((1,1,3,3)).type(torch.DoubleTensor)
sobel_dy  = sobel_dy.view((1,1,3,3)).type(torch.DoubleTensor)
central_dx  = central_dx.view((1,1,3,3)).type(torch.DoubleTensor) 
central_dy  = central_dy.view((1,1,3,3)).type(torch.DoubleTensor)
upwPlus_dx  = upwPlus_dx.view((1,1,3,3)).type(torch.DoubleTensor)
upwPlus_dy  = upwPlus_dy.view((1,1,3,3)).type(torch.DoubleTensor)
upwMinus_dx  = upwMinus_dx.view((1,1,3,3)).type(torch.DoubleTensor)
upwMinus_dy  = upwMinus_dy.view((1,1,3,3)).type(torch.DoubleTensor)

kernel_gaussian_blur  = kernel_gaussian_blur.view((1,1,3,3)).type(torch.DoubleTensor)
kernel_gaussian_blur2  = kernel_gaussian_blur2.view((1,1,5,5)).type(torch.DoubleTensor)
laplacian  = laplacian.view((1,1,3,3)).type(torch.DoubleTensor)

upw2ndPlus_dx  = upw2ndPlus_dx.view((1,1,5,5)).type(torch.DoubleTensor)
upw2ndPlus_dy  = upw2ndPlus_dy.view((1,1,5,5)).type(torch.DoubleTensor)
upw2ndMinus_dx  = upw2ndMinus_dx.view((1,1,5,5)).type(torch.DoubleTensor)
upw2ndMinus_dy  = upw2ndMinus_dy.view((1,1,5,5)).type(torch.DoubleTensor)


def Heaviside(x, eps):
    return 0.5 + 1/np.pi * np.arctan2(x, eps)
def Dirac_delta(x, eps):
    return 1/np.pi * eps / (np.power(x,2) + eps*eps)

def torch_Heaviside(x, eps):
#    eps = torch.from_numpy(eps).double()
    eps = eps.expand(x.size()[0], x.size()[1])
    return 0.5 + 1/np.pi * torch.atan2(x, eps)
def torch_Dirac_delta(x, eps): # note that this is the so-called "Poisson kernel"
    return 1/np.pi * eps/(x.pow(2.) + eps*eps)

