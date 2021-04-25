# Python packages
import numpy as np

import matplotlib.pyplot as plt
# Pytorch imports
import torch
import torch.nn.modules.loss as Loss 
import torch.nn.functional as F
#lc:

import os, math, uuid, sys, random
from scipy import spatial
from skimage.util.shape import view_as_windows
from scipy import interpolate

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))

from data import utils
from levelset import Constants
#
# Helpers Import
#from Solver import DeepFlowSolver
#from Helper import printTensorAsImage,logMsg

class Velocity:
    class VELPACK:
        VEL  = 0
        COS  = 0
        SIN  = 0
        VISCOSITY = 1e-5

    def __init__(self, velx, vely):
        self.VELPACK.VEL  = np.sqrt(velx*velx + vely*vely)
        self.VELPACK.COS  = velx / self.VELPACK.VEL
        self.VELPACK.SIN  = vely / self.VELPACK.VEL

    def passTupple(self):
        return self.VELPACK
    
    def getVelVector(self):
        return np.array([self.VELPACK.VEL*self.VELPACK.COS,
                         self.VELPACK.VEL*self.VELPACK.SIN])

    def getVelMagnitude(self):
        return self.VELPACK.VEL

    def updateViscosity(self, visc):
        self.VELPACK.VISCOSITY = visc

binaryMask = np.array([ 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                                       ])


# Input:
#        binaryMask is a pytorch array with size [sizeX, sizeY]
#        pressureField is a pytorch array with size [sizeX, sizeY]
def calculateDragLift_phi(binaryMask, openFoamMask, deltaMask, normX, normY, gradient_abs, pressureField, upstreamVel, verbose=False, xdim=256):
    kernel_dx = torch.Tensor( [[-1,  0,  1],
                               [-2,  0,  2],
                               [-1,  0,  1]] ) * (1/8)
    kernel_dy = torch.Tensor( [[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]] ) * (1/8)
    #kernel_dx = torch.Tensor( [[-0,  0,  0],
    #                           [-1,  0,  1],
    #                           [-0,  0,  0]] ) * (1/2)
    #kernel_dy = torch.Tensor( [[-0, -1, -0],
    #                           [ 0,  0,  0],
    #                           [ 0,  1,  0]] ) * (1/2)
    #kernel_dy = torch.Tensor( [[ 1,  2,  1],
    #                           [ 0,  0,  0],
    #                           [-1, -2, -1]] ) * (1/4)
    kernel_dx  = kernel_dx.view((1,1,3,3)).type(torch.DoubleTensor)
    kernel_dy  = kernel_dy.view((1,1,3,3)).type(torch.DoubleTensor)
    
    #plt.figure()
    #plt.title("dx_layer*pressure in image space")
    ##plt.imshow(dx_layer.view(xdim,xdim).detach().numpy())
    #jacobiRatio = xdim/2. #pixels
    #plt.imshow(torch.mul(dx_layer, pressureField).view(xdim,xdim).detach().numpy()  / jacobiRatio)
    #plt.colorbar()
    #plt.figure()
    #plt.title("pressureField in image space")
    #plt.imshow(pressureField.detach().numpy())
    #plt.colorbar()
    
    if verbose:
        #printTensorAsImage(dx_layer[0][0], "dx", display=True)
        #printTensorAsImage(dy_layer[0][0], "dy", display=True)
        printTensorAsImage(pressureField, "pressure", display=True)
####
    print("xdim in force calc.:", xdim)
    jacobiRatio = xdim/2. #pixels
####

    #pressureField = pressureField.mul(1-openFoamMask.detach()) 
    #BINARIZE_ON=False
    #if BINARIZE_ON==True:
    #else:
    #force_x = torch.sum(torch.mul(-normX, pressureField).mul(deltaMask).mul(gradient_abs).mul(1-openFoamMask) )* (2./(xdim-1))**2
    #force_y = torch.sum(torch.mul(-normY, pressureField).mul(deltaMask).mul(gradient_abs).mul(1-openFoamMask) )* (2./(xdim-1))**2
    #force_x = torch.sum(torch.mul(-normX, pressureField.mul(1-binaryMask)).mul(deltaMask).mul(gradient_abs) )* (2./(xdim-1))**2
    #force_y = torch.sum(torch.mul(-normY, pressureField.mul(1-binaryMask)).mul(deltaMask).mul(gradient_abs) )* (2./(xdim-1))**2
    #force_x = torch.sum(torch.mul(-normX, pressureField).mul(deltaMask).mul(gradient_abs) )* (2./(xdim-1))**2 *2.
    #force_y = torch.sum(torch.mul(-normY, pressureField).mul(deltaMask).mul(gradient_abs) )* (2./(xdim-1))**2 *2.
    force_x = torch.sum(torch.mul(-normX, pressureField).mul(deltaMask).mul(gradient_abs) )* (2./(xdim-1))**2 
    force_y = torch.sum(torch.mul(-normY, pressureField).mul(deltaMask).mul(gradient_abs) )* (2./(xdim-1))**2 
    #drag   = force_x * upstreamVel.COS + force_y * upstreamVel.SIN # need to be careful!!!!
    #lift   = force_x * upstreamVel.SIN - force_y * upstreamVel.COS # need to be careful!!!!
    if verbose:
        drag.register_hook(logMsg)
        force_x.register_hook(logMsg)
        force_y.register_hook(logMsg)
        #dx_layer.register_hook(logMsg)
        #dy_layer.register_hook(logMsg)

    #drag   = drag/0.5/upstreamVel.VEL/upstreamVel.VEL # outward-normal is positive
    #force_x   = force_x/0.5/upstreamVel.VEL/upstreamVel.VEL # outward-normal is positive
    #force_y   = force_y/0.5/upstreamVel.VEL/upstreamVel.VEL # outward-normal is positive
    print("pressure forces in X & Y directions:", force_x.item(), force_y.item())
    return force_x, force_y

# Input:
#        pressureField is a pytorch array with size [sizeX, sizeY]
def calculateDragLift_visc_phi(binaryMask, openFoamMask, deltaMask, normX, normY, gradient_abs, velocityXField, velocityYField, upstreamVel, verbose=False, xdim=256):
    VISC_CORR = False
    #kernel_dx = torch.Tensor( [[-0,  0,  0],
    #                           [-1,  0,  1],
    #                           [-0,  0,  0]] ) * (1/2)
    #kernel_dy = torch.Tensor( [[-0, -1, -0],
    #                           [ 0,  0,  0],
    #                           [ 0,  1,  0]] ) * (1/2)
    kernel_dx = torch.Tensor( [[-1,  0,  1],
                               [-2,  0,  2],
                               [-1,  0,  1]] ) * (1/8)
    kernel_dy = torch.Tensor( [[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]] ) * (1/8)
    #kernel_dy = torch.Tensor( [[ 1,  2,  1],
    #                           [ 0,  0,  0],
    #                           [-1, -2, -1]] ) * (1/4)
    #kernel_dy = torch.Tensor( [[-1,  0,  1],
    #                           [-2,  0,  2],
    #                           [-1,  0,  1]] ) * (1/4)
    #kernel_dx = torch.Tensor( [[-1, -2, -1],
    #                           [ 0,  0,  0],
    #                           [ 1,  2,  1]] ) * (1/4)
    ##kernel_dy = torch.Tensor( [[ 1,  2,  1],
    ##                           [ 0,  0,  0],
    ##                           [-1, -2, -1]] ) * (1/4)
# lc: Sobel Operator --> normal vector
# lc: in the sampled-pixel space, down is y-positive, right is x-positive
    kernel_dx  = kernel_dx.view((1,1,3,3)).type(torch.DoubleTensor) 
    kernel_dy  = kernel_dy.view((1,1,3,3)).type(torch.DoubleTensor) 
    temp_idim  = velocityXField.size()[0] 
    temp_jdim  = velocityXField.size()[1]
    #kernel_dx = kernel_dx.double()
    #kernel_dy = kernel_dy.double() # make sure they are of the same type 
    #plt.figure()
    #plt.colorbar()
    #plt.figure()
    #plt.title("velocityX in the image space")
    #plt.imshow(velocityXField.detach().numpy())
    #plt.colorbar()
    #plt.figure()
    #plt.title("velocityY in the image space")
    #plt.imshow(velocityYField.detach().numpy())
    #plt.colorbar()
    #plt.show()
    if VISC_CORR: 
        uuField = velocityXField.mul(velocityXField)
        uvField = velocityXField.mul(velocityYField)
        vvField = velocityYField.mul(velocityYField)
        #print(uuField.size()) #1x1x res x res (resolution)?

    velocityXField = velocityXField.view(1,1,temp_idim,temp_jdim)
    velocityYField = velocityYField.view(1,1,temp_idim,temp_jdim)
    #print(velocityXField.size()) #1x1xres x res (resolution)?
    if VISC_CORR: 
        uuField = velocityXField.view(1,1,temp_idim,temp_jdim)
        uvField = velocityYField.view(1,1,temp_idim,temp_jdim)
        vvField = velocityYField.view(1,1,temp_idim,temp_jdim)
    
   

    #print("before omega calculation") 
    # lc: conv2d requires 4 dimensional inputs
    gridSize=2./(xdim-1)
    omega  =  F.conv2d(velocityYField, kernel_dx.cuda(), padding=1).double() / gridSize
    omega  = omega - F.conv2d(velocityXField, kernel_dy.cuda(), padding=1).double() / gridSize
    omega = omega.view(xdim,xdim)
    #omega = omega.mul(1-openFoamMask.detach()) 
    #omega = omega.mul(1-binaryMask.detach()) 
    #omega = interpolateInside(omega, [0])
    if False:
        dataCopy = omega.detach().numpy()
        dataCopy[np.where(openFoamMask==1)] = np.nan
        
        x = np.arange(0, dataCopy.shape[1])
        y = np.arange(0, dataCopy.shape[0])
        array = np.ma.masked_invalid(dataCopy)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]

        omega = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="nearest")
        omega = torch.from_numpy(omega).double()

    if False:
        plt.figure()
        plt.imshow(omega.detach().numpy()) #, levels)
        #plt.imshow(omega.detach().numpy(), vmin=-1e-4, vmax=1e-4) #, levels)
        plt.colorbar()
        plt.savefig("./figures/omega.png")

    #print(omega) 
    #torch.set_printoptions(threshold=10000)
    #print(omega.size()) #torch.Size([res, res]) res x res (resolution) after conv2d
    # normal-> = dx i-> + dy j->
    # p normal-> = p nx-> + p ny->  


    # v_t-> = v-> dot t->
    # v_t_x = vx tx
    # v_t_y = vy ty 
    #force_x = -torch.sum(torch.mul(-normY, omega).mul(deltaMask).mul(gradient_abs).mul(1-openFoamMask))*gridSize*gridSize
    #force_y =  torch.sum(torch.mul(-normX, omega).mul(deltaMask).mul(gradient_abs).mul(1-openFoamMask))*gridSize*gridSize
    #force_x = -torch.sum(torch.mul(-normY, omega).mul(deltaMask).mul(gradient_abs))*gridSize*gridSize
    #force_y =  torch.sum(torch.mul(-normX, omega).mul(deltaMask).mul(gradient_abs))*gridSize*gridSize
    force_x = -torch.sum(torch.mul(-normY, omega).mul(deltaMask).mul(gradient_abs))*gridSize*gridSize*2.
    force_y =  torch.sum(torch.mul(-normX, omega).mul(deltaMask).mul(gradient_abs))*gridSize*gridSize*2.
    
    if VISC_CORR: 
        corr_x = torch.sum(    ( torch.mul(-normX, uuField).add( torch.mul(-normY, uvField) ) ).mul(deltaMask).mul(gradient_abs))*gridSize*gridSize
        corr_y = torch.sum(    ( torch.mul(-normX, uvField).add( torch.mul(-normY, vvField) ) ).mul(deltaMask).mul(gradient_abs))*gridSize*gridSize
        sign = 1
        force_x = force_x + corr_x*sign
        force_y = force_y + corr_y*sign
    #drag   = force_x * upstreamVel.COS + force_y * upstreamVel.SIN # need to be careful

    #VISCOSITY = 0.5e-3
    #print("viscosity:", upstreamVel.VISCOSITY)
    force_x   = -force_x* upstreamVel.VISCOSITY #/0.5/upstreamVel.VEL/upstreamVel.VEL
    force_y   = -force_y* upstreamVel.VISCOSITY #/0.5/upstreamVel.VEL/upstreamVel.VEL
    print("viscous forces in X & Y directions:",force_x.item(), force_y.item())
    return force_x, force_y


