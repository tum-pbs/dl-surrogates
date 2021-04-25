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

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))

from data import utils

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
def calculateDragLift(binaryMask, openFoamMask, pressureField, upstreamVel, verbose=False, xdim=256):
    #kernel_dx = torch.Tensor( [[-1,  0,  1],
    #                           [-2,  0,  2],
    #                           [-1,  0,  1]] ) * (1/8)
    #kernel_dy = torch.Tensor( [[-1, -2, -1],
    #                           [ 0,  0,  0],
    #                           [ 1,  2,  1]] ) * (1/8)
    kernel_dx = torch.Tensor( [[-0,  0,  0],
                               [-1,  0,  1],
                               [-0,  0,  0]] ) * (1/2)
    kernel_dy = torch.Tensor( [[-0, -1, -0],
                               [ 0,  0,  0],
                               [ 0,  1,  0]] ) * (1/2)
    #kernel_dy = torch.Tensor( [[ 1,  2,  1],
    #                           [ 0,  0,  0],
    #                           [-1, -2, -1]] ) * (1/4)
    kernel_dx  = kernel_dx.view((1,1,3,3)).type(torch.DoubleTensor)
    kernel_dy  = kernel_dy.view((1,1,3,3)).type(torch.DoubleTensor)
    binaryMask = binaryMask.view((1,1,binaryMask.size()[0], binaryMask.size()[1]))
    
    dx_layer   = F.conv2d(binaryMask, kernel_dx, padding=1).double() * 2. #face normal vector should be 1.0 not 0.5
    dy_layer   = F.conv2d(binaryMask, kernel_dy, padding=1).double() * 2. #face normal vector should be 1.0 not 0.5
    #dx_layer   = F.conv2d(binaryMask, kernel_dx, padding=1).double() 
    #dy_layer   = F.conv2d(binaryMask, kernel_dy, padding=1).double() 
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
        printTensorAsImage(dx_layer[0][0], "dx", display=True)
        printTensorAsImage(dy_layer[0][0], "dy", display=True)
        printTensorAsImage(pressureField, "pressure", display=True)
####
    print("xdim in force calc.:", xdim)
    jacobiRatio = xdim/2. #pixels
####

    #force_x = torch.sum(torch.mul(dx_layer.view(xdim,xdim), pressureField.mul(1-openFoamMask.detach()) ) ) / jacobiRatio #.detach()
    #force_y = torch.sum(torch.mul(dy_layer.view(xdim,xdim), pressureField.mul(1-openFoamMask.detach()) ) ) / jacobiRatio #.detach()
    force_x = torch.sum(torch.mul(dx_layer.view(xdim,xdim), pressureField) ) / jacobiRatio
    force_y = torch.sum(torch.mul(dy_layer.view(xdim,xdim), pressureField) ) / jacobiRatio
    #drag   = force_x * upstreamVel.COS + force_y * upstreamVel.SIN # need to be careful!!!!
    #lift   = force_x * upstreamVel.SIN - force_y * upstreamVel.COS # need to be careful!!!!
    if verbose:
        drag.register_hook(logMsg)
        force_x.register_hook(logMsg)
        force_y.register_hook(logMsg)
        dx_layer.register_hook(logMsg)
        dy_layer.register_hook(logMsg)

    #drag   = drag/0.5/upstreamVel.VEL/upstreamVel.VEL # outward-normal is positive
    #force_x   = force_x/0.5/upstreamVel.VEL/upstreamVel.VEL # outward-normal is positive
    #force_y   = force_y/0.5/upstreamVel.VEL/upstreamVel.VEL # outward-normal is positive
    print("pressure forces in X & Y directions:", force_x.item(), force_y.item())
    return force_x, force_y

# Input:
#        binaryMask is a pytorch array with size [sizeX, sizeY]
#        pressureField is a pytorch array with size [sizeX, sizeY]
def calculateDragLift_visc(binaryMask, openFoamMask, velocityXField, velocityYField, upstreamVel, verbose=False):
    VISC_CORR = False
    kernel_dx = torch.Tensor( [[-0,  0,  0],
                               [-1,  0,  1],
                               [-0,  0,  0]] ) * (1/2)
    kernel_dy = torch.Tensor( [[-0, -1, -0],
                               [ 0,  0,  0],
                               [ 0,  1,  0]] ) * (1/2)
    #kernel_dx = torch.Tensor( [[-1,  0,  1],
    #                           [-2,  0,  2],
    #                           [-1,  0,  1]] ) * (1/8)
    #kernel_dy = torch.Tensor( [[-1, -2, -1],
    #                           [ 0,  0,  0],
    #                           [ 1,  2,  1]] ) * (1/8)
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
    binaryMask = binaryMask.view((1,1,binaryMask.size()[0], binaryMask.size()[1]))
    temp_idim  = velocityXField.size()[0] 
    temp_jdim  = velocityXField.size()[1]
    #kernel_dx = kernel_dx.double()
    #kernel_dy = kernel_dy.double() # make sure they are of the same type 
    #plt.figure()
    #plt.title("binaryMask in the image space")
    #plt.imshow(binaryMask.view(temp_idim, temp_jdim).detach().numpy())
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
    #print(binaryMask.size()) #1x1x res x res (resolution)?
    #print(velocityXField.size()) #1x1xres x res (resolution)?
    #print(binaryMask)  
    
    dx_layer   = F.conv2d(binaryMask, kernel_dx, padding=1).double() * 2. #face normal vector should be 1.0 not 0.5
    dy_layer   = F.conv2d(binaryMask, kernel_dy, padding=1).double() * 2. #face normal vector should be 1.0 not 0.5
    #dx_layer   = F.conv2d(binaryMask, kernel_dx, padding=1).double()  
    #dy_layer   = F.conv2d(binaryMask, kernel_dy, padding=1).double()  
    #utils.saveAsImage('velX.png', velocityXField.view(temp_idim,temp_jdim).detach().numpy()) # [2] binary mask for boundary
    #utils.saveAsImage('velY.png', velocityYField.view(temp_idim,temp_jdim).detach().numpy()) # [2] binary mask for boundary
    #utils.saveAsImage('binaryMask.png', binaryMask.view(temp_idim,temp_jdim).detach().numpy()) # [2] binary mask for boundary
    #utils.saveAsImage('dx.png', dx_layer.view(temp_idim,temp_jdim).detach().numpy()) # [2] binary mask for boundary
    #utils.saveAsImage('dy.png', dy_layer.view(temp_idim,temp_jdim).detach().numpy()) # [2] binary mask for boundary
   

    #print("before omega calculation") 
    # lc: conv2d requires 4 dimensional inputs
    omega  =  F.conv2d(velocityYField, kernel_dx, padding=1).double()
    omega  = omega - F.conv2d(velocityXField, kernel_dy, padding=1).double()
    omega = omega.view(temp_idim,temp_jdim)
    #omega = omega.mul(1-openFoamMask.detach()) 
    #omega = omega.mul(1-binaryMask.detach()) 
    if False:
        plt.figure()
        plt.imshow(omega.detach().numpy(), vmin=-1e-5, vmax=1e-5) #, levels)
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
    force_x = -torch.sum(torch.mul(dy_layer.view(temp_idim,temp_jdim), omega) )
    force_y =  torch.sum(torch.mul(dx_layer.view(temp_idim,temp_jdim), omega) )

    if VISC_CORR: 
        corr_x = torch.sum( torch.mul(dx_layer, uuField).add( torch.mul(dy_layer, uvField) ) )
        corr_y = torch.sum( torch.mul(dx_layer, uvField).add( torch.mul(dy_layer, vvField) ) )

        force_x = force_x - corr_x
        force_y = force_y - corr_y
    #drag   = force_x * upstreamVel.COS + force_y * upstreamVel.SIN # need to be careful
    #lift   = force_x * upstreamVel.SIN - force_y * upstreamVel.COS # need to be careful!!!!
    #VISCOSITY = 0.5e-3
    #print("viscosity:", upstreamVel.VISCOSITY)
    force_x   = -force_x* upstreamVel.VISCOSITY #/0.5/upstreamVel.VEL/upstreamVel.VEL
    force_y   = -force_y* upstreamVel.VISCOSITY #/0.5/upstreamVel.VEL/upstreamVel.VEL
    print("viscous forces in X & Y directions:",force_x.item(), force_y.item())
    return force_x, force_y


class DragLoss(Loss._Loss):
    def __init__(self, velx, vely, **kwargs):
        super(DragLoss, self).__init__()
        self.ps = DeepFlowSolver(velx, vely, logStates=kwargs['logStates'] if 'logStates' in kwargs else False)
        
        self.kwargs = kwargs
        self.kwargs['upstream'] = Velocity(velx, vely)
        
        self.solver    = kwargs['solver'] if 'solver' in kwargs else self.ps.pressureSolver
        self.verbose   = kwargs['verbose'] if 'verbose' in kwargs else False
        self.normalize = kwargs['normalize'] if 'normalize' in kwargs else True

    def _normalizePressure(self, pressureField):
        upstreamVelMagn = self.kwargs['upstream'].getVelMagnitude()
        npPressureField = pressureField.detach().numpy() 
        npPressureField = npPressureField / (0.5 * upstreamVelMagn * upstreamVelMagn) 
        return torch.Tensor( npPressureField ).double()

    def forward(self, **params):
        params        = {**params, **self.kwargs}
        #pressureField = self.solver(**params)
        #lc:
        pressureField, velocityXField, velocityYField = self.solver(**params)
        #lc: be careful about velocityYField in the physical space and sampled space!!!!
    
        #velocityYField = - velocityYField 

        #processedPF   = self._normalizePressure(pressureField) if self.normalize else pressureField
        #drag, _ = calculateDragLift(params['binaryMask'], processedPF, params['upstream'].passTupple(), self.verbose)
        drag, _ = calculateDragLift(params['binaryMask'], pressureField, params['upstream'].passTupple(), self.verbose)
        drag_visc, _ = calculateDragLift_visc(params['binaryMask'], velocityXField, velocityYField, params['upstream'].passTupple(), self.verbose)
        
        VISC = True
        if VISC: 
            drag += drag_visc 

        return drag
        #return drag*drag
