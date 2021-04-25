
# Python packages
import random, json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import skfmm
import math  

# Pytorch imports
# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.modules.loss as Loss 
import torch.nn.functional as F
#from torchviz import make_dot

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))

# Helpers Import
from Operators import *
from Constants import *
from Binarizer import binarizer
from BoundaryConditions import *
## Random Seed ##


SEED = random.randint(0,2**32-1)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)

def satisfy_area_constraint(test_phi, gridSize, refArea, AREA_LOSS_PLOT=False, BINARIZE=False):
    res = test_phi.size(0)
    init_phi = test_phi
    shift = torch.Tensor([[0.0]]) 
    shift = Variable(shift.cuda(), requires_grad=True)
    #area_optimizer = optim.SGD([shift], lr=0.01, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
    area_optimizer = optim.Adam([shift], lr=1e-6, betas=(0.9, 0.999), eps=1e-06, weight_decay=0, amsgrad=False)

    print("Reference Area is:", refArea)    
    if AREA_LOSS_PLOT:
        time = []
        history = []
    N_SUB_AREA = 1500 #1000
    for isub in range(N_SUB_AREA):
        test_phi = test_phi.add(shift.expand(res,res))
        #test_binaryMask = 1-torch.sigmoid(test_phi.mul(coeff))
        test_binaryMask = calc_binaryMask(test_phi, BIN=BINARIZE)
        #binaryMask = calc_binaryMask(False)
        area = torch.sum(test_binaryMask)*gridSize**2
        area_loss = (area - refArea).pow(2.).pow(.5)
        area_loss.backward()
        if AREA_LOSS_PLOT:
            time.append(isub)
            history.append(area_loss.item())
            X, Y = np.meshgrid(np.linspace(-1,1,res), np.linspace(-1,1,res))
            print(isub, shift)
        area_optimizer.step()
        area_optimizer.zero_grad()
        if AREA_LOSS_PLOT and isub%10==0:
            plt.figure()
            plt.imshow(test_binaryMask.detach().numpy())
            #plt.imshow(test_phi.detach().numpy())
            plt.savefig("isub_"+str(isub)+".png")
            plt.close()
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, test_phi.detach().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_zlim3d(bottom=None, top=None, emit=True, auto=False, zmin=-0.45, zmax=1.25)
            fig.savefig("isub_3d_sdf_"+str(isub)+".png")
            #plt.show()
            plt.close()
        
        if(area_loss.item()<=1e-3):
            print("Area constraint satisfied after", isub, "iterations. Area diff=", area_loss.item())   
            output = test_phi
            break
    if isub >= N_SUB_AREA-1:
        print("Area constraint might not be satisfied. Area diff=", area_loss.item())    
    output = test_phi
    if AREA_LOSS_PLOT:
        plt.figure()
        plt.plot(time, history)
        plt.savefig("area_loss_history.png")
        #plt.show()
        plt.close()
        print("Will return shift:", output)

    return output - init_phi


def calc_gradient(phi, gridSize, UPWIND_SECOND_ORDER=True):
    phiGhost = BC_grad(phi, gridSize)
    #phiGhost = BC_periodic(phi)
    res = phi.size(0)

    gradientX = F.conv2d(phiGhost.view(1,1,res+2,res+2), central_dx.cuda(), padding=0).double()/gridSize  #face normal vector should be 1.0 not 0.5
    gradientY = F.conv2d(phiGhost.view(1,1,res+2,res+2), central_dy.cuda(), padding=0).double()/gridSize  #face normal vector  

    gradient_abs = (torch.mul(gradientX, gradientX)+torch.mul(gradientY, gradientY)).pow(0.5)
    eps = 1e-18
    normX = gradientX.div(gradient_abs + eps)
    normY = gradientY.div(gradient_abs + eps)
    lap_phi    = F.conv2d(phiGhost.view(1,1,res+2,res+2), laplacian.cuda(), padding=0).double()  /gridSize/gridSize    
    phi_mean    = F.conv2d(phiGhost.view(1,1,res+2,res+2), kernel_gaussian_blur.cuda(), padding=0).double()  

    k_curvature = F.conv2d(normX.view(1,1,res,res), central_dx.cuda(), padding=1).double()/gridSize + F.conv2d(normY.view(1,1,res,res), central_dy.cuda(), padding=1).double()/gridSize
    # note that this version of k_curvature is calculated using the central differencing!!!
    # we may want to do an upwind-like or stabler version suggested by Sethian!!! .... 28 Feb. 2020
    if UPWIND_SECOND_ORDER:
        phiGhost = BC_grad(phiGhost, gridSize)
        #phiGhost = BC_periodic(phiGhost)
        gradPlusX = F.conv2d(phiGhost.view(1,1,res+4,res+4), upw2ndPlus_dx.cuda(), padding=0).double()/gridSize  #face normal vector should be 1.0 not 0.5
        gradPlusY = F.conv2d(phiGhost.view(1,1,res+4,res+4), upw2ndPlus_dy.cuda(), padding=0).double()/gridSize  #face normal vector  

        gradMinusX = F.conv2d(phiGhost.view(1,1,res+4,res+4), upw2ndMinus_dx.cuda(), padding=0).double()/gridSize  #face normal vector should be 1.0 not 0.5
        gradMinusY = F.conv2d(phiGhost.view(1,1,res+4,res+4), upw2ndMinus_dy.cuda(), padding=0).double()/gridSize  #face normal vector  
    else:
        #phiGhost = BC_grad(phiGhost, gridSize)
        #phiGhost = BC_periodic(phiGhost)
        gradPlusX = F.conv2d(phiGhost.view(1,1,res+2,res+2), upwPlus_dx.cuda(), padding=0).double()/gridSize  #face normal vector should be 1.0 not 0.5
        gradPlusY = F.conv2d(phiGhost.view(1,1,res+2,res+2), upwPlus_dy.cuda(), padding=0).double()/gridSize  #face normal vector  

        gradMinusX = F.conv2d(phiGhost.view(1,1,res+2,res+2), upwMinus_dx.cuda(), padding=0).double()/gridSize  #face normal vector should be 1.0 not 0.5
        gradMinusY = F.conv2d(phiGhost.view(1,1,res+2,res+2), upwMinus_dy.cuda(), padding=0).double()/gridSize  #face normal vector  



    lap_phi     = lap_phi.view(res,res)
    phi_mean     = phi_mean.view(res,res)
    k_curvature = k_curvature.view(res,res)
    gradPlusX   = gradPlusX.view(res,res)
    gradPlusY   = gradPlusY.view(res,res)
    gradMinusX  = gradMinusX.view(res,res)
    gradMinusY  = gradMinusY.view(res,res)
    gradientX   = gradientX.view(res,res)
    gradientY   = gradientY.view(res,res)
    gradient_abs   = gradient_abs.view(res,res)
    normX = normX.view(res,res)
    normY = normY.view(res,res)
  
    meanMask = calc_binaryMask(phi_mean, BIN=True)

    return gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, meanMask



def calc_rhs(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize):
    res = u.size(0)
    k1 =         - ( torch.max(u,torch.zeros_like(u)).mul(gradMinusX) + torch.min(u,torch.zeros_like(u)).mul(gradPlusX) )
    k1 = k1.add( - ( torch.max(v,torch.zeros_like(v)).mul(gradMinusY) + torch.min(v,torch.zeros_like(v)).mul(gradPlusY) ) )
    ## Convection term Abs version:
    #k1 = -phi.grad.mul( gradient_abs ) #this works
    #LCM_TERM = False
    #if LCM_TERM:
    #    k1 = k1.add( mu_coeff_grad*(lap_phi - k_curvature)  )
    #else: 
    #    diff_rate = torch_Heaviside(gradient_abs - 1., torch.Tensor([1./55.]).double())
    #    normal_x = diff_rate.mul(gradientX)
    #    normal_y = diff_rate.mul(gradientY)
    #    div_normal = F.conv2d(normal_x.view(1,1,res,res), central_dx, padding=1).double() + F.conv2d(normal_y.view(1,1,res,res), central_dy, padding=1).double()
    #    div_normal = div_normal.view(res,res)/gridSize
    #    k1 = k1.add( mu_coeff_grad*div_normal  )
    #    # it seems there is no benefit to have a private rhs var.
    return k1

def calc_rhs_sethian(f_func, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, meanMask, gridSize):
    res = f_func.size(0)
    # in x-direction  
    # in y-direction
    nablaPlus  = torch.max(gradMinusX, torch.zeros_like(gradMinusX)).pow(2.) + torch.min(gradPlusX, torch.zeros_like(gradPlusX)).pow(2.)  \
                +torch.max(gradMinusY, torch.zeros_like(gradMinusY)).pow(2.) + torch.min(gradPlusY, torch.zeros_like(gradPlusY)).pow(2.)    
    # in x-direction
    # in y-direction
    nablaMinus = torch.max(gradPlusX, torch.zeros_like(gradPlusX)).pow(2.) + torch.min(gradMinusX, torch.zeros_like(gradMinusX)).pow(2.)  \
                +torch.max(gradPlusY, torch.zeros_like(gradPlusY)).pow(2.) + torch.min(gradMinusY, torch.zeros_like(gradMinusY)).pow(2.)    

    nablaPlus  = nablaPlus.pow(0.5)
    nablaMinus = nablaMinus.pow(0.5) 
#
    if KAPPA_SMOOTH=="SMOOTH_NAIVE":

    # Naive one / Naive modified:   
        f_func = f_func.add( -mu_coeff2*torch.abs(k_curvature) )

    elif KAPPA_SMOOTH=="SMOOTH_KMAX":

        k_curvature_max = torch.Tensor([[1./(2*gridSize)]])
        k_curvature_max = k_curvature_max.expand(res,res)
        #k_curvature_min = -k_curvature_max
        #f_func = f_func.add( -mu_coeff2*(torch.max(torch.abs(k_curvature) - k_curvature_max, torch.zeros_like(k_curvature))) )   # to keep the convex surface only.... Malladi & Sethian (1996).
        f_func = f_func.add( -mu_coeff2*(torch.max(k_curvature - k_curvature_max, torch.zeros_like(k_curvature))) )   # to keep the convex surface only.... Malladi & Sethian (1996).


    elif KAPPA_SMOOTH=="SMOOTH_CONVEX":
    # Sethian convex:
        f_func = f_func.add( -mu_coeff2*torch.max(k_curvature, torch.zeros_like(k_curvature)) )  # to keep the convex surface only.... Malladi & Sethian (1996).
    # Sethian concave:
    #f_func = f_func.add( -mu_coeff2*torch.min(k_curvature, torch.zeros_like(k_curvature)) )  # to keep the convex surface only.... Malladi & Sethian (1996).
    
    # new idea:
    #f_func = f_func.add( -mu_coeff2*torch.max(  (k_curvature - k_curvature_max), torch.zeros_like(k_curvature)) )  # to keep the convex surface only.... Malladi & Sethian (1996).
    elif KAPPA_SMOOTH=="SMOOTH_MS":
        # Malladi & Sethian Fmin/max:
        f_func = f_func.add( -mu_coeff2*torch.max(k_curvature, torch.zeros_like(k_curvature)).mul(1-meanMask) )  # to keep the convex surface only.... Malladi & Sethian (1996).
        f_func = f_func.add( -mu_coeff2*torch.min(k_curvature, torch.zeros_like(k_curvature)).mul(meanMask) )  # to keep the convex surface only.... Malladi & Sethian (1996).


    # new idea:
    #f_func = f_func.add( -mu_coeff2*torch.max(  torch.min(k_curvature, k_curvature_max), torch.zeros_like(k_curvature)).mul(1-meanMask) )  # to keep the convex surface only.... Malladi & Sethian (1996).
    #f_func = f_func.add( -mu_coeff2*torch.min(  torch.max(k_curvature, k_curvature_min), torch.zeros_like(k_curvature)).mul(meanMask) )  # to keep the convex surface only.... Malladi & Sethian (1996).
    #f_func = f_func.add( -mu_coeff2*torch.max(  (k_curvature - k_curvature_max), torch.zeros_like(k_curvature)).mul(1-meanMask) )  # to keep the convex surface only.... Malladi & Sethian (1996).
    #f_func = f_func.add( -mu_coeff2*torch.min(  (k_curvature - k_curvature_min), torch.zeros_like(k_curvature)).mul(meanMask) )  # to keep the convex surface only.... Malladi & Sethian (1996).

    #f_func = f_func.add( -mu_coeff2*torch.max(  (k_curvature - k_curvature_max), torch.zeros_like(k_curvature)) )  # to keep the convex surface only.... Malladi & Sethian (1996).
    k1 =         - ( torch.max(f_func,torch.zeros_like(f_func)).mul(nablaPlus) + torch.min(f_func,torch.zeros_like(f_func)).mul(nablaMinus) )
    
    if LCM_TERM:
        k1 = k1.add( mu_coeff_grad*(lap_phi - k_curvature)  )

    


    return k1

def calc_rhs_central(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize):
    res = u.size(0)
    k1 = -u.mul(gradientX) 
    k1 = k1.add( -v.mul(gradientY)  )


    #k1 = k1.add( mu_coeff2*lap_phi  )

    diff_rate = torch_Heaviside(gradient_abs - 1., torch.Tensor([1./55.]).double())
    normal_x = diff_rate.mul(gradientX)
    normal_y = diff_rate.mul(gradientY)
    div_normal = F.conv2d(normal_x.view(1,1,res,res), central_dx, padding=1).double() + F.conv2d(normal_y.view(1,1,res,res), central_dy, padding=1).double()
    div_normal = div_normal.view(res,res)/gridSize
    k1 = k1.add( mu_coeff_grad*div_normal  )
    # it seems there is no benefit to have a private rhs var.
    return k1


def calc_binaryMask(phi, BIN=False):
    if BIN:
        binaryMask = 1 - binarizer(phi)
    else:
        # sigmoid or so-called Fermi-Dirac function
        # eps_sigmoid = 2*gridSize is chosen according to Zahedi & Tornberg, JCP vol. 229, 2199-2219 (2010).
        # coeff = 1/eps_sigmoid
        # sigmoid(x) = 1/(1 + exp(-x))
        # Fermi-Dirac_eps(x) = 1/(1 + exp(-x/eps))
        binaryMask = 1-torch.sigmoid(phi*coeff) 
        #binaryMask = 1-torch_Heaviside(phi, torch.Tensor([1./coeff]).double()) # arctan2
    return binaryMask

def calc_deltaMask(phi, BIN=False, OFFSET=0):
    if False: #BIN:
        #deltaMask = 1 - binarizer(phi)


        print("not implemented yet")
    else:
        # sigmoid or so-called Fermi-Dirac function
        # eps_sigmoid = 2*gridSize is chosen according to Zahedi & Tornberg, JCP vol. 229, 2199-2219 (2010).
        # coeff = 1/eps_sigmoid
        # sigmoid(x) = 1/(1 + exp(-x))
        # Fermi-Dirac_eps(x) = 1/(1 + exp(-x/eps))
        deltaMask = (1-torch.sigmoid(phi*coeff)).mul(torch.sigmoid(phi*coeff))*coeff

        #deltaMask = (1-torch.sigmoid((phi-OFFSET)*coeff)).mul(torch.sigmoid((phi-OFFSET)*coeff))*coeff
        #deltaMask = torch_Dirac_delta(phi, 1./coeff)
    return deltaMask

def calc_deltaMask_eps(phi, eps_value):
    copyphi = np.copy(phi.detach().numpy())
    #cond_1 = np.abs(copyphi)<eps_value
    cond_2 = np.abs(copyphi)>= eps_value
    
    deltaMask = ( 1+np.cos(np.pi*copyphi/eps_value) )/2./eps_value
    deltaMask[cond_2] = 0
    return torch.from_numpy(deltaMask).double()


def calc_deltaMask_eps2(phi, eps_value):
    copyphi = np.copy(phi.detach().numpy())
    #cond_1 = np.abs(copyphi)<eps_value
    cond_2 = np.abs(copyphi)>= eps_value
    
    deltaMask = ( 1 - np.abs(copyphi/eps_value) )/eps_value
    deltaMask[cond_2] = 0
    return torch.from_numpy(deltaMask).double()

def calc_s_function(phi, eps_s_function=1e-5):
    return phi.div( (phi.pow(2.) + eps_s_function**2).pow(0.5) )

def reinit_rhs(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize, s_function):
    # define u and v outside this subroutine
    # source term is S(phi_0)
    # reuse the existing upwind scheme
    res = u.size(0)
    k1 =         - ( torch.max(u,torch.zeros_like(u)).mul(gradMinusX) + torch.min(u,torch.zeros_like(u)).mul(gradPlusX) )
    k1 = k1.add( - ( torch.max(v,torch.zeros_like(v)).mul(gradMinusY) + torch.min(v,torch.zeros_like(v)).mul(gradPlusY) ) )
    k1 = k1.add( s_function)


    return k1

def reinit_rhs_sethian(f_func, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize, s_function):
    res = f_func.size(0)
    # in x-direction  
    # in y-direction
    nablaPlus  = torch.max(gradMinusX, torch.zeros_like(gradMinusX)).pow(2.) + torch.min(gradPlusX, torch.zeros_like(gradPlusX)).pow(2.)  \
                +torch.max(gradMinusY, torch.zeros_like(gradMinusY)).pow(2.) + torch.min(gradPlusY, torch.zeros_like(gradPlusY)).pow(2.)    
    # in x-direction
    # in y-direction
    nablaMinus = torch.max(gradPlusX, torch.zeros_like(gradPlusX)).pow(2.) + torch.min(gradMinusX, torch.zeros_like(gradMinusX)).pow(2.)  \
                +torch.max(gradPlusY, torch.zeros_like(gradPlusY)).pow(2.) + torch.min(gradMinusY, torch.zeros_like(gradMinusY)).pow(2.)    

    nablaPlus  = nablaPlus.pow(0.5)
    nablaMinus = nablaMinus.pow(0.5) 
   

    k1 =         - ( torch.max(f_func,torch.zeros_like(f_func)).mul(nablaPlus) + torch.min(f_func,torch.zeros_like(f_func)).mul(nablaMinus) )
    k1 = k1.add( s_function)

    


    return k1


def calc_gradient_pressure(phi, gridSize):
    phiGhost = BC_dirichlet(phi, 3,3,10,0)
    res = phi.size(0)
    lap_phi    = F.conv2d(phiGhost.view(1,1,res+2,res+2), laplacian, padding=0).double()  /gridSize/gridSize    
    lap_phi     = lap_phi.view(res,res)
    return lap_phi
