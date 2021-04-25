#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 23:36:47 2019

@author: liwei
"""
import numpy as np
import math
import torch


def sign_func(phi, h):
    return torch.div(phi, torch.pow( torch.pow(phi, 2) + h*h, 0.5))

def BC_grad(phi, gridSize=0.01):
    ####################################################
    #######   call boundary condition          ########
    ###################################################
    #bc_north  = phi[-1,:]  
    #bc_south  = phi[ 0,:] 
    #bc_west = phi[:,-1]    
    #bc_east = phi[:, 0]   
    bc_north  = phi[0,:]  + phi[0,:] - phi[1,:]
    bc_south  = phi[-1,:] + phi[-1,:] - phi[-2,:]
    bc_west = phi[:,0]    + phi[:,0] - phi[:,1]
    bc_east = phi[:,-1]   + phi[:,-1] - phi[:,-2]

    bc_nw =  phi[ 0, 0] + phi[0, 0] - phi[1, 1]
    bc_sw =  phi[-1, 0] + phi[-1, 0] - phi[-2,1]
    bc_ne =  phi[ 0,-1] + phi[0,-1] - phi[1,-2]
    bc_se =  phi[-1,-1] + phi[-1,-1]- phi[-2,-2]
    ###################################################
    #print("Double check variables", bc_se.is_cuda)
    bc_west = bc_west.double().view(-1,1)
    bc_east = bc_east.double().view(-1,1)
    bc_south = bc_south.double()
    bc_north = bc_north.double()
    #print("Double check variables", bc_south.is_cuda)
    #print("Double check variables", bc_east.is_cuda)

    bc_sw = torch.Tensor([bc_sw]).type(torch.DoubleTensor).cuda()
    bc_nw = torch.Tensor([bc_nw]).type(torch.DoubleTensor).cuda()
    bc_se = torch.Tensor([bc_se]).type(torch.DoubleTensor).cuda()
    bc_ne = torch.Tensor([bc_ne]).type(torch.DoubleTensor).cuda()

    #print("Double check variables", bc_sw.is_cuda, bc_south.is_cuda, bc_se.is_cuda)
    bc_south = torch.cat((bc_sw,bc_south,bc_se), 0).view(1,-1)
    bc_north = torch.cat((bc_nw,bc_north,bc_ne), 0).view(1,-1)

    ###################################################

    #print("Double check variables", bc_west.is_cuda, phi.is_cuda, bc_east.is_cuda)
    phi_ext = torch.cat((bc_west,phi,bc_east), 1) #.cuda()
    #print("phi_ext shape now:",phi_ext.shape)
    #print("Double check variables", bc_north.is_cuda, phi_ext.is_cuda, bc_south.is_cuda)
    phi_ext = torch.cat((bc_north,phi_ext,bc_south), 0) #.cuda()
    #print("phi_ext shape now:",phi_ext.shape)
    return phi_ext

def BC_periodic(phi):
###################################################
    #######   call boundary condition          ########
    ###################################################
    bc_north  = phi[-1,:]  
    bc_south  = phi[ 0,:] 
    bc_west = phi[:,-1]    
    bc_east = phi[:, 0]   

    bc_nw =  phi[-1,-1]  
    bc_sw =  phi[ 0,-1]  
    bc_ne =  phi[-1, 0]  
    bc_se =  phi[ 0, 0]  
    ###################################################

    bc_west = bc_west.double().view(-1,1)
    bc_east = bc_east.double().view(-1,1)
    bc_south = bc_south.double()
    bc_north = bc_north.double()

    bc_sw = torch.Tensor([bc_sw]).type(torch.DoubleTensor)
    bc_nw = torch.Tensor([bc_nw]).type(torch.DoubleTensor)
    bc_se = torch.Tensor([bc_se]).type(torch.DoubleTensor)
    bc_ne = torch.Tensor([bc_ne]).type(torch.DoubleTensor)

    bc_south = torch.cat((bc_sw,bc_south,bc_se), 0).view(1,-1)
    bc_north = torch.cat((bc_nw,bc_north,bc_ne), 0).view(1,-1)

    ###################################################

    phi_ext = torch.cat((bc_west,phi,bc_east), 1)
    #print("phi_ext shape now:",phi_ext.shape)
    phi_ext = torch.cat((bc_north,phi_ext,bc_south), 0)
    #print("phi_ext shape now:",phi_ext.shape)
    return phi_ext

def BC_periodic_skip(phi): # this is wrong so it is not used 
###################################################
    #######   call boundary condition          ########
    ###################################################
    bc_north  = phi[-2,:]  
    bc_south  = phi[ 1,:] 
    bc_west = phi[:,-2]    
    bc_east = phi[:, 1]   

    bc_nw =  phi[-2,-2]  
    bc_sw =  phi[ 1,-2]  
    bc_ne =  phi[-2, 1]  
    bc_se =  phi[ 1, 1]  
    ###################################################

    bc_west = bc_west.double().view(-1,1)
    bc_east = bc_east.double().view(-1,1)
    bc_south = bc_south.double()
    bc_north = bc_north.double()

    bc_sw = torch.Tensor([bc_sw]).type(torch.DoubleTensor)
    bc_nw = torch.Tensor([bc_nw]).type(torch.DoubleTensor)
    bc_se = torch.Tensor([bc_se]).type(torch.DoubleTensor)
    bc_ne = torch.Tensor([bc_ne]).type(torch.DoubleTensor)

    bc_south = torch.cat((bc_sw,bc_south,bc_se), 0).view(1,-1)
    bc_north = torch.cat((bc_nw,bc_north,bc_ne), 0).view(1,-1)

    ###################################################

    phi_ext = torch.cat((bc_west,phi,bc_east), 1)
    #print("phi_ext shape now:",phi_ext.shape)
    phi_ext = torch.cat((bc_north,phi_ext,bc_south), 0)
    #print("phi_ext shape now:",phi_ext.shape)
    return phi_ext


def BC_dirichlet(phi, NORTH, SOUTH, WEST, EAST, gridSize=0.01):
    ####################################################
    #######   call boundary condition          ########
    ###################################################
    #bc_north  = phi[0,:]  + phi[0,:] - phi[1,:]
    #bc_south  = phi[-1,:] + phi[-1,:] - phi[-2,:]
    #bc_north  = phi[-1,:]  
    #bc_south  = phi[ 0,:] 
    #bc_west = phi[:,-1]    
    #bc_east = phi[:, 0]   
    res = phi.size(0)
    temp_y = -np.linspace(-1,1,res)
    WEST = 2+np.cos(np.pi*temp_y)
    WEST = torch.from_numpy(WEST).view(res)
   
    bc_north  = torch.zeros_like(phi[0,:]).add(NORTH)
    #print(bc_north.shape)
    #print(WEST.shape)
    bc_south  = torch.zeros_like(phi[-1,:]).add(SOUTH) 
    bc_west = torch.zeros_like(phi[:,0]).add(WEST) 
    bc_east = torch.zeros_like(phi[:,-1]).add(EAST) 

    bc_nw =  torch.zeros_like(phi[ 0, 0]).add(.5*(NORTH+3)) 
    bc_sw =  torch.zeros_like(phi[-1, 0]).add(.5*(SOUTH+3))
    bc_ne =  torch.zeros_like(phi[ 0,-1]).add(.5*(NORTH+EAST)) 
    bc_se =  torch.zeros_like(phi[-1,-1]).add(.5*(SOUTH+EAST))
    ###################################################

    bc_west = bc_west.double().view(-1,1)
    bc_east = bc_east.double().view(-1,1)
    bc_south = bc_south.double()
    bc_north = bc_north.double()

    bc_sw = torch.Tensor([bc_sw]).type(torch.DoubleTensor)
    bc_nw = torch.Tensor([bc_nw]).type(torch.DoubleTensor)
    bc_se = torch.Tensor([bc_se]).type(torch.DoubleTensor)
    bc_ne = torch.Tensor([bc_ne]).type(torch.DoubleTensor)

    bc_south = torch.cat((bc_sw,bc_south,bc_se), 0).view(1,-1)
    bc_north = torch.cat((bc_nw,bc_north,bc_ne), 0).view(1,-1)

    ###################################################

    phi_ext = torch.cat((bc_west,phi,bc_east), 1)
    #print("phi_ext shape now:",phi_ext.shape)
    phi_ext = torch.cat((bc_north,phi_ext,bc_south), 0)
    #print("phi_ext shape now:",phi_ext.shape)
    return phi_ext
