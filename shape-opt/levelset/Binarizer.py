#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 21:20:33 2019

@author: liwei
"""

# Python packages
import random, os, sys, datetime, time
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))
from Constants import *
import torch
import numpy as np
def torch_Heaviside(x, eps):
#    eps = torch.from_numpy(eps).double()
    eps = eps.expand(x.size()[0], x.size()[1]).double()
    return 0.5 + 1/np.pi * torch.atan2(x, eps)
def torch_Dirac_delta(x, eps):
    return 1/np.pi * eps/(x.pow(2.) + eps*eps)
# Inherit from Function
class Binarizer(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(self, input):
#        output = torch.Tensor( (input.detach().numpy()!=0).astype(int) )
        #output = torch.Tensor( (input.detach().numpy()>0.0).astype(int) )
         
        output = (input > torch.Tensor([0]).cuda()).double()*1 
        #output = (input > torch.Tensor([0])).double()*1 


        #output = torch_Heaviside(input.double(), torch.Tensor([1/55.]).double() )
        #diff_rate = torch_Heaviside(gradient_phi - 1., torch.Tensor([1./55.]).double())
        delta_func = torch_Dirac_delta(input.double(), 1/coeff)
        #delta_func = torch.ones(1).double() #torch_Dirac_delta(input.double(), 2/64.)
        self.save_for_backward(delta_func)
        return output.double()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        grad_input = output.mul(grad_output.double())
        return grad_input.double()

binarizer = Binarizer.apply
