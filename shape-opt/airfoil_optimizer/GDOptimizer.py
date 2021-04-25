import torch
import numpy as np
from Helper      import printTensorAsImage, logMsg

_EPSILON = 1e-2

class GDOptimizer:
    class TargetParam:
        ALL = ''

    def __init__(self, model, lr, **kwargs):
        self.model          = model
        self.lr             = lr
        self.verbose        = kwargs['verbose'] if 'verbose' in kwargs else False
        self.targetParam    = kwargs['targetParam'] if 'targetParam' in kwargs else self.TargetParam.ALL
        self.centralizeFlag = kwargs['centralize'] if 'centralize' in kwargs else False
        self.maxLR          = kwargs['maxLearningRate'] if 'maxLearningRate' in kwargs else 1e+9
        self.minLR          = kwargs['minLearningRate'] if 'minLearningRate' in kwargs else 1e-5
        self.normGrad       = kwargs['normalizeGradient'] if 'normalizeGradient' in kwargs else False

        if 'takeAverageGrad' in kwargs and isinstance(kwargs['takeAverageGrad'], int):
            self.averageGrads       = True
            self.numOfHistGrad      = kwargs['takeAverageGrad']
            self.gradHistory        = {}
        else:
            self.averageGrads       = False
        
        if 'areaConstraint' in kwargs and isinstance(kwargs['areaConstraint'], (int, float)):
            self.areaConstraintFlag = True
            self.areaConstraint     = kwargs['areaConstraint'] 
        else:
            self.areaConstraintFlag = False

        if 'crossSection' in kwargs and isinstance(kwargs['crossSection'], (int, float)):
            self.crossSectionFlag = True
            self.crossSection = kwargs['crossSection']
        else:
            self.crossSectionFlag = False

        self.integrityConstraintFlag    = kwargs['integrityConstraint'] if 'integrityConstraint' in kwargs else False

    def setLearningRate(self, lr):
        if self.minLR < lr < self.maxLR:
            self.lr = lr
    
    def centralize_shape(self, name):
        params  = self.model._parameters[name].detach().numpy()
        shift_amount = (0.5 - params[0][self.model.numOfParams//2]) 
        params += shift_amount
        self.model._parameters[name].data = torch.Tensor(params)

    def satisfy_area_constraint(self, name):
        params  = self.model._parameters[name].detach().numpy()
        delta   = (self.areaConstraint - np.sum(params)) / self.model.numOfParams
        params += delta
        self.model._parameters[name].data = torch.Tensor(params)

    def satisfy_cross_section_constraint(self, name):
        params    = self.model._parameters[name].detach().numpy()
        maxWidth  = self.model.maxWidth; minWidth  = self.model.minWidth
        widths    = params * (maxWidth - minWidth) + minWidth
        params    = np.where(widths > self.crossSection, (self.crossSection - minWidth) / (maxWidth - minWidth) - _EPSILON, params )
        widths    = params * (maxWidth - minWidth) + minWidth
        widths    = widths * (self.crossSection / np.max(widths))
        params    = (widths - minWidth) / (maxWidth - minWidth)
        self.model._parameters[name].data = torch.Tensor(params)

    def satisfy_integrity_constraint(self, name):
        params  = self.model._parameters[name].detach().numpy()
        
        self.model.forward(suppress = True)
        rectBlocks = (self.model.rectangleBlocks.detach().numpy()!=0).astype(int)

        upper_indicies   = np.where(rectBlocks[0] == 1)        
        for i in range(1, rectBlocks.shape[0]):
            current_indicies = np.where(rectBlocks[i] == 1)

            # check relative to upper rectangle, case lower rectangle is on the left side
            if( current_indicies[0][-1] < upper_indicies[0][0] ):
                shift_amount = upper_indicies[0][0] - current_indicies[0][-1] + 5
                upper_indicies = current_indicies + shift_amount
                params[0][i] = ( (upper_indicies[0][0] + upper_indicies[0][-1] ) // 2 - self.model.leftMost ) / (self.model.rightMost - self.model.leftMost)

            # check relative to upper rectangle, case lower rectangle is on the right side
            elif( current_indicies[0][0] > upper_indicies[0][-1] ):
                shift_amount = current_indicies[0][0] - upper_indicies[0][-1] + 5
                upper_indicies = current_indicies - shift_amount
                params[0][i] = ( (upper_indicies[0][0] + upper_indicies[0][-1] ) // 2 - self.model.leftMost ) / (self.model.rightMost - self.model.leftMost)

            # integritiy is already satisfied
            else:
                upper_indicies = current_indicies

        self.model._parameters[name].data = torch.Tensor(params)

    def normalize_gradient(self, grad_data):
        copy_grad_data = grad_data.clone()
        return copy_grad_data / copy_grad_data.max()

    def get_average_gradient(self, name, grad_data):
        if name not in self.gradHistory:
            self.gradHistory[name] = []
            self.gradHistory[name].append(grad_data.clone())
            self.gradHistory[name] *= self.numOfHistGrad
        else:
            self.gradHistory[name].append(grad_data.clone())
            self.gradHistory[name] = self.gradHistory[name][1:]
        
        return sum(self.gradHistory[name])/len(self.gradHistory[name])

    def step(self):
        # Verbose Part
        if self.verbose:
            logMsg(10*"-")

        for name, param in self.model.named_parameters():
            # Verbose Part
            if self.targetParam in name: 
                if self.normGrad:
                    param.grad.data = self.normalize_gradient(param.grad.data)
                
                if self.verbose:
                    if param.grad is None:
                        logMsg("Parameter '{}' = {}".format(name, param.data))
                        logMsg("Grad is None")
                    else:
                        logMsg("Parameter '{}' = {}".format(name, param.data))
                        logMsg("Gradient of parameter '{}' = {}".format(name,param.grad.data))
                    logMsg("Updating param '{}'.".format(name))
                
                # Update procedure
                if self.averageGrads:
                    param.data.add_(-self.lr, self.get_average_gradient(name, param.grad.data))
                else:
                    param.data.add_(-self.lr, param.grad.data)  
                
                # Satisfy Constant Cross Section Contraint
                if self.crossSectionFlag: self.satisfy_cross_section_constraint(name)
                # Satisfy Area Constraint
                if self.areaConstraintFlag: self.satisfy_area_constraint(name)
                # Satisfy Integrity Constraint
                if self.integrityConstraintFlag: self.satisfy_integrity_constraint(name)
                # Centerilize The Shape
                if self.centralizeFlag: self.centralize_shape(name)

        # Verbose Part
        if self.verbose:
            logMsg(10*"-")

    def zero_grad(self):
        for name, param in self.model.named_parameters():
            if self.targetParam in name:
                param.grad.zero_()
