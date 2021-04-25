# Python packages
import numpy as np

# Pytorch imports
import torch
import torch.nn.modules.loss as Loss 
import torch.nn.functional as F
from matplotlib import pyplot as plt
# Helpers Import
from Solver import DeepFlowSolver
from Helper import printTensorAsImage,logMsg

class Velocity:
    class VELPACK:
        VEL  = 0
        COS  = 0
        SIN  = 0

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


class DragLoss(Loss._Loss):
    def __init__(self, velx, vely, **kwargs):
        super(DragLoss, self).__init__()
        self.ps = DeepFlowSolver(velx, vely, logStates=kwargs['logStates'] if 'logStates' in kwargs else False)
        
        self.kwargs = kwargs
        self.kwargs['upstream'] = Velocity(velx, vely)
        
        self.solver    = kwargs['solver'] if 'solver' in kwargs else self.ps.pressureSolver
        self.verbose   = kwargs['verbose'] if 'verbose' in kwargs else False
        self.normalize = kwargs['normalize'] if 'normalize' in kwargs else True
#
# lc: 31. Oct. 2019
        self.model     = kwargs['model'] if 'model' in kwargs else ShapeGen
        self.viscosity     = kwargs['viscosity'] if 'viscosity' in kwargs else 1e-5
#
    # function to extract grad
    def set_grad(self, var):
        def hook(grad):
            var.grad = grad
        return hook

    def _normalizePressure(self, pressureField):
        upstreamVelMagn = self.kwargs['upstream'].getVelMagnitude()
        npPressureField = pressureField.detach().numpy() 
        npPressureField = npPressureField / (0.5 * upstreamVelMagn * upstreamVelMagn) 
        return torch.Tensor( npPressureField )


    # Input:
    #        binaryMask is a pytorch array with size [sizeX, sizeY]
    #        pressureField is a pytorch array with size [sizeX, sizeY]
    def _calculateDragLift(self, binaryMask, pressureField, upstreamVel, verbose=False, jacobiRatio=128):
        kernel_dx = torch.Tensor( [[-1,  0,  1],
                                [-2,  0,  2],
                                [-1,  0,  1]] ) * (1/8)
        kernel_dy = torch.Tensor( [[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]] ) * (1/8)
        #kernel_dx = torch.Tensor( [[-0,  0,  0],
        #                           [-1,  0,  1],
        #                           [-0,  0,  0]] ) * (1/2)
        #kernel_dy = torch.Tensor( [[ 0, -1,  0],
        #                           [ 0,  0,  0],
        #                           [ 0,  1,  0]] ) * (1/2) # keep this

        kernel_dx  = kernel_dx.view((1,1,3,3))
        kernel_dy  = kernel_dy.view((1,1,3,3))
        self.binaryMask_pressure = binaryMask.view((1,1,binaryMask.size()[0], binaryMask.size()[1]))
        self.binaryMask_pressure.register_hook(self.set_grad(self.binaryMask_pressure))
        temp_idim  = pressureField.size()[0] 
        temp_jdim  = pressureField.size()[1]

        self.dx_layer_pressure   = F.conv2d(self.binaryMask_pressure, kernel_dx, padding=1).double() * 2. #face normal vector should be 1.0 not 0.5
        self.dy_layer_pressure   = F.conv2d(self.binaryMask_pressure, kernel_dy, padding=1).double() * 2. #face normal vector should be 1.0 not 0.5
        
        # if verbose:
        #     printTensorAsImage(self.dx_layer_pressure[0][0], "dx", display=True)
        #     printTensorAsImage(self.dy_layer_pressure[0][0], "dy", display=True)
        #     printTensorAsImage(pressureField, "pressure", display=True)
        #     ####
        #     #Jacobi Ratio
        #     print("Jacobi Ratio:", jacobiRatio)
        #     ####
        #pressureField = pressureField.view(1,1,temp_idim,temp_jdim).mul(1-self.binaryMask) # mult. a mask not needed in the current case; might be useful later
        pressureField = pressureField.view(1,1,temp_idim,temp_jdim)

        drag_x = torch.sum(torch.mul(self.dx_layer_pressure, pressureField ) ) / jacobiRatio
        drag_y = torch.sum(torch.mul(self.dy_layer_pressure, pressureField ) ) / jacobiRatio
        
        self.dx_layer_pressure.register_hook(self.set_grad(self.dx_layer_pressure))
        self.dy_layer_pressure.register_hook(self.set_grad(self.dy_layer_pressure))

        drag   = drag_x * upstreamVel.COS + drag_y * upstreamVel.SIN # need to be careful!!!!

        if verbose:
            drag.register_hook(logMsg)
            drag_x.register_hook(logMsg)
            drag_y.register_hook(logMsg)
            #self.dx_layer_pressure.register_hook(logMsg)
            #self.dy_layer_pressure.register_hook(logMsg)

        #plt.figure() # use for debug
        #plt.title("pressure in the image space")
        #plt.imshow(pressureField.view(temp_idim, temp_jdim).detach().numpy()[2:-2,2:-2])
        #plt.colorbar()
        #plt.figure()
        #plt.title("dx_layer * pressure in the image space")
        #plt.imshow(torch.mul(self.dx_layer_pressure, pressureField ).view(temp_idim, temp_jdim).detach().numpy())
        #plt.colorbar()
        #plt.figure()
        #plt.title("dy_layer * pressure in the image space")
        #plt.imshow(torch.mul(self.dy_layer_pressure, pressureField ).view(temp_idim, temp_jdim).detach().numpy())
        #plt.colorbar()
    
        #drag   = drag/0.5/upstreamVel.VEL/upstreamVel.VEL # outward-normal is positive
        print("pressure forces in X & Y directions:",drag_x.item(), drag_y.item())
        logMsg("pressure forces in X & Y directions: {}, {}".format(drag_x.item(), drag_y.item()))
        return drag, 1
        #return drag_x, drag_y
    
    # Input:
    #        binaryMask is a pytorch array with size [sizeX, sizeY]
    #        pressureField is a pytorch array with size [sizeX, sizeY]
    def _calculateDragLift_visc(self, binaryMask, velocityXField, velocityYField, upstreamVel, verbose=False, VISCOSITY=1e-5):
        kernel_dx = torch.Tensor( [[-1,  0,  1],
                                [-2,  0,  2],
                                [-1,  0,  1]] ) * (1/8)
        kernel_dy = torch.Tensor( [[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]] ) * (1/8)
        #kernel_dx = torch.Tensor( [[-0,  0,  0],
        #                           [-1,  0,  1],
        #                           [-0,  0,  0]] ) * (1/2)
        #kernel_dy = torch.Tensor( [[ 0, -1,  0],
        #                           [ 0,  0,  0],
        #                           [ 0,  1,  0]] ) * (1/2) # keep this 
    # lc: Sobel Operator --> normal vector
    # lc: in the sampled-pixel space, down is y-positive, right is x-positive
        kernel_dx  = kernel_dx.view((1,1,3,3))
        kernel_dy  = kernel_dy.view((1,1,3,3))
        self.binaryMask_visc = binaryMask.view((1,1,binaryMask.size()[0], binaryMask.size()[1]))
        self.binaryMask_visc.register_hook(self.set_grad(self.binaryMask_visc))

        temp_idim  = velocityXField.size()[0] 
        temp_jdim  = velocityXField.size()[1]
        velocityXField = velocityXField.view(1,1,temp_idim,temp_jdim)
        velocityYField = -velocityYField.view(1,1,temp_idim,temp_jdim) # reverse sign here to make coordinates consistent!
        #velocityXField = velocityXField.mul(1-self.binaryMask) 
        #velocityYField = velocityYField.mul(1-self.binaryMask) # mult. a mask not needed in the current case; might be useful later
        #print(binaryMask.size()) #1x1x 128 x 128 (resolution)?
        #print(velocityXField.size()) #1x1x128 x 128 (resolution)?
        #print(binaryMask)  
        
        self.dx_layer_visc   = F.conv2d(self.binaryMask_visc, kernel_dx, padding=1).double() * 2. 
        self.dy_layer_visc   = F.conv2d(self.binaryMask_visc, kernel_dy, padding=1).double() * 2. 
            
        self.dx_layer_visc.register_hook(self.set_grad(self.dx_layer_visc))
        self.dy_layer_visc.register_hook(self.set_grad(self.dy_layer_visc))

        #print("before omega calculation") 
        # lc: conv2d requires 4 dimensional inputs
        #print(binaryMask.size()) 
        #print(kernel_dx.size()) 
        #print(velocityYField.size()) 
        #print(binaryMask)
        #print(kernel_dx) 
        #print(velocityYField)
        kernel_dx = kernel_dx
        kernel_dy = kernel_dy # make sure they are of the same type 
        omega  =  F.conv2d(velocityYField, kernel_dx, padding=1)
        omega  = omega - F.conv2d(velocityXField, kernel_dy, padding=1)
        omega = omega.mul(1-self.binaryMask_visc) 
        #print(omega) 
        #torch.set_printoptions(threshold=10000)
        #print(omega.size()) #torch.Size([128, 128]) 128 x 128 (resolution) after conv2d
        #print(dx_layer.size()) #torch.Size([128, 128]) 128 x 128 (resolution) after conv2d
        #print("after omega calculation") #torch.Size([128, 128]) 128 x 128 (resolution) after conv2d
        # normal-> = dx i-> + dy j->
        # p normal-> = p nx-> + p ny->  
        #plt.figure() # use for debug
        #plt.title("omega in the image space")
        #plt.imshow(omega.view(temp_idim, temp_jdim).detach().numpy()[2:-2,2:-2])
        #plt.colorbar()

        #plt.figure()
        #plt.title("velocityXin the image space")
        #plt.imshow(velocityXField.view(temp_idim, temp_jdim).detach().numpy()[2:-2,2:-2])
        #plt.colorbar()
    
        #plt.figure()
        #plt.title("velocityYin the image space")
        #plt.imshow(velocityYField.view(temp_idim, temp_jdim).detach().numpy()[2:-2,2:-2])
        #plt.colorbar()
        #plt.show()
    
    
        # omega X normal -->
        # -n2 omega i 
        #  n1 omega j
        # force = - VISCOSITY (omega X normal)
        drag_x = -torch.sum(torch.mul(-self.dy_layer_visc, omega) ) * VISCOSITY
        #drag_y = -torch.sum(torch.mul(dx_layer_visc, omega) )
        drag_y = -torch.sum(torch.mul(self.dx_layer_visc, omega) ) * VISCOSITY
        drag   = drag_x * upstreamVel.COS + drag_y * upstreamVel.SIN # need to be careful
        #drag   = -drag*VISCOSITY #/0.5/upstreamVel.VEL/upstreamVel.VEL
        #drag   = drag*VISCOSITY #/0.5/upstreamVel.VEL/upstreamVel.VEL
        #print("viscous drag:",drag.item())
        print("viscous forces in X & Y directions:",drag_x.item(), drag_y.item())
        print("pass viscosity into visc. calc.:", VISCOSITY)
        logMsg("viscous forces in X & Y directions: {}, {}".format(drag_x.item(), drag_y.item()))
        return drag, 1
        #return drag_x, drag_y

    def forward(self, **params):
        params        = {**params, **self.kwargs}
        #pressureField = self.solver(**params)
        #lc:
        self.pressureField, velocityXField, velocityYField = self.solver(**params)
        #lc: be careful about velocityYField in the physical space and sampled space!!!!
    
        # plt.subplot(1,3, 1); plt.imshow(velocityXField); plt.colorbar()
        # plt.subplot(1,3, 2); plt.imshow(velocityYField); plt.colorbar()
        # plt.subplot(1,3, 3); plt.imshow(self.pressureField); plt.colorbar()
        # plt.show()  

        #velocityYField = - velocityYField  # do we really need to reverse this?
        #backup case we don't but for the standard OB run we probably need.

        #processedPF   = self._normalizePressure(pressureField) if self.normalize else pressureField
        #drag, _ = calculateDragLift(params['binaryMask'], processedPF, params['upstream'].passTupple(), self.verbose)
        drag, _      = self._calculateDragLift(params['binaryMask'], self.pressureField, params['upstream'].passTupple(), self.verbose, self.model.gridSize/2.0) # Here 2.0 is the domain size
        drag_visc, _ = self._calculateDragLift_visc(params['binaryMask'], velocityXField, velocityYField, params['upstream'].passTupple(), self.verbose, self.viscosity) # pass viscosity into visc. calc.
        print("Drag (pressure part, raw data):", drag.item(),     100*drag.item()/(drag.item()+drag_visc.item()),"%")
        print("Drag (viscous part, raw data):", drag_visc.item(), 100*drag_visc.item()/(drag.item()+drag_visc.item()), "%")
        logMsg("Drag (pressure part, raw data): {}.".format(drag.item()))
        logMsg("Drag (viscous part, raw data): {}.".format(drag_visc.item()))
        
        VISC_CALC = True
        COEFF_CALC = True
        if VISC_CALC: 
            drag += drag_visc
        if COEFF_CALC:
            upstreamVelMagn = self.kwargs['upstream'].getVelMagnitude()
            physicalChordLength =   self.model.getChordLength() / self.model.gridSize * 2.0
            print("Physical chord length (not used in loss calc.):", physicalChordLength)
            drag = drag / 0.5/upstreamVelMagn/upstreamVelMagn #/ physicalChordLength  # to be added soon... Liwei .... later add Density

        return drag
        #return drag*drag
