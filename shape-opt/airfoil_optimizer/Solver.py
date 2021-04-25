import numpy as np
import torch
import os, sys
import random
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))

#--------- Project Imports ----------#
from train.DfpNet             import TurbNetG
from train.dataset            import TurbDataset
from torch.utils.data         import DataLoader
from data.utils               import makeDirs
from data.asymmetricDataGen   import genSQDatFile
from data.dataGen             import genMesh, runSim, processResult, outputProcessing
from train.utils              import *
from airfoil_optimizer.Helper import printTensorAsImage, logMsg


binaryMask = np.array([ 
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,1,1,1,0,0,0,0,0],
                        [0,0,1,1,1,0,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0,0],
                        [0,0,0,1,1,1,1,0,0,0],
                        [0,0,0,0,1,1,1,0,0,0],
                        [0,0,0,0,1,1,1,1,1,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                                              ])

class OpenFoamSolver:
    def __init__(self, velX, velY, resolution, **kwargs):
        self.fsX =  velX
        self.fsY =  velY
        self.res =  resolution
        
        makeDirs( ["./online_solver/data_pictures"] )
        
        self.slvDump  = kwargs['solverDump'] if 'solverDump' in kwargs else "./online_solver/data_pictures/"
        self.tempFile = kwargs['tempFile'] if 'tempFile' in kwargs else './online_solver/SQ_ONLINE.dat'
        self.verbose  = kwargs['verbose'] if 'verbose' in kwargs else False
        self.upSampl  = kwargs['upSamplingRate'] if 'upSamplingRate' in kwargs else 1

        self.count = 0 
    
    def pressureSolver(self, **params):
        self.count += 1
       
        model = params['model'] if 'model' in params else None

        model.exportDatFile(outPath=self.tempFile)
        binaryMask = model.binaryMask.detach().numpy()
        
        if self.verbose: logMsg("\tResulting freestream vel x,y: {},{}".format(self.fsX,self.fsY))

        os.chdir("../data/OpenFOAM/")
        
        if genMesh("../../airfoil_optimizer/online_solver/SQ_ONLINE.dat") != 0:
            logMsg("\tmesh generation failed, aborting");
        else:
            runSim(self.fsX, self.fsY, res=self.res, oversamplingRate=self.upSampl)
        
        os.chdir("..")
        pressure = outputProcessing('OptSim_' + str(self.count), self.fsX, self.fsY, 
                                    res=self.res, binaryMask=binaryMask, 
                                    oversamplingRate= self.upSampl, imageIndex=self.count, interpolation=False)[3]
        #lc##### 
        velocityX = outputProcessing('OptSim_' + str(self.count), self.fsX, self.fsY, 
                                    res=self.res, binaryMask=binaryMask, 
                                    oversamplingRate= self.upSampl, imageIndex=self.count, interpolation=False)[4]
        velocityY = outputProcessing('OptSim_' + str(self.count), self.fsX, self.fsY, 
                                    res=self.res, binaryMask=binaryMask, 
                                    oversamplingRate= self.upSampl, imageIndex=self.count, interpolation=False)[5]
        ########
        os.chdir("../airfoil_optimizer/")
        
        pressure = np.flipud(pressure.transpose())
        #lc##### 
        velocityX = np.flipud(velocityX.transpose())
        velocityY = np.flipud(velocityY.transpose())
        ########
        
        #import pdb; pdb.set_trace()
        # plt.subplot(1,3, 1); plt.imshow(velocityX); plt.colorbar()
        # plt.subplot(1,3, 2); plt.imshow(velocityY); plt.colorbar()
        # plt.subplot(1,3, 3); plt.imshow(pressure); plt.colorbar()
        # plt.show() 


        if self.verbose: 
            logMsg("\tMeshing and Simulation are done!")
            printTensorAsImage(pressure, self.slvDump + 'pressure_' + str(self.count), display=False)
            #lc##### 
            printTensorAsImage(velocityX, self.slvDump + 'velocityX_' + str(self.count), display=False)
            printTensorAsImage(velocityY, self.slvDump + 'velocityY_' + str(self.count), display=False)
            ########
        
         
        return torch.from_numpy(pressure.copy()), torch.from_numpy(velocityX.copy()), torch.from_numpy(velocityY.copy())


class ModelNotFound(Exception):
    pass

class DeepFlowSolver:
    def __init__(self, Velx, Vely, 
                 channelExpo=5, modelPath='./saved_models/modelG',
                 logStates = False):
        
        if not os.path.isfile(modelPath):
            raise ModelNotFound
        
        self.log = logStates
        if self.log: 
            self.counter = 0
            makeDirs( ["./solver_dump"])

        self.netG = TurbNetG(channelExponent=channelExpo)
        self.netG.load_state_dict( torch.load(modelPath, map_location='cpu') )
        self.netG.eval()

        self.Velx = Velx#/100
        self.Vely = Vely/0.01
    
    def _getInputArray(self, binaryMask):
        binaryMaskInv = np.flipud((binaryMask-1)*(-1)).transpose()
        channelfsX    = binaryMaskInv * self.Velx
        channelfsY    = binaryMaskInv * self.Vely
        input = np.zeros((1, 3, binaryMaskInv.shape[0],binaryMaskInv.shape[1]))
        input[0, 0:,] = channelfsX
        input[0, 1:,] = channelfsY
        input[0, 2:,] = binaryMaskInv

        return input

    def _getPressure(self, fullSolution):
        pressure = fullSolution[0, 0, :]
        pressure = np.flipud(pressure.transpose())
        
        return pressure
    
    def _getVelocityX(self, fullSolution):
        velocityX = fullSolution[0, 1, :]
        velocityX = np.flipud(velocityX.transpose())
        
        return velocityX

    def _getVelocityY(self, fullSolution):
        velocityY = fullSolution[0, 2, :]
        velocityY = np.flipud(velocityY.transpose())
        
        return velocityY

    def pressureSolver(self, **kwargs):
        self.counter += 1

        binaryMask = kwargs['binaryMask'] if 'binaryMask' in kwargs else None
        if type(binaryMask).__module__ == torch.__name__:
            numpyMask = binaryMask.detach().numpy()
        elif type(binaryMask).__module__ == np.__name__:
            numpyMask = binaryMask

        input      = self._getInputArray(numpyMask)                                                                                                                                                

        # #import pdb; pdb.set_trace()
        # for i in range(3):
        #     plt.subplot(1,3, i+1)
        #     plt.imshow(input[0][i])
        #     plt.colorbar()
        # plt.show()    

        prediction = self.netG(torch.Tensor(input)).detach().numpy()

        # #import pdb; pdb.set_trace()
        # for i in range(3):
        #     plt.subplot(1,3, i+1)
        #     plt.imshow(prediction[0][i])
        #     plt.colorbar()
        # plt.show()    

        pressure   = self._getPressure(prediction)
        velocityX  = self._getVelocityX(prediction)
        velocityY  = self._getVelocityY(prediction)

        ##### Denormalization #####
        # denormalization code copy-pasted from train/dataset.py
        max_targets_0 = 1.13598 
        max_targets_1 = 0.61682
        max_targets_2 = 1.23718
        pressure  /= (1.0/max_targets_0)
        velocityX /= (1.0/max_targets_1)
        velocityY /= (1.0/max_targets_2)        
        
        # make dimless
        v_norm = 0.01 # changes with dataset!
        pressure  *= v_norm**2
        velocityX *= v_norm
        velocityY *= v_norm
        ###########################


        # import pdb; pdb.set_trace()
        # plt.subplot(1,3, 1); plt.imshow(velocityX); plt.colorbar()
        # plt.subplot(1,3, 2); plt.imshow(velocityY); plt.colorbar()
        # plt.subplot(1,3, 3); plt.imshow(pressure); plt.colorbar()
        # plt.show()  


        if self.log:
            imageOut('./solver_dump/fullSolution_' + str(self.counter), input[0,:], prediction[0,:], normalize=True)        
            printTensorAsImage(pressure,  './solver_dump/pressure_'  + str(self.counter), display=False)
            printTensorAsImage(velocityX, './solver_dump/velocityX_' + str(self.counter), display=False)
            printTensorAsImage(velocityY, './solver_dump/velocityY_' + str(self.counter), display=False)

        return torch.from_numpy(pressure.copy()), torch.from_numpy(velocityX.copy()), torch.from_numpy(velocityY.copy())
