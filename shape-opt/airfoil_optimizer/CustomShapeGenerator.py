# Python packages
import random, json
import matplotlib.pyplot as plt
import numpy as np
import math  

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as Loss 

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))

# Helpers Import
from data.asymmetricDataGen import genSQDatFile
from Helper                 import _getDiscrateSpaceVector, printTensorAsImage, pascalsTriangle, logMsg, logError
from ShapeGenerator         import ShapeGenerator

## Random Seed ##
SEED = random.randint(0,2**32-1)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)

##################### Data Gen Settings ##################
_FEASIBLE_REGIONS_X = [20, 58, 70, 108]                     #       0   20     60  68     108 128
_FEASIBLE_REGIONS_Y = [20, 108]                             #       |   |   x   |  |   x   |   |
_NUMBER_OF_REGIONS_X = 5 #e.g. num. of regs. betw. 20-60    #       |   |   x   |  |   x   |   |                                                            

_FEASIBLE_REGIONS_LP_TP_X = [64 - 0, 64 + 0]
_SYMMETRIC_SHAPE = True

_OUTPUT_DIRECTORY = "./../data/database/bezier_symmetric/"
##########################################################

class CustomShapeGenerator(nn.Module):
    def __init__(self):
        super(CustomShapeGenerator, self).__init__()
        self.sg = ShapeGenerator(gridSize=128, numOfPointsPerCurve=6, refArea=2000, refWedgeAngle=None)
        self.sg.importShape("params/initialShape_6BezierPoints.param")
        self.sg.numOfPointsPerCurve = 6

        self.numOfRegionX = _NUMBER_OF_REGIONS_X
        self.numOfRegionY = self.sg.numOfPointsPerCurve - 2

        self._get_feasible_regions_for_bezier_points()

    def _get_feasible_regions_for_bezier_points(self):
        # Feasible region for TP and LP on the x axis
        self.TP_LP_region_limits_x = np.zeros((1,2))
        self.TP_LP_region_limits_x[0][0] = _FEASIBLE_REGIONS_LP_TP_X[0]
        self.TP_LP_region_limits_x[0][1] = _FEASIBLE_REGIONS_LP_TP_X[1]

        self.chunk_size_x = int((_FEASIBLE_REGIONS_X[3] - _FEASIBLE_REGIONS_X[2]) / self.numOfRegionX)
        self.chunk_size_y = int((_FEASIBLE_REGIONS_Y[1] - _FEASIBLE_REGIONS_Y[0]) / self.numOfRegionY)

        # Divide into sub regions on the X axis
        self.CP_right_region_limits_x = np.zeros((self.numOfRegionX, 2))
        self.CP_right_region_limits_x[0][0] = _FEASIBLE_REGIONS_X[2]
        self.CP_right_region_limits_x[0][1] = self.CP_right_region_limits_x[0][0] + self.chunk_size_x

        self.CP_left_region_limits_x = np.zeros((self.numOfRegionX, 2))
        self.CP_left_region_limits_x[0][0] = _FEASIBLE_REGIONS_X[0]
        self.CP_left_region_limits_x[0][1] = self.CP_left_region_limits_x[0][0] + self.chunk_size_x

        for i in range(1, self.numOfRegionX):
            self.CP_right_region_limits_x[i][0] = self.CP_right_region_limits_x[i-1][1]
            self.CP_right_region_limits_x[i][1] = self.CP_right_region_limits_x[i][0] + self.chunk_size_x
            self.CP_left_region_limits_x[i][0]  = self.CP_left_region_limits_x[i-1][1]
            self.CP_left_region_limits_x[i][1]  = self.CP_left_region_limits_x[i][0] + self.chunk_size_x

        # Divide into sub regions on the Y axis
        self.region_limits_y = np.zeros((self.numOfRegionY, 2))
        self.region_limits_y[0][0] = _FEASIBLE_REGIONS_Y[0]
        self.region_limits_y[0][1] = self.region_limits_y[0][0] + self.chunk_size_y

        for j in range(1, self.numOfRegionY):
            self.region_limits_y[j][0] = self.region_limits_y[j-1][1]
            self.region_limits_y[j][1] = self.region_limits_y[j][0] + self.chunk_size_y 
     
    def generate_shapes(self):

        for sample in range(4):
            region_idx_list = np.zeros(2 * self.numOfRegionY)          
            setOfShapes = []
            idx = 0
            
            if _SYMMETRIC_SHAPE:
                max_idx = self.numOfRegionX**(self.numOfRegionY)
            else:
                max_idx = self.numOfRegionX**(2*self.numOfRegionY)

            while(1):
                self.sg._parameters["LP"].data = torch.Tensor( [[np.random.uniform(low=self.TP_LP_region_limits_x[0][0], high=self.TP_LP_region_limits_x[0][1]), _FEASIBLE_REGIONS_Y[0] ] ])   
                self.sg._parameters["TP"].data = torch.Tensor( [[np.random.uniform(low=self.TP_LP_region_limits_x[0][0], high=self.TP_LP_region_limits_x[0][1]), _FEASIBLE_REGIONS_Y[1] ] ])   

                CP_left_region_idx = region_idx_list[:self.numOfRegionY] 
                CP_right_region_idx  = region_idx_list[self.numOfRegionY:] 

                for cp in range(self.numOfRegionY):
                    regx_right = int(CP_right_region_idx[cp])
                    regx_left  = int(CP_left_region_idx[cp])
                    
                    # if (np.random.randint(low=0, high=2) == 1):
                    ### first CP_right, then CP_left
                    self.sg._parameters["CP_Right"].data[cp] = torch.Tensor( [[np.random.uniform(low=self.CP_right_region_limits_x[regx_right][0], high=self.CP_right_region_limits_x[regx_right][1]), 
                                                                               np.random.uniform(low=self.region_limits_y[cp][0], high=self.region_limits_y[cp][1])] ])            
                    
                    if(_SYMMETRIC_SHAPE):
                        self.sg._parameters["CP_Left"].data[cp][1]  = self.sg._parameters["CP_Right"].data[cp][1]
                        self.sg._parameters["CP_Left"].data[cp][0]  = 128 - self.sg._parameters["CP_Right"].data[cp][0]
                    else:
                        self.sg._parameters["CP_Left"].data[cp] = torch.Tensor( [[np.random.uniform(low=self.CP_left_region_limits_x[self.numOfRegionX-1-regx_left][0], high=self.CP_left_region_limits_x[self.numOfRegionX-1-regx_left][1]), 
                                                                                  np.random.uniform(low=self.region_limits_y[cp][0], high=self.region_limits_y[cp][1])] ])            
                    # else:
                    ### first CP_left, then CP_right
                    # self.sg._parameters["CP_Left"].data[cp] = torch.Tensor( [[np.random.uniform(low=self.CP_left_region_limits_x[regx_left][0], high=self.CP_left_region_limits_x[regx_left][1]), 
                    #                                                             np.random.uniform(low=self.region_limits_y[cp][0], high=self.region_limits_y[cp][1])] ])            
                    
                    # if(_SYMMETRIC_SHAPE):
                    #     self.sg._parameters["CP_Right"].data[cp][1]  = self.sg._parameters["CP_Left"].data[cp][1]
                    #     self.sg._parameters["CP_Right"].data[cp][0]  = 128 - self.sg._parameters["CP_Left"].data[cp][0]
                    # else:
                    #     self.sg._parameters["CP_Right"].data[cp] = torch.Tensor( [[np.random.uniform(low=self.CP_right_region_limits_x[self.numOfRegionX-1-regx_left][0], high=self.sg._parameters["CP_Left"].data[cp][0] + 10), 
                    #                                                                 np.random.uniform(low=self.region_limits_y[cp][0], high=self.region_limits_y[cp][1])] ])            

                bezierCurves = self.sg._getBezierCurves(self.sg._parameters["LP"], self.sg._parameters["TP"], self.sg._parameters["CP_Right"], self.sg._parameters["CP_Left"], returnFloat=True) 
                setOfShapes.append(bezierCurves)
                
                image = self.sg.forward() # area constraint is applied in forward routine
                self.sg.exportDatFile(OUT_DIR=_OUTPUT_DIRECTORY)

                # plt.scatter(bezierCurves[:,0], bezierCurves[:,1], marker=".", alpha=1, linewidths=0.1); plt.ylim(0,128); plt.xlim(0,128)
                # plt.savefig("./shapes/shape_{}".format(idx))
                # plt.clf()

                print("Shape generated: {}".format(idx))
                idx += 1

                if idx==max_idx:
                    break

                remainder = idx
                for cp in range(self.numOfRegionY*2):
                    region_idx_list[cp] = remainder // self.numOfRegionX**(2*self.numOfRegionY - 1 - cp) #CP_right_region_idx[cp]
                    remainder           = idx        % self.numOfRegionX**(2*self.numOfRegionY - 1 - cp)
               
            """
            randShapes = np.random.randint(low=0, high=max_idx-1, size=min(40, max_idx))
            for i in range(min(40, max_idx)):
                plt.subplot(5,8,i+1)
                plt.scatter(setOfShapes[randShapes[i]][:,0], setOfShapes[randShapes[i]][:,1], marker=".", alpha=1, linewidths=0.1); plt.ylim(0,128); plt.xlim(0,128)
            
            plt.show()
            """

            #image = self.sg.forward()
            #printTensorAsImage(self.sg.getBinaryMaskWithBezierPoints(), "image", display=True)

if __name__ == "__main__":
    csg = CustomShapeGenerator()
    csg._get_feasible_regions_for_bezier_points()
    csg.generate_shapes()
