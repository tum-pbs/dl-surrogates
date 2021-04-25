# Python packages
import random, json
import matplotlib.pyplot as plt
import numpy as np
import math  
# import skfmm

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

## Random Seed ##
SEED = random.randint(0,2**32-1)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)

class ShapeGenerator(nn.Module):
    def __init__(self, gridSize, numOfPointsPerCurve, refArea, refWedgeAngle):
        super(ShapeGenerator, self).__init__()
        self.numOfControlPointsPerCurve = numOfPointsPerCurve - 2
        self.numOfPointsPerCurve        = numOfPointsPerCurve
        self.gridSize                   = gridSize
        self.refArea                    = refArea  
        self.refWedgeAngle              = refWedgeAngle   

        self.visc_grad_x = None
        self.visc_grad_y = None
        self.pressure_grad_x = None
        self.pressure_grad_y = None

        self._generateLearnableParams()

    def _generateLearnableParams(self):
        self.register_parameter("LP", torch.nn.Parameter(torch.tensor(np.random.uniform(low=40, high=80, size=(1,2)))))
        self.register_parameter("TP", torch.nn.Parameter(torch.tensor(np.random.uniform(low=40, high=80, size=(1,2)))))

        self.register_parameter("CP_Right", torch.nn.Parameter(torch.tensor(np.random.uniform(low=40, high=80, size=(self.numOfControlPointsPerCurve, 2)))))
        self.register_parameter("CP_Left",  torch.nn.Parameter(torch.tensor(np.random.uniform(low=40, high=80, size=(self.numOfControlPointsPerCurve, 2)))))

        self._parameters["LP"].requires_grad_()
        self._parameters["TP"].requires_grad_()
        self._parameters["CP_Right"].requires_grad_()
        self._parameters["CP_Left"].requires_grad_()

    def _writePointsIntoFile(self, dataName, array, fileName):
        with open(fileName, "wt") as outFile:
            outFile.write(dataName + "\n")
            prev_point = array[0]
            outFile.write( 4*" " + np.array2string( array[0], precision=10, separator=' ' )[1:-1].lstrip(' ') + "\n" )
            for i in range(1, array.shape[0]):
                if not (array[i][0] == prev_point[0] and array[i][1] == prev_point[1]):
                    outFile.write( 4*" " + np.array2string( array[i], precision=10, separator=' ' )[1:-1].lstrip(' ') + "\n" )
                    prev_point = array[i]

    def importShape(self, jsonDictFilePath):
        listOfParams = ['LP', 'TP', 'CP_Right', 'CP_Left']
        if os.path.exists(jsonDictFilePath):
            try:
                with open(jsonDictFilePath, 'r') as params:
                    paramDict = json.load(params)
                    if all(paramName in paramDict for paramName in listOfParams):
                        for paramName in listOfParams:
                            self._parameters[paramName].data = torch.Tensor(paramDict[paramName]) 
                        logMsg("Parameter(s) '{}' has/have loaded from file {}.".format(', '.join(listOfParams), jsonDictFilePath))  
                        return
                    logError("Initializataions parameters doesn't contain all list of params: '{}'".format(', '.join(listOfParams)))
                    return
            except Exception as err:
                logError("Opening file '{}' has failed.".format(jsonDictFilePath))
            return
        logError("File doesn't exist. ({})".format(jsonDictFilePath))
        return

    def exportShape(self, exportFilePath):
        listOfParams = ['LP', 'TP', 'CP_Right', 'CP_Left']
        try:
            dumpDict = {}
            for paramName in listOfParams:
                dumpDict[paramName] = self._parameters[paramName].tolist()
            with open(exportFilePath, 'w') as export:
                json.dump(dumpDict, export)
        except Exception as err:
            logError("Error happened while exporting parameters to {}. ({})".format(exportFilePath, str(err)))
            return False
        return True
    
    def exportDatFile(self, outPath=None, OUT_DIR = "./../data/database/bezier_airfoil_database/"):
        bezierCurves = self._getBezierCurves(self._parameters["LP"], self._parameters["TP"], self._parameters["CP_Right"], self._parameters["CP_Left"], returnFloat=True)

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        
        filesuffix = 0
        filesInDirectory = os.listdir(OUT_DIR)
        if len(filesInDirectory) > 0:
            filesuffix = max([int(fileName.split('.')[0].split('SQ_')[1]) for fileName in filesInDirectory]) + 1
        
        dataNeme = "SQ_" + str(filesuffix).zfill(5)
        outFile  = os.path.join(OUT_DIR, dataNeme + ".dat")

        SQ_points = bezierCurves.detach().numpy().copy()
        SQ_points = SQ_points / self.gridSize * 2 - 1
        SQ_points[:, 1] *= -1 
        
        if outPath:
            self._writePointsIntoFile('SQ_ONLINE',  SQ_points, outPath)
        else:
            self._writePointsIntoFile(dataNeme,     SQ_points, outFile)   

    def _getBezierCurves(self, lp, tp, cr, cl, numOfSamples=None, returnFloat=False):
        if not numOfSamples: numOfSamples = 4*self.gridSize 
        samplePoints = np.linspace(0, 1, numOfSamples)
        self.bezierMatrix = torch.Tensor(self._getSampleMatrix(samplePoints, self.numOfPointsPerCurve))

        leftCurve = [lp.data, cl.data, tp.data]
        leftCurve = torch.cat(leftCurve)
        rightCurve = [lp.data, cr.data, tp.data]
        rightCurve = torch.cat(rightCurve)

        if returnFloat:
            leftCurveSamplePoints  = torch.mm(self.bezierMatrix, leftCurve)
            rightCurveSamplePoints = torch.mm(self.bezierMatrix, rightCurve)
            return torch.cat([leftCurveSamplePoints, torch.from_numpy(np.flipud(rightCurveSamplePoints.detach().numpy()).copy())])
        
        else:
            self.leftCurveSamplePoints  = torch.floor(torch.mm(self.bezierMatrix, leftCurve)).int()
            self.rightCurveSamplePoints = torch.floor(torch.mm(self.bezierMatrix, rightCurve)).int()            
            return torch.cat([self.leftCurveSamplePoints, torch.from_numpy(np.flipud(self.rightCurveSamplePoints.detach().numpy()).copy())])

    def _getSampleMatrix(self, t, N):
        comb = pascalsTriangle(N, normalize=False, centerBiased=False)
        C = np.reshape(np.repeat(np.array([comb]), t.shape[0]), (N, -1))
        T = np.array([t**i for i in range(N)])
        T_prime = np.array([(1-t)**i for i in range(N-1, -1, -1)])
        return np.multiply(np.transpose(C), np.transpose(np.multiply(T, T_prime)))

    def _calculateBezierCurveDerivatives(self, lp, tp, cr, cl, numOfSamples=None):
        if not numOfSamples: numOfSamples = 4*self.gridSize 
        samplePoints = np.linspace(0, 1, numOfSamples)
        self.bezierMatrix = torch.Tensor(self._getSampleMatrix(samplePoints, self.numOfPointsPerCurve-1))

        leftCurve = [lp.data, cl.data, tp.data]
        leftCurve = torch.cat(leftCurve)
        rightCurve = [lp.data, cr.data, tp.data]
        rightCurve = torch.cat(rightCurve)

        leftCurveGradients  = self.numOfPointsPerCurve * (torch.mm(self.bezierMatrix, leftCurve[1:]) - torch.mm(self.bezierMatrix, leftCurve[:-1]))
        rightCurveGradients = self.numOfPointsPerCurve * (torch.mm(self.bezierMatrix, rightCurve[1:]) - torch.mm(self.bezierMatrix, rightCurve[:-1]))
        return rightCurveGradients, leftCurveGradients 

    def _getWedgeAngles(self, rightCurveGradients, leftCurveGradients):
        lp_gradient_left  = leftCurveGradients[0]
        lp_gradient_right = rightCurveGradients[0]
        
        tp_gradient_left  = leftCurveGradients[-1]     
        tp_gradient_right = rightCurveGradients[-1]

        lp_wedge = torch.acos(torch.dot(lp_gradient_left, lp_gradient_right) / ( torch.sqrt(torch.dot(lp_gradient_left, lp_gradient_left)) *  torch.sqrt(torch.dot(lp_gradient_right, lp_gradient_right)) ) )
        lp_wedge = lp_wedge * 180 / math.pi

        tp_wedge = torch.acos(torch.dot(tp_gradient_left, tp_gradient_right) / ( torch.sqrt(torch.dot(tp_gradient_left, tp_gradient_left)) *  torch.sqrt(torch.dot(tp_gradient_right, tp_gradient_right)) ) )
        tp_wedge = tp_wedge * 180 / math.pi

        return lp_wedge, tp_wedge

    def _getBoundingBox(self, points):
        points = points.detach().numpy()
        
        row_min = min(points[:,1])
        row_max = max(points[:,1])
        col_min = min(points[:,0])
        col_max = max(points[:,0])        

        return row_min, row_max, col_min, col_max

    def _rasterizeCurves(self, curvePoints):
        binaryMask = np.zeros(shape=(self.gridSize, self.gridSize))

        row_min, row_max, col_min, col_max = self._getBoundingBox(curvePoints)

        points = curvePoints.detach().tolist()

        center_row = 0; center_col = 0
        for row in range(row_min, row_max+1):
            for col in range(col_min, col_max+1):
                if( [col, row] in points and 0 < col < self.gridSize and 0 < row < self.gridSize):
                    binaryMask[row, col] = 1
                    center_row += row
                    center_col += col

                else:
                    hit_left = False; hit_right = False; hit_up = False; hit_down = False

                    ## cast ray to up
                    row_m = row
                    while row_m > row_min-2:
                        if( [col, row_m] in points and 0 < col < self.gridSize and 0 < row < self.gridSize ):
                            hit_up = True
                            break
                        row_m -=1

                    ## cast ray to down
                    row_p = row
                    while row_p < row_max+2:
                        if( [col, row_p] in points and 0 < col < self.gridSize and 0 < row < self.gridSize ):
                            hit_down = True
                            break
                        row_p +=1

                    ## cast ray to left
                    col_m = col
                    while col_m > col_min-2:
                        if( [col_m, row] in points and 0 < col < self.gridSize and 0 < row < self.gridSize ):
                            hit_left = True
                            break
                        col_m -=1

                    ## cast ray to right
                    col_p = col
                    while col_p < col_max+2:
                        if( [col_p, row] in points and 0 < col < self.gridSize and 0 < row < self.gridSize ):
                            hit_right = True
                            break
                        col_p +=1

                
                    if hit_up and hit_down and hit_left and hit_right:
                        binaryMask[row, col] = 1
                        center_row += row
                        center_col += col

        area = np.sum(binaryMask)
        self.center_row = center_row / area
        self.center_col = center_col / area
        self.curveCenter = torch.Tensor([[self.center_col, self.center_row]])

        print("Image center: " + str(self.center_col) + ", " + str(self.center_row))
        return torch.Tensor(binaryMask)

    # def _set_SDF_values(self, curvePoints):
    #     binaryMask   = self._rasterizeCurves(curvePoints)
    #     printTensorAsImage(binaryMask, "shapeBinaryOrijinal", display=True)

    #     shape = np.where(binaryMask==1, -1, 1)        
    #     for point in range(curvePoints.shape[0]):
    #         shape[curvePoints[point][1]][curvePoints[point][0]] = 0

    #     printTensorAsImage(shape, "shapePreprocessed", display=True)

    #     shapeSDF = skfmm.distance(shape, dx=0.01)
    #     printTensorAsImage(shapeSDF, "shapeSDF", display=True)

    #     recoveredBinary = np.where(shapeSDF>0, 0, 1)
    #     printTensorAsImage(recoveredBinary, "recoveredBinary", display=True)

    #     #plt.contour(np.flipud(shapeSDF), levels=20); plt.colorbar()
    #     return shapeSDF

    def getChordLength(self):
        return math.sqrt((self._parameters["LP"].data[0][0] - self._parameters["TP"].data[0][0])**2 + (self._parameters["LP"].data[0][1] - self._parameters["TP"].data[0][1])**2) 

    def setDesignLoss(self, designLoss):
        self.designLoss = designLoss

    def _shiftImageToCenter(self):
        shift = torch.Tensor([[self.gridSize/2, self.gridSize/2]]) - self.curveCenter
        print("Image is shifted by: " + str(shift))

        self._parameters["LP"]       = self._parameters["LP"] + shift                                
        self._parameters["TP"]       = self._parameters["TP"] + shift                                
        self._parameters["CP_Right"] = self._parameters["CP_Right"] + shift 
        self._parameters["CP_Left"]  = self._parameters["CP_Left"]  + shift         

        self.bezierCurves = self._getBezierCurves(self._parameters["LP"], self._parameters["TP"], self._parameters["CP_Right"], self._parameters["CP_Left"] )
        self.binaryMask   = self._rasterizeCurves(self.bezierCurves)

    def _satisfy_area_constraint(self):
        area = torch.sum(self.binaryMask).item()
        
        distance_lp = self._parameters["LP"]       - self.curveCenter
        distance_tp = self._parameters["TP"]       - self.curveCenter
        distance_cr = self._parameters["CP_Right"] - self.curveCenter 
        distance_cl = self._parameters["CP_Left"]  - self.curveCenter 

        lp = self._parameters["LP"]      
        tp = self._parameters["TP"]      
        cr = self._parameters["CP_Right"]
        cl = self._parameters["CP_Left"] 
        binaryMask = self.binaryMask
        bezierCurves = self.bezierCurves

        scaling = 1.0
        for count in range(15):
            if (area / self.refArea  < 1.005) and (area / self.refArea > 0.995):
                break
            
            scaling  = scaling * (1 + (self.refArea - area)/area * 5e-1)
            if scaling > 0: scaling = np.minimum(scaling, 1.5)
            if scaling < 0: scaling = np.maximum(scaling, -1.5)            

            new_distance_lp = distance_lp * scaling
            new_distance_tp = distance_tp * scaling
            new_distance_cr = distance_cr * scaling
            new_distance_cl = distance_cl * scaling            

            print("Area / ReferanceArea: " + str(area / self.refArea) + ", scaling shape with factor " + str(scaling) + ", atemp: " + str(count+1))

            lp = self.curveCenter + new_distance_lp
            tp = self.curveCenter + new_distance_tp
            cr = self.curveCenter + new_distance_cr
            cl = self.curveCenter + new_distance_cl

            bezierCurves = self._getBezierCurves(lp, tp, cr, cl)
            binaryMask   = self._rasterizeCurves(bezierCurves)
            area  = torch.sum(binaryMask)

        print("    Area / ReferanceArea: " + str(area / self.refArea) + ", constraint satisfied. ")
        self._parameters["LP"] = lp
        self._parameters["TP"] = tp
        self._parameters["CP_Right"] = cr
        self._parameters["CP_Left"]  = cl
        self.binaryMask = binaryMask
        self.bezierCurves = bezierCurves

    def _satisfy_wedge_angle_constraint(self):
        rightCurveGradients, leftCurveGradients = self._calculateBezierCurveDerivatives(self._parameters["LP"], self._parameters["TP"], self._parameters["CP_Right"], self._parameters["CP_Left"])
        lp_wedge, tp_wedge = self._getWedgeAngles(rightCurveGradients, leftCurveGradients)

        wedgeDiff_lp = self.refWedgeAngle - lp_wedge 
        wedgeDiff_tp = self.refWedgeAngle - tp_wedge

        chord_vector            = self._parameters["TP"].detach() - self._parameters["LP"].detach()    
        chord_vector_normalized = chord_vector.div(torch.norm(chord_vector, p=2, dim=1).expand_as(chord_vector))

        new_lp = self._parameters["LP"]
        new_tp = self._parameters["TP"]

        for count in range(50):
            wedgeDiff_lp = self.refWedgeAngle - lp_wedge 
            wedgeDiff_tp = self.refWedgeAngle - tp_wedge

            # satisfy constraint for lp
            if not  abs( self.refWedgeAngle - lp_wedge) < 0.1:
                scaling_lp = wedgeDiff_lp / self.refWedgeAngle * 5
                if scaling_lp < -5: scaling_lp = np.maximum(scaling_lp, -5)
                new_lp     = new_lp + chord_vector_normalized * scaling_lp
                
                pixel_count = torch.norm(chord_vector_normalized * scaling_lp, p=2, dim=1)
                print("WedgeAngle_LP / ReferanceWedgeAngle: " + str(lp_wedge / self.refWedgeAngle) + ", LP has been moved along chord vector by " + str(pixel_count) + " pixels, atemp: " + str(count+1))

            # satisfy constraint for tp
            if not  abs( self.refWedgeAngle - tp_wedge ) < 0.1:
                scaling_tp  = wedgeDiff_tp / self.refWedgeAngle * 5
                if scaling_tp < -5: scaling_tp = np.maximum(scaling_tp, -5)
                new_tp      = new_tp - chord_vector_normalized * scaling_tp
                
                pixel_count = torch.norm(chord_vector_normalized * scaling_tp, p=2, dim=1)
                print("WedgeAngle_TP / ReferanceWedgeAngle: " + str(tp_wedge / self.refWedgeAngle) + ", TP has been moved along chord vector by " + str(pixel_count) + " pixels, atemp: " + str(count+1))

            leftCurveGradients, rightCurveGradients = self._calculateBezierCurveDerivatives(new_lp, new_tp, self._parameters["CP_Right"], self._parameters["CP_Left"])
            lp_wedge, tp_wedge = self._getWedgeAngles(leftCurveGradients, rightCurveGradients)
            

        print("    LP_WedgeAngle / ReferanceWedgeAngle: " + str(lp_wedge / self.refWedgeAngle) + ", constraint satisfied. ")
        print("    TP_WedgeAngle / ReferanceWedgeAngle: " + str(tp_wedge / self.refWedgeAngle) + ", constraint satisfied. ")
        self._parameters["LP"] = new_lp
        self._parameters["TP"] = new_tp

        bezierCurves = self._getBezierCurves(new_lp, new_tp, self._parameters["CP_Right"], self._parameters["CP_Left"])
        binaryMask   = self._rasterizeCurves(bezierCurves)

        self.binaryMask = binaryMask
        self.bezierCurves = bezierCurves

    def getBinaryMaskWithBezierPoints(self):
        binaryMask = torch.Tensor(self.binaryMask.detach().numpy().copy())

        for i in range(self.numOfControlPointsPerCurve):
            col = self._parameters["CP_Right"].data[i][0].int()
            row = self._parameters["CP_Right"].data[i][1].int()
            if 0 < col < self.gridSize and 0 < row < self.gridSize: 
                binaryMask[row][col] = 1            

            col = self._parameters["CP_Left"].data[i][0].int()
            row = self._parameters["CP_Left"].data[i][1].int()
            if 0 < col < self.gridSize and 0 < row < self.gridSize: 
                binaryMask[row][col] = 1  

        return binaryMask

    def forward(self):
        self.bezierCurves = self._getBezierCurves(self._parameters["LP"], self._parameters["TP"], self._parameters["CP_Right"], self._parameters["CP_Left"] )
        self.binaryMask   = self._rasterizeCurves(self.bezierCurves)

        if self.refWedgeAngle != None:  self._satisfy_wedge_angle_constraint()
        self._shiftImageToCenter()
        if self.refArea != None: self._satisfy_area_constraint()

        self.binaryMask.requires_grad_()
        return self.binaryMask

    def _calculateBezierCurveNormals(self, lp, tp, cr, cl):
        rightCurveGradients, leftCurveGradients  = self._calculateBezierCurveDerivatives(lp, tp, cr, cl)

        rightCurveNormals = np.zeros_like(rightCurveGradients)
        leftCurveNormals  = np.zeros_like(leftCurveGradients)

        for i in range(rightCurveGradients.shape[0]):
            rightCurveNormals[i][0] = -1 
            rightCurveNormals[i][1] = (rightCurveGradients[i][0] / rightCurveGradients[i][1] ) 

            magnitude = rightCurveNormals[i][0]*rightCurveNormals[i][0] + rightCurveNormals[i][1]*rightCurveNormals[i][1]
            rightCurveNormals[i] = rightCurveNormals[i] /  np.sqrt(magnitude)

            leftCurveNormals[i][0] = 1 / np.sqrt(magnitude)
            leftCurveNormals[i][1] = (-leftCurveGradients[i][0] / leftCurveGradients[i][1]) / np.sqrt(magnitude)

            magnitude = leftCurveNormals[i][0]*leftCurveNormals[i][0] + leftCurveNormals[i][1]*leftCurveNormals[i][1]
            leftCurveNormals[i] = leftCurveNormals[i] /  np.sqrt(magnitude)

        return rightCurveNormals, leftCurveNormals

    def _extract_grads(self, samplePoints, sampleNormals, dLoss_dBinaryMask, bezierMatrix):
        dLoss_dSamplePoints = np.zeros(shape=(samplePoints.shape[0],2))
        
        for i in range(samplePoints.shape[0]):
            ix = samplePoints[i][0]
            iy = samplePoints[i][1]

            neighbors = dLoss_dBinaryMask[iy-2:iy+3, ix-2:ix+3]
            numOfInterpolant = np.argwhere(neighbors != 0).shape[0]
            if numOfInterpolant != 0:         
                dLoss_dSamplePoints[i]  = (np.sum(neighbors) / numOfInterpolant) * sampleNormals[i]
            else:
                dLoss_dSamplePoints[i] = 0  
        
        transposeBezierMatrix = np.transpose(bezierMatrix)
        dLoss_dControlPoints  = np.matmul( transposeBezierMatrix, dLoss_dSamplePoints)

        return dLoss_dControlPoints

    def backward(self, usePressureGrads, useViscousGrads):
        leftCurveSamplePoints  = self.leftCurveSamplePoints.detach().numpy()
        rightCurveSamplePoints = self.rightCurveSamplePoints.detach().numpy()
        bezierMatrix = self.bezierMatrix.detach().numpy()

        dLoss_dBinaryMaskPressure = np.zeros((self.gridSize, self.gridSize))
        dLoss_dBinaryMaskViscous  = np.zeros((self.gridSize, self.gridSize))
        
        if usePressureGrads:
            dLoss_dBinaryMaskPressure = dLoss_dBinaryMaskPressure + np.reshape( self.designLoss.binaryMask_pressure.grad.detach().numpy(), (self.gridSize, self.gridSize))
        
        if useViscousGrads:
            dLoss_dBinaryMaskViscous  = dLoss_dBinaryMaskViscous + np.reshape( self.designLoss.binaryMask_visc.grad.detach().numpy(), (self.gridSize, self.gridSize))

        rightCurveNormals, leftCurveNormals = self._calculateBezierCurveNormals(self._parameters["LP"], self._parameters["TP"], self._parameters["CP_Right"], self._parameters["CP_Left"])

        grads_leftCurve_pressure  = self._extract_grads(leftCurveSamplePoints, leftCurveNormals, dLoss_dBinaryMaskPressure, bezierMatrix)
        grads_rightCurve_pressure = self._extract_grads(rightCurveSamplePoints, rightCurveNormals, dLoss_dBinaryMaskPressure, bezierMatrix)
        grads_leftCurve_viscous   = self._extract_grads(leftCurveSamplePoints, leftCurveNormals, dLoss_dBinaryMaskViscous, bezierMatrix)
        grads_rightCurve_viscous  = self._extract_grads(rightCurveSamplePoints, rightCurveNormals, dLoss_dBinaryMaskViscous, bezierMatrix)

        grads_leftCurve_viscous[:,1] = grads_leftCurve_viscous[:,1]*-1
        grads_rightCurve_viscous[:,1] = grads_rightCurve_viscous[:,1]*-1
        
        grads_leftCurve  = np.zeros_like(grads_leftCurve_pressure)
        grads_rightCurve = np.zeros_like(grads_leftCurve_pressure)

        if usePressureGrads:
            grads_leftCurve  += grads_leftCurve_pressure 
            grads_rightCurve += grads_rightCurve_pressure
        
        if useViscousGrads:
            grads_leftCurve  += grads_leftCurve_viscous  
            grads_rightCurve += grads_rightCurve_viscous 

        print("Grads Left Curve")
        print(grads_leftCurve)
        logMsg("Grads Left Curve: \n {}".format(grads_leftCurve))
        print("Grads Right Curve")
        print(grads_rightCurve)
        logMsg("Grads Right Curve: \n {}".format(grads_rightCurve))        

        
        self._parameters["LP"].grad = torch.Tensor( [(grads_leftCurve[0]  + grads_rightCurve[0]) /2 ] )
        self._parameters["TP"].grad = torch.Tensor( [(grads_leftCurve[-1] + grads_rightCurve[-1])/2 ] )

        self._parameters["CP_Right"].grad = torch.Tensor( grads_rightCurve[1:-1] )
        self._parameters["CP_Left"].grad  = torch.Tensor( grads_leftCurve[1:-1] )

if __name__ == "__main__":
    sg = ShapeGenerator(gridSize=128, numOfPointsPerCurve=4, refArea=2000, refWedgeAngle=None)
    sg.importShape("params/dfs_converged.param")
    bezierCurves = sg._getBezierCurves(sg._parameters["LP"], sg._parameters["TP"], sg._parameters["CP_Right"], sg._parameters["CP_Left"])
        
    #shapeSDF = sg._set_SDF_values(bezierCurves)
    
    image = sg.forward()
    printTensorAsImage(sg.getBinaryMaskWithBezierPoints(), "image", display=True)
    
    # sg.exportShape("sphere_shape.param")
    print("Chord Length: " + str(sg.getChordLength()))
    print("Area: " + str(torch.sum(image)))
    sg.exportDatFile()
