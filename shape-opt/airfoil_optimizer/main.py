import torch, random, os, datetime, time
import numpy as np
import traceback

from DesignLoss               import DragLoss
from ShapeGenerator     import ShapeGenerator
from Helper                   import printTensorAsImage
from GDOptimizer              import GDOptimizer
from Helper                   import printTensorAsImage, convertSecond, setupLogger, logMsg, logError, Mode, Solver, Verbose
from Solver                   import OpenFoamSolver, DeepFlowSolver

# Problem Definition
_VEL   =  0.01
_ANGLE = 0.0 * 11.25 /180.0 * np.pi
_VISCOSITY  =  0.00019712 # RN=20 now       ## RN=1 when Lref=0.39424, vel=0.01 and visc = 0.0039424

# Problem Modelling and Solving Tecnique
_RESOLUTION                = 128
_NUM_OF_POINTS_PER_CURVE   = 4
_SOLVER                    = Solver.OpenFoamSolver
_MODEL_PATH                = './saved_models/modelG'
_LOAD_PARAMETERS_FROM      = './params/sphere_shape.param'
_EXPORT_FINAL_PARAMS       = None # './params/final.params'
_AREA_CONSTRAINT           = 2000 #pixels
_WEDGE_ANGLE_CONSTRAINT    = None #102.6
_ITERATON                  = 3001
_DECOUPLE_DRAG_TERMS       = 0    # Decouple every x iterations, 0 means no decoupling

# OpenFOAM Parameters
_UP_SAMPLING_RATE = 1

# Learning Rates and Loss Parameters
_LEARNING_RATE         = 1
_AVERAGE_NUM_GRADIENTS = 1

# Dynamic Learning Rate Parameters
_LR_INCREASE_EVERY         = 25      #iter
_LR_INCREASE_RATE          = 0.99
_MAX_LEARNING_RATE         = 5e-2
_MIN_LEARNING_RATE         = 1e-6

# Debug Parameters
_VERBOSE                   = [True,True,True,True]
_DUMP_MESH                 = True
_DUMP_FREQUENCY            = 1
_DISPLAY                   = False
_SAVE_EVERY                = 1      #iter
_SAVE_FOLDER               = "saved_figures"


###########  BEGINING OF THE PROCESS  ###########
torch.set_default_dtype(torch.float64)

## Random Seed ##
SEED = random.randint(0,2**32-1)
np.random.seed(SEED)

## Velocity Field ##
_VELX =  np.sin(_ANGLE) * _VEL   # towrds right if positive
_VELY =  np.cos(_ANGLE) * _VEL   # downward if positive

ShapeGen        = ShapeGenerator(gridSize =_RESOLUTION, numOfPointsPerCurve = _NUM_OF_POINTS_PER_CURVE, refArea = _AREA_CONSTRAINT, refWedgeAngle = _WEDGE_ANGLE_CONSTRAINT)
Optimizer       = GDOptimizer(model=ShapeGen, lr=_LEARNING_RATE, verbose=_VERBOSE[1], maxLearningRate=_MAX_LEARNING_RATE, minLearningRate=_MIN_LEARNING_RATE, takeAverageGrad=_AVERAGE_NUM_GRADIENTS)

if _SOLVER == Solver.OpenFoamSolver:
    PressureSolver  = OpenFoamSolver(_VELX, -1*_VELY, _RESOLUTION, verbose=_VERBOSE[2], upSamplingRate=_UP_SAMPLING_RATE)
elif _SOLVER == Solver.DeepFlowSolver:
    PressureSolver  = DeepFlowSolver(_VELX, -1*_VELY, logStates=_VERBOSE[2], modelPath=_MODEL_PATH)

Criterion = DragLoss(velx=_VELX, vely=_VELY, logStates=_VERBOSE[2], verbose=True, model=ShapeGen, solver=PressureSolver.pressureSolver, viscosity=_VISCOSITY)

def main():
    ShapeGen.setDesignLoss(Criterion)

    if not os.path.exists(_SAVE_FOLDER):
        os.makedirs(_SAVE_FOLDER)

    if _LOAD_PARAMETERS_FROM:
        ShapeGen.importShape(_LOAD_PARAMETERS_FROM)

    loss_arr = []
    decoupling_toggle_flag = True
    for iter in range(_ITERATON):
        image = ShapeGen.forward()     
        loss  = Criterion(binaryMask = image)
        loss.backward()

        if _DECOUPLE_DRAG_TERMS == 0:
            ShapeGen.backward(usePressureGrads=True, useViscousGrads=True)
        else:
            if iter % _DECOUPLE_DRAG_TERMS == 0: 
                decoupling_toggle_flag = not decoupling_toggle_flag
            ShapeGen.backward(usePressureGrads=decoupling_toggle_flag, useViscousGrads=not decoupling_toggle_flag)

        if iter % _SAVE_EVERY == 0:
            printTensorAsImage(ShapeGen.getBinaryMaskWithBezierPoints(), _SAVE_FOLDER + "/ImageAirfoil_{}".format(iter), display=_DISPLAY)

        loss_arr.append(loss.detach().numpy())
        logMsg("\tDesign Loss: {} || iter: {}".format( loss.detach().numpy(), iter), std=True)
        logMsg(40*'-', std=True)

        Optimizer.step()
        Optimizer.zero_grad()

        if iter % _LR_INCREASE_EVERY == 0:
            Optimizer.setLearningRate(Optimizer.lr*_LR_INCREASE_RATE)
    
    if _EXPORT_FINAL_PARAMS:
        ShapeGen.exportShape(_EXPORT_FINAL_PARAMS)

if __name__ == '__main__':
    start = time.time()
    try:
        setupLogger('process.log')
        logMsg("###  Process Start With  ###")
        logMsg("Problem parameters: \n\t\t\t\t\tResolution: {} \n\t\t\t\t\tNumber of Points Per Curve: {}".format(_RESOLUTION, _NUM_OF_POINTS_PER_CURVE))
        logMsg("VEL X : {} \n\t\t\t\t\tVEL_Y : {}".format(_VELX, _VELY))
        logMsg("Using {} as the shape generator and {} as the solver.".format(type(ShapeGen).__name__, type(PressureSolver).__name__))
        if _SOLVER == Solver.OpenFoamSolver: logMsg("Up-sampling rate for the solver is {}.".format(_UP_SAMPLING_RATE))
        logMsg("Learning rate is {} for width".format(_LEARNING_RATE))
        logMsg("On {}".format(str(datetime.datetime.now())))
        main()
    except Exception as e:
        logError("\tUnhandeled exception has happened: \n\t\t\t\t\t\t{}".format(str(e)))
        traceback.print_exc()
    end = time.time()
    logMsg("\tExecution time: {}".format(convertSecond(end - start)))
