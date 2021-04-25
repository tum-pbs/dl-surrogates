import math
import numpy as np
import torch
import logging
import matplotlib.pyplot as pyplt

class Mode:
    ShapeGen           = 1
    AsymmetricShapeGen = 2
    RotatingShapeGen   = 3
    BezierShapeGen     = 4
    DebugShapeGen      = 5

class Solver:
    DummySolver        = 1
    OpenFoamSolver     = 2
    DeepFlowSolver     = 3

class Verbose:
    OFF       = [False, False, False, False]
    Level_1   = [True , False, False, False]
    Level_2   = [True , True , False, False]
    Level_3   = [True , True , True , False]
    Level_4   = [True , True , True , True ]
    SHAPE_GEN = [True , False, False, False]
    OPTIMIZER = [False, True,  False, False]
    SOLVER    = [False, False, True , False]
    CRITERION = [False, False, False, True ]

def _getDiscrateSpaceVector(dimension):
    '''
        Generates numpy array with entries starting from 1 to _GRID_SIZE (inclusive)
        with step size "1". This array has entries as much as discrate domain pixels 
    '''
    return np.arange(1, dimension + 1, 1 )

def _getContShapeVector(domain, size):
    '''
        Generates numpy array representing object with in 1D projection
    '''
    return np.concatenate((np.full((domain - size) // 2, 0 ), np.full(size, 1), np.full(domain - size - (domain - size) // 2, 0 )))

def _generateShape(height, width):
    listOfColumns = []
    for element in width:
        listOfColumns.append(height*element)

    binaryMask = torch.stack( listOfColumns, 1)
    
    return binaryMask

def printTensorAsImage(array, name, display=False):
    '''
        Creates output image from numpy or tensor array
        Displays it if display is set to 'True'
    '''
    if type(array).__module__ == torch.__name__:
        numpyOut = array.detach().numpy()
    else:
        numpyOut = array

    np.savez_compressed(name + ".npz", a=numpyOut)
    pyplt.imshow(numpyOut)

    pyplt.savefig(name)
    if display:
        pyplt.show()

def combination(n, r):
    '''
        Calculates and returns C(n, r), i.e. combination of n with r.
    '''
    return int((math.factorial(n)) / ((math.factorial(r)) * math.factorial(n - r)))

def pascalsTriangle(row, normalize=True, centerBiased=True):
    '''
        Generates pascal triangle with specified number of rows and returns the last row.
        It can be also specified if values should be normalized to make sum of elements in the row sould be '1' or not
        Additionaly corner values can be trimmed 25% and that amount can be added into the center.
        This function is used when preparing filter kernel for smoother. 
    '''
    result = []
    for count in range(row):
        _row = []
        for element in range(count + 1): 
            _row.append(combination(count, element))
        result.append(_row)
    if centerBiased:
        if row%2 == 1:
            result[-1][0] -= 0.25
            result[-1][-1]-= 0.25
            result[-1][row//2] += 0.50
        else:
            result[-1][0] -= 0.25
            result[-1][-1]-= 0.25
            result[-1][row//2 - 1] += 0.25
            result[-1][row//2    ] += 0.25
    if normalize:
        dnm = sum(result[-1])
        return [nmn / dnm for nmn in result[-1]]
    return result[-1]

def convertSecond(seconds, level=4):
    '''
        Takes seconds as input and returns granuler time values as xx Hours xx Minutes and xx Seconds
    '''
    result = []
    time_intervals = [('days', 86400), ('hours', 3600),
                      ('minutes', 60), ('seconds',  1)]

    for name, count in time_intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:level])

def setupLogger(fileName, ignore=['matplotlib']):
    '''
        Sets up logger with a given file name. 
        Add those modules that needs to be suppressed from logger output into ignore list.
    '''
    logging.basicConfig(filename=fileName,level=logging.DEBUG)
    for module in ignore:
        logging.getLogger(module).setLevel(logging.ERROR)

def logMsg(msg, std=False):
    '''
        Log info
    '''
    logging.info(msg)
    if std: print(msg)

def logError(msg, std=True):
    '''
        Log error
    '''
    logging.error(msg)
    if std: print(msg)
