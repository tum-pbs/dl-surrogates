################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import os, math, uuid, sys, random
import numpy as np
from scipy import spatial
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
#from Helper import printTensorAsImage
from scipy import interpolate

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))

from data import utils

class MODE:
    DATAGEN = 1
    CUSTOM  = 2

mode              = MODE.CUSTOM
samples           = 200           # no. of datasets to produce
resolution        = 128           # solution resolution
oversamplingRate  = 1             # oversampling rate (powers of two)
airfoil_database  = "./database/square_airfoil_database/"
real_airfoil_db   = "./database/airfoil_database/"
test_airfoil_db   = "./database/test/"
output_dir        = "./train_sets/train_square/"
real_airfoil_out  = "./train_sets/train_set/"
randomVelocity    = False

if randomVelocity:
    freestream_angle  = math.pi / 8.  # -angle ... angle
    freestream_length = 10.           # len * (1. ... factor)
    freestream_length_factor = 10.    # length factor
else:
    freestream_length = 0.01
    freestream_angle  = math.pi / 16 * 7

def genMesh(airfoilFile):
    utils.makeDirs( ["./constant/polyMesh", "./constant/polyMesh/sets"] )
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)]))<1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        #output += "Point({}) = {{ {}, {}, 0.00000000, 0.01}};\n".format(pointIndex, ar[n][0], ar[n][1])
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
        #output += "Point({}) = {{ {}, {}, 0.00000000, 0.0025}};\n".format(pointIndex, ar[n][0], ar[n][1])
        #output += "Point({}) = {{ {}, {}, 0.00000000, 0.001}};\n".format(pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                outFile.write(line)

    if os.system("/home/liwei/Codes/gmsh-4.4.1-Linux64/bin/gmsh airfoil.geo -3 -o airfoil.msh -format msh2 -option gmsh_options.c > /dev/null") != 0:
        print("error during mesh creation!")
        return(-1)

    if os.system("/opt/openfoam5/platforms/linux64GccDPInt32Opt/bin/gmshToFoam airfoil.msh > /dev/null") != 0:
        print("error during conversion to OpenFoam mesh!")
        return(-1)

    with open("constant/polyMesh/boundary", "rt") as inFile:
        with open("constant/polyMesh/boundaryTemp", "wt") as outFile:
            inBlock = False
            inAerofoil = False
            for line in inFile:
                if "front" in line or "back" in line:
                    inBlock = True
                elif "aerofoil" in line:
                    inAerofoil = True
                if inBlock and "type" in line:
                    line = line.replace("patch", "empty")
                    inBlock = False
                if inAerofoil and "type" in line:
                    line = line.replace("patch", "wall")
                    inAerofoil = False
                outFile.write(line)
    os.rename("constant/polyMesh/boundaryTemp","constant/polyMesh/boundary")

    return(0)

def genPointCloud(res, oversamplingRate):
    _res = res*oversamplingRate
    output = ""
    for y in range(_res):
        for x in range(_res):
            #xf = (x / _res - 0.5) * 2
            #yf = (y / _res - 0.5) * 2
            xf = (x / (_res-1) - 0.5) * 2
            yf = (y / (_res-1) - 0.5) * 2
            #output += "({} {} 0.5)\n".format(xf, yf)
            output += "({} {} 0.05)\n".format(xf, yf)
            
    with open("system/internalCloud_template", "rt") as inFile:
        with open("system/internalCloud", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                outFile.write(line)
    
def runSim(freestreamX, freestreamY, user_viscosity=1e-5, res=128, oversamplingRate=1):
    genPointCloud(res, oversamplingRate)
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)

    with open("transportProperties_template", "rt") as inFile:
        with open("constant/transportProperties", "wt") as outFile:
            for line in inFile:
                line = line.replace("USER_VISCOSITY", "{}".format(user_viscosity))
                outFile.write(line)

    os.system("./Allclean")
    os.system("./Allclean_proc")
    os.system("decomposePar > 1_decomposer.log")
    os.system("/usr/bin/mpirun -n 9 simpleFoam -parallel > 2_foam.log")
    os.system("reconstructPar -time 6000 > 3_reconstructor.log")


def downsample(input, downsampleRate):
    _dr = downsampleRate
    output = []
    # Channel number two contains binary mask where inside is specified with 1 whereas outside with 0
    blocks_denum          = np.sum(view_as_windows(input[2], (_dr, _dr), step=_dr ), axis=(2,3))
    modified_blocks_denum = np.where(blocks_denum == 0, (_dr*_dr), blocks_denum)

    for i in range(input.shape[0]):
        blocks = view_as_windows(input[i], (_dr, _dr), step=_dr )
        output.append( np.divide( np.sum(blocks, axis=(2,3)) , modified_blocks_denum) )

    return np.asarray(output)

def processResult(freestreamX, 
                  freestreamY, 
                  pfile='OpenFOAM/postProcessing/internalCloud/6000/cloud_p.xy', 
                  ufile='OpenFOAM/postProcessing/internalCloud/6000/cloud_U.xy', 
                  res=128, 
                  oversamplingRate=1):
    # output layout channels:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output
    _res     = res*oversamplingRate
    npOutput = np.zeros((6, _res, _res))

    ar = np.loadtxt(pfile)
    curIndex = 0

    for y in range(_res):
        for x in range(_res):
            #xf = (x / _res - 0.5) * 2
            #yf = (y / _res - 0.5) * 2
            xf = (x / (_res-1) - 0.5) * 2
            yf = (y / (_res-1) - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[3][x][y] = ar[curIndex][3]
                curIndex += 1
                # fill input as well
                npOutput[0][x][y] = freestreamX
                npOutput[1][x][y] = freestreamY
            else:
                npOutput[3][x][y] = 0
                # fill mask
                npOutput[2][x][y] = 1.0

    ar = np.loadtxt(ufile)
    curIndex = 0

    for y in range(_res):
        for x in range(_res):
            #xf = (x / _res - 0.5) * 2
            #yf = (y / _res - 0.5) * 2
            xf = (x / (_res-1) - 0.5) * 2
            yf = (y / (_res-1) - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[4][x][y] = ar[curIndex][3]
                npOutput[5][x][y] = ar[curIndex][4]
                curIndex += 1
            else:
                npOutput[4][x][y] = 0
                npOutput[5][x][y] = 0
                
    return downsample(npOutput, oversamplingRate)

def interpolateMinorDiff(source, target, interpolationPoints, iteration=1):
    for _ in range(iteration):
        for ind in interpolationPoints:
            neighbors  = source[ind[0]-1:ind[0]+2,ind[1]-1:ind[1]+2]
            numOfInterpolant = np.argwhere(neighbors != 0).shape[0]
            if numOfInterpolant > 0:
                target[tuple(ind)] = np.sum(neighbors)/numOfInterpolant

def correctMinorDiff(binaryMask, npOutput, interpolateInterior=False, numOfIter=1):
    binaryMask = np.flipud(binaryMask).transpose() # convert to openfoam domain
    # Get Open Mask From Open Foam and Invert
    openFoamMask = np.where(npOutput[2]==0, 1, 0)
    # Get pixels where corrections will be made
    diff = binaryMask - openFoamMask if not interpolateInterior else -1*openFoamMask

    pointsToAssignZero  = np.argwhere(diff > 0)
    pointsToInterpolate = np.argwhere(diff < 0)
    npOutputCopy        = np.copy(npOutput) if not interpolateInterior else npOutput

    for chn in range(npOutputCopy.shape[0]):
        for ind in pointsToAssignZero:
            npOutputCopy[chn][tuple(ind)] = 0

        interpolateMinorDiff(npOutput[chn], npOutputCopy[chn], pointsToInterpolate, numOfIter)

    return npOutputCopy

def correctBinaryMask(npArray):
    npCopy = np.copy(npArray)
    npCopy[2] = np.where(npArray[2]==0, 1, 0)
    return npCopy


#def interpolateInside(data, channels, method, xx, yy):
def interpolateInsideReverse(data, channels, method):
    dataCopy = np.copy(data)
    #import pdb; pdb.set_trace()

    binaryMask =  dataCopy[2]
    dataCopy   = np.copy(-dataCopy)
    for chn in channels: # [3,4,5]

        #dataCopy[chn][np.where(dataCopy[chn]==0)] = np.nan
        dataCopy[chn][np.where(binaryMask==1)] = np.nan
        
        x = np.arange(0, dataCopy[chn].shape[1])
        y = np.arange(0, dataCopy[chn].shape[0])
        array = np.ma.masked_invalid(dataCopy[chn])
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]

        dataCopy[chn]= interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method=method)
    
    return data*2 + dataCopy

def interpolateInside(data, channels, method):
    dataCopy = np.copy(data)
    #import pdb; pdb.set_trace()

    binaryMask = dataCopy[2]
    for chn in channels: # [3,4,5]

        #dataCopy[chn][np.where(dataCopy[chn]==0)] = np.nan
        dataCopy[chn][np.where(binaryMask==1)] = np.nan
        
        x = np.arange(0, dataCopy[chn].shape[1])
        y = np.arange(0, dataCopy[chn].shape[0])
        array = np.ma.masked_invalid(dataCopy[chn])
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]

        dataCopy[chn]= interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method=method)
    
    return dataCopy

def outputProcessing(basename, 
                     freestreamX, 
                     freestreamY,
                     binaryMask=None,
                     dataDir=output_dir, 
                     pfile='OpenFOAM/postProcessing/internalCloud/6000/cloud_p.xy', 
                     ufile='OpenFOAM/postProcessing/internalCloud/6000/cloud_U.xy', 
                     res=128,
                     oversamplingRate=1, 
                     imageIndex=0): 
    utils.makeDirs(["./data_pictures", output_dir])
    npOutput = processResult(freestreamX, freestreamY, pfile=pfile, ufile=ufile, res=res, oversamplingRate=oversamplingRate)

    ##npOutput = interpolateInside(npOutput, [3,4,5])

    utils.saveAsImage('data_pictures/pressure_%04d.png'%(imageIndex), npOutput[3])
    utils.saveAsImage('data_pictures/velX_%04d.png'  %(imageIndex), npOutput[4])
    utils.saveAsImage('data_pictures/velY_%04d.png'  %(imageIndex), npOutput[5])
    utils.saveAsImage('data_pictures/inputX_%04d.png'%(imageIndex), npOutput[0])
    utils.saveAsImage('data_pictures/inputY_%04d.png'%(imageIndex), npOutput[1])

    #fileName = dataDir + str(uuid.uuid4()) # randomized name
    fileName = dataDir + "%s_%d_%d" % (basename, int(freestreamX*100), int(freestreamY*100) )
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, a=npOutput)

    if binaryMask is not None:
        return correctBinaryMask(correctMinorDiff(binaryMask, npOutput))
    return correctBinaryMask(npOutput)

def dataGenMode():
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    print("Seed: {}".format(seed))
    
    files = os.listdir(airfoil_database)
    files.sort()
    if len(files)==0:
        print("error - no airfoils found in %s" % airfoil_database)
        exit(1)

    # main
    for n in range(samples):
        print("Run {}:".format(n))

        fileNumber = np.random.randint(0, len(files))
        basename = os.path.splitext( os.path.basename(files[fileNumber]) )[0]
        print("\tusing {}".format(files[fileNumber]))
        
        if randomVelocity:
            length = freestream_length * np.random.uniform(1.,freestream_length_factor) 
            angle  = np.random.uniform(-freestream_angle, freestream_angle) 
        else:
            length  = freestream_length
            angle   = freestream_angle
        
        fsX =  math.sin(angle) * length
        fsY = -math.cos(angle) * length 

        print("\tUsing len %5.3f angle %+5.3f " %( length,angle )  )
        print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY))

        os.chdir("./OpenFOAM/")
        if genMesh("../" + airfoil_database + files[fileNumber]) != 0:
            print("\tmesh generation failed, aborting");
            os.chdir("..")
            continue
        
        runSim(fsX, fsY, resolution, oversamplingRate)
        os.chdir("..")

        outputProcessing(basename, fsX, fsY, res=resolution, oversamplingRate=oversamplingRate ,imageIndex=n)
        print("\tdone")

def customMode(): 
    if randomVelocity:
        length = freestream_length * np.random.uniform(1.,freestream_length_factor) 
        angle  = np.random.uniform(-freestream_angle, freestream_angle) 
    else:
        length  = freestream_length
        angle   = freestream_angle

    fsX =  math.sin(angle) * length
    fsY = -math.cos(angle) * length 
    basename = "Trial"
    filename = "SQ_7.dat"
    os.chdir("./OpenFOAM/")
    print("Generating Mesh")
    if genMesh("../" + airfoil_database + filename) != 0:
        print("\tmesh generation failed, aborting")
        os.chdir("..")
        return

    print("Running Simulation")    
    runSim(fsX, fsY, resolution, oversamplingRate)
    print("|")
    print("--->Done!")
    os.chdir("..")

    print("Postprocessing is started...")
    outputProcessing(basename, fsX, fsY, res=resolution, oversamplingRate=oversamplingRate ,imageIndex=1)
    print("|")
    print("--->Done!")


def main():
    if mode == MODE.DATAGEN:
        dataGenMode()
    elif mode == MODE.CUSTOM:
        customMode()

if __name__=='__main__':
    main()

'''
def printTensorAsImage(array, name, display=True):
    import matplotlib.pyplot as pyplt
    pyplt.imshow(array)
    pyplt.savefig(name)
    if display:
        pyplt.show()


binaryMask   = np.load('1_GROUND_TRUTH.npz')['a']
npResults    = np.load('2_OF_SOLUTION.npz')['a']
#printTensorAsImage(np.flipud(npResults[3].transpose()), "02_CorrectedBinary")

npResults = correctMinorDiff(binaryMask, npResults)
printTensorAsImage(binaryMask, "01_GroundTruth")
printTensorAsImage(np.flipud(npResults[3].transpose()), "03_CorrectedPressure")
'''
