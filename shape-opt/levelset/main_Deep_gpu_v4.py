

# Python packages
import random, os, sys, datetime
import time  as sys_time
import numpy as np
from matplotlib import pyplot as plt
import skfmm
#import  importlib.util
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))
myFolder = os.getcwd()
from scipy import spatial
from skimage.util.shape import view_as_windows
from scipy import interpolate
import pickle


# Pytorch imports
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.modules.loss as Loss 
import torch.nn.functional as F

# Helpers Import
import Operators
import Constants
from LevelSet import *

from torch.utils.data         import DataLoader
from data.utils               import makeDirs
from data.asymmetricDataGen   import *
from data.dataGen             import genMesh, runSim, processResult, outputProcessing,interpolateInsideReverse, interpolateInside, correctMinorDiff
from train.utils              import *
from airfoil_optimizer.Helper import *
from airfoil_optimizer.DesignLoss_old              import * #calculateDragLift, calculateDragLift_visc
from airfoil_optimizer.DesignLoss_modified              import calculateDragLift_phi, calculateDragLift_visc_phi
#--------- Project Imports ----------#
from train.DfpNet             import TurbNetG
from train.dataset            import TurbDataset

def main():
    cuda = torch.device('cuda') # Default CUDA device
    upstreamVel = Velocity(fsX,fsY)
    print(upstreamVel.VELPACK.VISCOSITY)
    print(myFolder)
    upstreamVel.updateViscosity(viscosity)
    print("viscosity value updated: ",upstreamVel.VELPACK.VISCOSITY)
    velMag = fsX**2+fsY**2
    dynHead = .5*velMag
    velMag = math.sqrt(velMag)
    print("Ref. Len. Diameter= ", r0*2, "; Re_D =", 2.0*velMag*r0/viscosity)
    if REINIT=="CLASSIC": #"FMM"  # Fast marching method
        print("Reinit. n_steps:", N_STEP_REINIT)
   
    if force_calc_old== True:
        print("Old implementation for the force calc.")
    else:
        print("New implementation for the force calc.")
    
    
    with open(modelPath+'/max_inputs.pickle', 'rb') as f: max_inputs = pickle.load(f)
    f.close()
    with open(modelPath+'/max_targets.pickle', 'rb') as f: max_targets = pickle.load(f)
    f.close()
    print("## max inputs  ##: ",max_inputs) 
    print("## max targets ##: ",max_targets) 
    
    
    ###############################################################################
    if RESTART:
        phi = torch.load(restFile)
        X, Y = np.meshgrid(np.linspace(-1,1,res), np.linspace(-1,1,res))
        XG, YG = np.meshgrid(np.linspace(-1-gridSize,1+gridSize,res+2), np.linspace(-1-gridSize,1+gridSize,res+2))
        if not HISTORY_CLEAR:
            tempfile = np.load(historyFile)
            time = tempfile['arr_0'].tolist()[:-1]  # not this means it doesn't include [-1]
                
            history_drag_pres = tempfile['arr_1'].tolist()[:-1]
            history_drag_visc = tempfile['arr_2'].tolist()[:-1]
            if time[-1]+1 != ITER_STA:
                print("Should start from", time[-1]+1, "... CHECK ITER_STA!!!!")
                exit()
        
    elif RESTART_UPSAMPLING:
        print("Upsampling restart mode: +++++++++++++++++++++++")
        phi = torch.load(restFile)
        res_old = phi.size()[-1]
        print("Old phi dim.:", res_old,"x",res_old)
        round_res_old = round(res/scale_factor_input)
        print("Round new phi dim.:", round_res_old,"x",round_res_old)
        phi = phi.view(1,1,round_res_old,round_res_old)
        upSampling = torch.nn.UpsamplingBilinear2d(size=None, scale_factor=scale_factor_input) # be careful the result will be similar to an "int()"
        # e.g. from 96 to 128, the "scale_factor_input" should be 1.333333334 not 1.333333, because 96x1.33333=int(127.9968)=127 not 128!!!!
        res_new = phi.size()[-1]
        print("New phi dim.:", res_new,"x",res_new)
        print("Check consistence, current dim:", res,"x",res)
        print(upSampling(phi).shape)
        phi = upSampling(phi).view(res,res)


        X, Y = np.meshgrid(np.linspace(-1,1,res), np.linspace(-1,1,res))
        XG, YG = np.meshgrid(np.linspace(-1-gridSize,1+gridSize,res+2), np.linspace(-1-gridSize,1+gridSize,res+2))
        if not HISTORY_CLEAR:
            tempfile = np.load(historyFile)
            time = tempfile['arr_0'].tolist()[:-1]  # not this means it doesn't include [-1]
                
            history_drag_pres = tempfile['arr_1'].tolist()[:-1]
            history_drag_visc = tempfile['arr_2'].tolist()[:-1]
            if time[-1]+1 != ITER_STA:
                print("Should start from", time[-1]+1, "... CHECK ITER_STA!!!!")
                exit()
        

    else:
        X, Y = np.meshgrid(np.linspace(-1,1,res), np.linspace(-1,1,res))
        XG, YG = np.meshgrid(np.linspace(-1-gridSize,1+gridSize,res+2), np.linspace(-1-gridSize,1+gridSize,res+2))
        WITH_SHAPE_TEST = False
        WITH_NOISE = False  
        theta                      = np.arctan2(Y, X)
        noise = .025*np.cos(theta*8)*r0
        noise1 = .25*np.cos(theta*1 + np.pi*0.5)*r0*2
        noise2 = 0.2*np.cos(theta*1)
        
        if WITH_SHAPE_TEST:
            phi = (X)**2+(Y)**2/0.25 - np.power(r0 + noise1 + noise2,2.)
        elif WITH_NOISE: 
            phi = (X)**2+(Y)**2 - np.power(r0 + noise,2.)
        else:
            #phi = (Y)**2/0.11730062024**2+(X)**2/0.53281900696**2 - 1 #np.power(r0,2.) # for rugby shape test (r0=.5)
            phi = (X)**2+(Y)**2 - np.power(r0,2.)

        phi = skfmm.distance(phi, dx=gridSize)
        phi = torch.from_numpy(phi).double()
    ###############################################################################
    X_cuda = torch.from_numpy(X).double().cuda()
    Y_cuda = torch.from_numpy(Y).double().cuda()
    print("Artificial viscosity:", mu_coeff2)
    phi = Variable(phi.cuda(), requires_grad=True)
    phi_old = torch.zeros_like(phi).cuda()
    if (not RESTART) and (not RESTART_UPSAMPLING) or HISTORY_CLEAR:
        time=[]
        history_drag_pres=[]
        history_drag_visc=[]

    dt = 0.001
    netG = TurbNetG(channelExponent=channelExpo)
    netG.load_state_dict( torch.load(modelPath+'/'+modelName, map_location='cpu') )
    #if torch.cuda.is_available:
    netG.cuda()
    netG.eval()

    start_time_main_loop = sys_time.time()
    for it in range(ITER_STA, ITER_END):
        print("iteration:", it)
        #phi = phi.cuda() 
        if AREA_CONSTRAINT and it%AREA_CHECK==0:
            test_phi = phi.data

            offset = satisfy_area_constraint(test_phi, gridSize, refArea, AREA_LOSS_PLOT=False) #, BINARIZE=BINARIZE) # want to make it smoother
            phi.data = phi.data.add(offset) # Be careful "detach" and should use phi.data, possibly because now phi has both "data" and "grad"


        binaryMask = calc_binaryMask(phi, BIN=BINARIZE)
       
        if not force_calc_old:
            gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, meanMask = calc_gradient(phi, gridSize)
            deltaMask = calc_deltaMask(phi, BIN=BINARIZE, OFFSET=0*gridSize)
            print(torch.sum(deltaMask.mul(gradient_abs)).item()*gridSize*gridSize)

    ###################################################################################################    
        cs = plt.contour(X, Y, phi.detach().cpu().numpy(), [0], colors='black', linewidths=(0.5))
        p_curve = cs.collections[0].get_paths()[0]
        v = p_curve.vertices
        x_curve = v[:,0]
        y_curve = v[:,1]
        writePointsIntoFile(str(it), v, "SQ_"+str(it)+".dat")
        axialChordLength = max(x_curve)-min(x_curve)
        verticalHeight = max(y_curve)-min(y_curve)
        print("Axial Chord Length: ", axialChordLength, "Asp. Ratio", axialChordLength/verticalHeight)
    ###################################################################################################    
    # lc: copied from "Solver.py" class Deep Flow Solver def pressureSolver
        
        v_norm = (fsX**2+fsY**2)**0.5  # v_norm: dimensional!

        Velx = fsX /max(max_inputs[0], 1e-20)
        Vely = fsY /max(max_inputs[1], 1e-20) #/100 #.... fsX=10

        print("v_norm:", v_norm) 


        binaryMaskInv = torch.transpose(binaryMask.data, 0, 1) #.detach().cpu().numpy()).transpose()
        channelfsX    = (1-binaryMaskInv) * Velx
        channelfsY    = (1-binaryMaskInv) * Vely
        input_gpu = torch.cat( (channelfsX, channelfsY, binaryMaskInv) )
        input_gpu = input_gpu.view((1, 3, binaryMaskInv.shape[0],binaryMaskInv.shape[1]))


        fullSolution = netG(input_gpu)

        pressure  = torch.transpose(fullSolution[0, 0, :], 0, 1) * max_targets[0] 
        velocityX = torch.transpose(fullSolution[0, 1, :], 0, 1) * max_targets[1] 
        velocityY = torch.transpose(fullSolution[0, 2, :], 0, 1) * max_targets[2] 
        

        if True:
            dataCopy = pressure.detach().cpu().numpy()
            dataCopy[np.where(binaryMask.detach().cpu().numpy()==1)] = np.nan
            
            x = np.arange(0, dataCopy.shape[1])
            y = np.arange(0, dataCopy.shape[0])
            array = np.ma.masked_invalid(dataCopy)
            xx, yy = np.meshgrid(x, y)
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]

            pressure = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="nearest")
            pressure = torch.from_numpy(pressure).cuda()

        #lc##### 
        os.chdir(myFolder)

        pressure  *= v_norm**2
        velocityX *= v_norm 
        velocityY *= v_norm # v_norm should a dimensional value!!!!!


        if force_calc_old:
            drag, lift = calculateDragLift(binaryMask, binaryMask, pressure, upstreamVel.passTupple(), False, res)
            drag_visc, lift_visc = calculateDragLift_visc(binaryMask, binaryMask, velocityX, velocityY, upstreamVel.passTupple(), False)
        else:
            drag, lift = calculateDragLift_phi(binaryMask, binaryMask, deltaMask, normX, normY, gradient_abs, pressure, upstreamVel.passTupple(), False, res)
            drag_visc, lift_visc = calculateDragLift_visc_phi(binaryMask, binaryMask, deltaMask, normX, normY, gradient_abs, velocityX, velocityY, upstreamVel.passTupple(), False, res)

        loss = (DRAG_WEIGHT*drag + drag_visc)/dynHead/projArea 
        print("loss:", loss.item(),"pres. drag:", drag.item()/dynHead/projArea, "visc. drag:", drag_visc.item()/dynHead/projArea, "Ratio:", drag.item()/drag_visc.item())
        loss.backward(retain_graph=True)

        ##### later will put this into a subroutine ####
        vn = phi.grad 

        vel_max = torch.max(vn.data).detach().cpu().numpy()
        vel_min = torch.min(vn.data).detach().cpu().numpy()
        vel_max = max(abs(vel_min), abs(vel_max))
        cfl = vel_max*dt/gridSize
        if DYN_DT:
            dt = gridSize/vel_max*CFL_DEF
        if VISC_DT:
            factor_visc = 0.85
            dt_visc = 0.5*gridSize**2/max(mu_coeff2, 1e-12) * factor_visc
            print("++++ CFL & vel_max,min & time-step ++++,", cfl, vel_max, vel_min, dt, dt_visc)
            #dt = min(dt, dt_visc) 
            dt = 1./(1./dt + 1./dt_visc) 
        print("time-step ++++ & CFL,", dt, cfl)



        phi_old = torch.zeros_like(phi)

        ######################################
        # Second order RK
        ######################################

        gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, meanMask = calc_gradient(phi, gridSize)
        # Convection term Velocity version:
        #u = SIGN*phi.grad.mul(normX)
        #v = SIGN*phi.grad.mul(normY)
        #k1 = calc_rhs(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize)
        k1 = calc_rhs_sethian(SIGN*vn, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, meanMask, gridSize)

        phi_old.data=phi_old.data.add(phi) 

        phi.data = phi.data.add( k1.data.mul(dt) )


        #########################################

        gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, meanMask = calc_gradient(phi, gridSize)

        #u = SIGN*phi.grad.mul(normX)
        #v = SIGN*phi.grad.mul(normY)
        #k2 = calc_rhs(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize)
        k2 = calc_rhs_sethian(SIGN*vn, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, meanMask, gridSize)

        phi.data = phi_old.data.add( 0.5*(k1.data.mul(dt) + k2.data.mul(dt)) )


        
        phi.grad.data.zero_()

        if REINIT=="FMM":
            phi = skfmm.distance(phi.detach().cpu().numpy(), dx=gridSize, self_test=True, order=FMM_ORDER, narrow=0.0, periodic=True)
            phi = torch.from_numpy(phi).double()
            phi = Variable(phi.cuda(), requires_grad=True)
        elif REINIT=="CLASSIC":
##########################################################
###############    Re-initialization        ############## do we want to put the reinit procedure before the loss calc?
##########################################################
            dtau = 0.1*gridSize
            phi_old = torch.zeros_like(phi)
            phi_old.data=phi_old.data.add(phi) 
            s_function = calc_s_function(phi_old, eps_s_function=gridSize)
            # we choose these values for dtau and eps_s_function according to the ref. paper: Sussman, Smereka & Osher, JCP vol. 114, 146-159 (1994)
            for it_sub in range(N_STEP_REINIT):
                # gradients, normals... need to be updated every iteration
                gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, _  = calc_gradient(phi, gridSize, UPWIND_SECOND_ORDER=False)
                #u = s_function.mul(normX)
                #v = s_function.mul(normY)
                #k1 = reinit_rhs(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize, s_function)
                k1 = reinit_rhs_sethian(s_function, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize, s_function)
# save f    or step 2:
                phi_old = torch.zeros_like(phi)
                phi_old.data=phi_old.data.add(phi) 

                phi.data = phi.data.add( k1.data.mul(dtau) )


# step 2    :
                gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, _  = calc_gradient(phi, gridSize, UPWIND_SECOND_ORDER=False)
                #u = s_function.mul(normX)
                #v = s_function.mul(normY)
                #k2 = reinit_rhs(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize, s_function)
                k2 = reinit_rhs_sethian(s_function, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize, s_function)
#########################################################
                phi.data = phi_old.data.add( 0.5*(k1.data.mul(dtau) + k2.data.mul(dtau)) )
#########################################################
            if BOTH_FMM:
                phi = skfmm.distance(phi.detach().cpu().numpy(), dx=gridSize, order=FMM_ORDER)
                phi = torch.from_numpy(phi).double()
                #phi = Variable(phi, requires_grad=True)    
                phi = Variable(phi.cuda(), requires_grad=True)
#########################################################


#########################################################
##############    Shift LSF back to Center   ############  This is not a mandotory procedure!!!
#########################################################

        if BARYCENTER_CONSTRAINT:
            for i_bary in range(N_BARY):
                binaryMask = calc_binaryMask(phi, BIN=BINARIZE)
                # Let's first calculate the bary center
                xc_pred = torch.mul(binaryMask, X_cuda)
                xc_pred = xc_pred.sum() * binaryMask.sum().pow(-1.)
                yc_pred = torch.mul(binaryMask, Y_cuda)
                yc_pred = yc_pred.sum() * binaryMask.sum().pow(-1.)
                print("pred. xc & yc:", xc_pred.item(), yc_pred.item()) 
                # then let's calculte the velocity which can move the LSF back to the origin with one time step
                # Convection term Velocity version:
                u = -xc_pred.expand(res,res)/dt/(N_BARY-i_bary)
                v = -yc_pred.expand(res,res)/dt/(N_BARY-i_bary)
             
        
        
                phi_old = torch.zeros_like(phi)
        
                ######################################
                # Second order RK
                ######################################
        
                gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, _  = calc_gradient(phi, gridSize)
                k1 = calc_rhs(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize)
                #k1 = calc_rhs_central(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize)
        
                phi_old.data=phi_old.data.add(phi) 
        
                phi.data = phi.data.add( k1.data.mul(dt) )
        
        
                
                # then let's calculte the velocity which can move the LSF back to the origin with one time step
        
                gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, _  = calc_gradient(phi, gridSize)
        
                k2 = calc_rhs(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize)
                #k2 = calc_rhs_central(u, v, gradPlusX, gradPlusY, gradMinusX, gradMinusY, gradientX, gradientY, gradient_abs, normX, normY, lap_phi, k_curvature, gridSize)
        
                phi.data = phi_old.data.add( 0.5*(k1.data.mul(dt) + k2.data.mul(dt)) )

            binaryMask = calc_binaryMask(phi, BIN=BINARIZE)
            # Let's first calculate the bary center
            xc_pred = torch.mul(binaryMask, X_cuda)
            xc_pred = xc_pred.sum() * binaryMask.sum().pow(-1.)
            yc_pred = torch.mul(binaryMask, Y_cuda)
            yc_pred = yc_pred.sum() * binaryMask.sum().pow(-1.)
            print("after", i_bary+1, "iterations, pred. xc & yc:", xc_pred.item(), yc_pred.item()) 

        time.append(it)
        history_drag_pres.append(drag.item())
        history_drag_visc.append(drag_visc.item())




        if it>ITER_STA and it%NWRITE==0:
            torch.save(phi, restFile+".saved")
            print("Save phi at time-step:", it)
            np.savez(historyFile, time, history_drag_pres, history_drag_visc)


        phi_old = torch.zeros_like(phi)
    
    end_time_main_loop = sys_time.time()
    print("Main loop execution time: {}".format(convertSecond(end_time_main_loop - start_time_main_loop)))
if __name__ == '__main__':
    start = sys_time.time()
    main()
    end = sys_time.time()

    print("Execution time: {}".format(convertSecond(end - start)))

