ITER_STA=00
ITER_END=201
RATE_DEF=1
SIGN = 1 # searching direction
CFL_DEF = 0.8

#REINIT="NONE"
#REINIT="CLASSIC" #"FMM"  # Fast marching method
REINIT="FMM"  # Fast marching method
BOTH_FMM = False # have to do this when Re is high
#REINIT_CLASSIC=True # Reinitialisation method
#N_STEP_REINIT= 50
FMM_ORDER=2

RESTART =False
HISTORY_CLEAR = False
RESTART_UPSAMPLING = False
scale_factor_input =.5 #.25 #0.666666667 #1.333334

DYN_DT = True


VISC_DT = False
NWRITE  = 10
NPLOT = 10
AREA_CONSTRAINT = True
AREA_CHECK = 1
BARYCENTER_CONSTRAINT = True
N_BARY= 5

BINARIZE= True
# lw: when BINARIZE=True, mu_coeff2 should be at least 5x bigger than soft one in the case of res=96; coarser mesh needs bigger mu
NOTRUN = False
TEST_DEBUG = False
DRAG_WEIGHT = 1.

xfactor = 1. # Hierachy
res = 128

#coeff = res/2
gridSize = 2./(res-1)
#coeff_eps = 1.5*gridSize/2.
#coeff = 1./coeff_eps
coeff = 1./gridSize
#restFile = "naca0040.pt"
restFile = "phi.pt"
historyFile = "history.npz"
#mu_coeff2= 0
#mu_coeff2= 2.e-2 #5.e-3 #is the one used in the case: 160 steps CFL=0.8 resolution=128, Re=10
mu_coeff2= 0.001 #0.64*gridSize #0.01 # 1.25/coeff #0.64/coeff # smaller mu when resolution is high

LCM_TERM= False
mu_coeff_grad= 0.01

r0 = 0.39424
fsX = 0.01 #0.4e-3
fsY = 0.0
#viscosity = 0.00019712 #0.0078848 #0.0019712
viscosity = 0.0078848 #0.0019712

#UPWIND_SECOND_ORDER = True
#refArea = 0.08339743045984155 #np.pi*r0*r0
refArea = 3.1415926535897932384626*r0*r0
#refArea = 0.268269781721831

projArea = 2*r0

force_calc_old= False
#KAPPA_SMOOTH="SMOOTH_KMAX"
#KAPPA_SMOOTH="SMOOTH_CONVEX"
KAPPA_SMOOTH="NONE"
#KAPPA_SMOOTH="SMOOTH_NAIVE"





channelExpo=7
modelPath='../../models/dataset-ranged-400'
modelName='modelG_dataset-ranged-400'
