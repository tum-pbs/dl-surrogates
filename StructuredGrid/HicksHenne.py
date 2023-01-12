import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os, math, uuid, sys, random
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


TFI_EDGE = True
TFI_EDGE_DEBUG = True
CHECK_AIRFOIL = True

liwei_plt_cm = plt.cm.get_cmap("twilight_r")

idim = 257 #65
jdim = 129 #33
jst=41
jfn=217

wake_lower_jst=1
wake_lower_jfn=41
wake_upper_jst=217
wake_upper_jfn=257

imax = idim
jmax = jdim


xmin, xmax = -0.5, 1.5
ymin, ymax = -1, 1
#xmin, xmax = -0.01, 0.01
#ymin, ymax = -0.01, 0.01


def hickshenne(x, xmi, width, SF, mag):
  #x = np.r_[0: 1: 1j*n]
  n = x.shape[0]
  print(x.shape)
  m = np.log(0.5)/np.log(xmi)
  b = np.zeros_like(x)
  for i in range(len(x)):
    temp = 0
    for j in range(len(m)):
      temp += SF*mag[j] * np.power( np.sin(np.pi*np.power(x[i],m[j])), width)
    b[i] = temp
  return b

################################################################################
#
# Function to read grid
#
################################################################################

def convert_center(x, y):

  imax, jmax = x.shape

  xc = np.zeros((imax-1,jmax-1))
  yc = np.zeros((imax-1,jmax-1))
  for j in range(0, jmax-1):
    for i in range(0, imax-1):
      xc[i,j] = 0.25*( 
              x[i,j]+x[i+1,j]+x[i,j+1]+x[i+1,j+1]
              )

  for j in range(0, jmax-1):
    for i in range(0, imax-1):
      yc[i,j] = 0.25*( 
              y[i,j]+y[i+1,j]+y[i,j+1]+y[i+1,j+1]
              )
  return xc, yc


def q_convert_point(q):

  imax_c, jmax_c = q.shape

  q_point = np.zeros((imax_c+1,jmax_c+1))
  #q_wake  = np.zeros((320,1))
  #q_i0    = q[0,:]
  #q_imax  = q[-1,:]
  #q_jmax  = q[:,-1]
  #q_j0    = q[:,0]
  #q_point[0,0] = q[0,0]

  for i in range(1, imax_c):
    for j in range(1, jmax_c):
      q_point[i,j] = 0.25*( 
              q[i,j]+q[i-1,j]+q[i,j-1]+q[i-1,j-1])

    j=0 # at walls ........ need special treatment
    q_point[i,j] = 0.25*( 
            q[i,j]+q[i-1,j]+q[i,j]+q[i-1,j])
#         1         1         0         0         0         1       321         0
#         1         2      2004         0         0       321       705         2
#              TWTYPE        CQ
#                  0.        0.
#         1         3         0         0         0       705      1025         0
#    q_wake[:,0] = 0.5*(q_point[0:320,0] + q_point[704:1024,0])
#    q_point[0:320,0]    = q_wake[:,0]
#    q_point[704:1024,0] = q_wake[:,0]


    j=jmax_c # at walls ........ need special treatment
    q_point[i,j] = 0.25*( 
            q[i,j-1]+q[i-1,j-1]+q[i,j-1]+q[i-1,j-1])

  i=0
  for j in range(1, jmax_c):
    q_point[i,j] = 0.25*( 
              q[i,j]+q[i,j]+q[i,j-1]+q[i,j-1])
  j=jmax_c # at walls ........ need special treatment
  q_point[i,j] = 0.25*( 
          q[i,j-1]+q[i,j-1]+q[i,j-1]+q[i,j-1])

  j=0 # at walls
  q_point[i,j] =  q[i,j]
  j=jmax_c # at walls
  q_point[i,j] =  q[i,j-1]


  i=imax_c
  for j in range(1, jmax_c):
    q_point[i,j] = 0.25*( 
              q[i-1,j]+q[i-1,j]+q[i-1,j-1]+q[i-1,j-1])
  j=jmax_c # at walls ........ need special treatment
  q_point[i,j] = 0.25*( 
          q[i-1,j-1]+q[i-1,j-1]+q[i-1,j-1]+q[i-1,j-1])

  j=0 # at walls
  q_point[i,j] =  q[i-1,j]
  j=jmax_c # at walls
  q_point[i,j] =  q[i-1,j-1]



#  #at wall    
#  q_point[1:,0 ] = q[: , 0]
#  #at exit
#  q_point[0 ,1:] = q[0,  :]
  




  return q_point

def read_grid(fname):

# Open grid file
  f = open(fname)

# Read imax, jmax
# 3D grid specifies number of blocks on top line
  line1 = f.readline()
  flag = len(line1.split())
  if flag == 1:
    threed = True
  else:
    threed = False

  if threed:
    line1 = f.readline()
    imax, kmax, jmax = [int(x) for x in line1.split()]
  else:
    imax, jmax = [int(x) for x in line1.split()]
    kmax = 1

# Read geometry data
  x = np.double(np.zeros((imax,jmax)))
  y = np.double(np.zeros((imax,jmax)))
  if threed:
    for j in range(0, jmax):
      for k in range(0, kmax):
        for i in range(0, imax):
          x[i,j] = float(f.readline())
    for j in range(0, jmax):
      for k in range(0, kmax):
        for i in range(0, imax):
          dummy = float(f.readline())
    for j in range(0, jmax):
      for k in range(0, kmax):
        for i in range(0, imax):
          y[i,j] = float(f.readline())
  else:
    for j in range(0, jmax):
      for i in range(0, imax):
        x[i,j] = float(f.readline())

    for j in range(0, jmax):
      for i in range(0, imax):
        y[i,j] = float(f.readline())

# Print message
  print('Successfully read grid file ' + fname)

# Close the file
  f.close

  return (imax, jmax, kmax, x, y, threed)

################################################################################
#
# Function to read Plot3D function file
#
################################################################################
def read_function_file(fname, imax, jmax, kmax, threed):

# Open stats file
  f = open(fname)

# Read first line to get variables category
  line1 = f.readline()
  varcat = line1[1:].rstrip()

# Second line gives variable names
  line1 = f.readline()
  varnames = line1[1:].rstrip()
  variables = varnames.split(", ")

# Number of variables
  nvars = len(variables)

# Initialize data and skip the next line
  values = np.zeros((nvars,imax,jmax))
  maxes = np.zeros((nvars))*-1000.0
  mins = np.ones((nvars))*1000.0
  line1 = f.readline()

# Read grid stats data, storing min and max
  for n in range(0, nvars):
    if (threed):
      for j in range(0, jmax):
        for k in range(0, kmax):
          for i in range(0, imax):
            values[n,i,j] = float(f.readline())
            if values[n,i,j] > maxes[n]:
              maxes[n] = values[n,i,j]
            if values[n,i,j] < mins[n]:
              mins[n] = values[n,i,j]
    else:
      for j in range(0, jmax):
        for i in range(0, imax):
          values[n,i,j] = float(f.readline())
          if values[n,i,j] > maxes[n]:
            maxes[n] = values[n,i,j]
          if values[n,i,j] < mins[n]:
            mins[n] = values[n,i,j]

# Print message
  print('Successfully read data file ' + fname)

# Close the file
  f.close

  return (varcat, variables, values, mins, maxes)





#imax, jmax, kmax, x_original, y_original, threed = read_grid("original_mesh.p3d")
imax, jmax, kmax, x_original, y_original, threed = read_grid("naca0012_0deg.p3d") # ---- the real original!!!
xs_initial = x_original[jst-1:jfn,0]
ys_initial = y_original[jst-1:jfn,0]




# 
# X = U + V - U*V
# U =  alpha_1^0 * X(xi_1, eta) + alpha_2^0 * X(xi_2, eta)
# alpha_1^0 = (1-xi)
# alpha_2^0 = xi
# V =  beta_1^0 * X(xi, eta_1) + beta_2^0 * X(xi, eta_2)
# beta_1^0 = (1-eta)
# beta_2^0 = eta
#  
#  
# Calculate arc length #
def calcArcLength(x, y):
    
    imax, jmax = x.shape[0], x.shape[1]
    print(imax, jmax)
    arcF  = np.double(np.zeros((imax, jmax)))
    arcG  = np.double(np.zeros((imax, jmax)))
    Pij   = np.double(np.zeros((imax, jmax)))
    xi    = np.double(np.zeros((imax, jmax)))
    eta   = np.double(np.zeros((imax, jmax)))
    
    
    #for j in range(0, jmax):
    #    arcF[:,j]=0
    #    for i in range(1, imax): 
    #        arcF[i,j] = ((x[i,j]-x[i-1,j])**2 + (y[i,j]-y[i-1,j])**2)**0.5
    #    #arcF[i,:] = arcF[i-1,:]+ ((x[i,:]-x[i-1,:])**2 + (y[i,:]-y[i-1,:])**2)**0.5
    #    #print(i, arcF[i,jmax//2])
    for i in range(1, imax):
        arcF[i,:] = arcF[i-1,:]+ ((x[i,:]-x[i-1,:])**2 + (y[i,:]-y[i-1,:])**2)**0.5
    for j in range(1, jmax):    
        arcG[:,j] = arcG[:,j-1]+ ((x[:,j]-x[:,j-1])**2 + (y[:,j]-y[:,j-1])**2)**0.5
    for j in range(0, jmax):
        arcF[:,j] = arcF[:,j]/arcF[imax-1,j]
    for i in range(0, imax):
        arcG[i,:] = arcG[i,:]/arcG[i,jmax-1]

    for i in range(0, imax):
        for j in range(0, jmax):
            temp_f = (arcF[i,jmax-1]-arcF[i,0])
            temp_g = (arcG[imax-1,j]-arcG[0,j])
            Pij[i,j] = 1 - temp_f*temp_g
            xi[i,j]  = (arcF[i,0]+arcG[0,j]*temp_f)/Pij[i,j]
            eta[i,j] = (arcG[0,j]+arcF[i,0]*temp_g)/Pij[i,j] 
     
    return arcF, arcG, Pij, xi, eta


def calcTFIEdge(dP_te, dP_0, F, G, xi, eta, lojst, lojfn, upjst, upjfn):
    imax, jmax = F.shape[0], F.shape[1]
    num_wake_points = lojfn-lojst+1
    dE_wake_lower  = np.double(np.zeros((2, num_wake_points, 1)))
    dE_wake_upper  = np.double(np.zeros((2, num_wake_points, 1)))
    for ivar in range(2):
        for i in range(0, num_wake_points):
            #dE_wake_lower[ivar, i, 0] = (F[lojfn-1,0]-F[lojst-1+i,0])*dP_0[ivar]  + F[lojst-1+i,0]*dP_te[ivar] 
            #dE_wake_upper[ivar, i, 0] = (F[upjfn-1,0]-F[upjst-1+i,0])*dP_te[ivar] + F[upjst-1+i,0]*dP_0[ivar] 
            mylocal_F = F[lojst-1+i,0]/F[lojfn-1,0]
            dE_wake_lower[ivar, i, 0] = (1.-mylocal_F)*dP_0[ivar]  + mylocal_F*dP_te[ivar] 
            mylocal_F = (F[upjst-1+i,0]-F[upjst-1,0])/(F[upjfn-1,0]-F[upjst-1,0])
            dE_wake_upper[ivar, i, 0] = (1.-mylocal_F)*dP_te[ivar] + mylocal_F*dP_0[ivar] 
    return dE_wake_lower, dE_wake_upper

def calcUV(dE, F, G, xi, eta):
    imax, jmax = F.shape[0], F.shape[1]
    imax_check = dE.shape
    print(imax, jmax, imax_check)
    Uij  = np.double(np.zeros((2, imax, jmax)))
    Vij  = np.double(np.zeros((2, imax, jmax)))
    for ivar in range(2):
        for i in range(0,imax):
            for j in range(0,jmax):
                Uij[ivar,i,j]=(1-xi[i,j]) *dE[ivar,0,0] + xi[i,j] *dE[ivar,imax-1,0]
                Vij[ivar,i,j]=(1-eta[i,j])*dE[ivar,i,0] + eta[i,j]*dE[ivar,i,0] 
    return Uij, Vij

def calcDeformation(dE, F, G, xi, eta, Uij, Vij):
    imax, jmax = F.shape[0], F.shape[1]
    imax_check = dE.shape
    print(imax, jmax, imax_check)
    dS  = np.double(np.zeros((2, imax, jmax)))
    for ivar in range(2):
        for i in range(0,imax):
            for j in range(0,jmax):
                dS[ivar,i,j]=Uij[ivar,i,j]+Vij[ivar,i,j]-(1-eta[i,j])*Uij[ivar,i,0]-eta[i,j]*Uij[ivar,i,jmax-1]
    return dS
    

def calcArcLength_backup(x, y):
    
    imax, jmax = x.shape[0], x.shape[1]
    print(imax, jmax)
    arcF  = np.double(np.zeros((imax, jmax)))
    arcG  = np.double(np.zeros((imax, jmax)))
    
    dF  = np.double(np.zeros((imax, jmax)))
    dG  = np.double(np.zeros((imax, jmax)))
    #arcF[0,:] = 0.0
    #arcG[:,0] = 0.0
    
    #for j in range(0, jmax):
    #    arcF[:,j]=0
    #    for i in range(1, imax): 
    #        arcF[i,j] = ((x[i,j]-x[i-1,j])**2 + (y[i,j]-y[i-1,j])**2)**0.5
    #    #arcF[i,:] = arcF[i-1,:]+ ((x[i,:]-x[i-1,:])**2 + (y[i,:]-y[i-1,:])**2)**0.5
    #    #print(i, arcF[i,jmax//2])
    for i in range(1, imax):
        dF[i,:] = ((x[i,:]-x[i-1,:])**2 + (y[i,:]-y[i-1,:])**2)**0.5 
    for j in range(1, jmax):    
        dG[:,j] = ((x[:,j]-x[:,j-1])**2 + (y[:,j]-y[:,j-1])**2)**0.5
        #arcG[:,j] = arcG[:,j-1]+ ((x[:,j]-x[:,j-1])**2 + (y[:,j]-y[:,j-1])**2)**0.5
    for j in range(0, jmax):
        for i in range(1, imax):
        #temp = arcF[i-1,:]
            arcF[i,j] = np.sum(dF[0:i,j])
        arcF[:,j] = arcF[:,j]/arcF[imax-1,j]
    
    for i in range(0, imax):
        for j in range(1, jmax):    
            arcG[i,j] = np.sum(dG[i,0:j])
        arcG[i,:] = arcG[i,:]/arcG[i,jmax-1]
     
    return arcF, arcG
    #return dF, dG



#f0, g0, Pij, xi, eta =  calcArcLength(x0, y0)
#dE  = np.double(np.zeros((2, imax, 1)))
#Uij, Vij = calcUV(dE, f0, g0, xi, eta)
#fig = plt.subplots(1, 2)
##fig.suptitle('Horizontally stacked subplots')
#
#vmin = 0
#vmax = 1.0
#lv = np.r_[vmin: vmax: 29j*5]
#plt.subplot(1,4,1, aspect=1.)
#plt.contourf(x0, y0, f0, lv, vmin=vmin, vmax=vmax, cmap='jet')
#plt.colorbar()
#plt.subplot(1,4,2, aspect=1.)
#plt.contourf(x0, y0, g0, lv, vmin=vmin, vmax=vmax, cmap='jet')
#plt.colorbar()
#plt.subplot(1,4,3, aspect=1.)
#plt.contourf(x0, y0, Uij[1], lv, vmin=vmin, vmax=vmax, cmap='jet')
#plt.colorbar()
#plt.subplot(1,4,4, aspect=1.)
#plt.contourf(x0, y0, Vij[1], lv, vmin=vmin, vmax=vmax, cmap='jet')
#plt.colorbar()

#########################################################################################
#folderName="."
##fileName = "aesurf1_mode1.plt"
##num_head_lines=3
#
#fileName = "sanity_checkmod_1.dat"
#num_head_lines=2
#f=open(folderName+"/"+fileName,"r")
#lines=f.readlines()
#x,y,z = [],[],[]
#phix,phiy,phiz = [],[],[]
#
#i=1
#for xline in lines:
#       #if i>18 and i<5700:
#       if i>num_head_lines:
#           x.append(float(xline.split()[0]))
#           y.append(float(xline.split()[1]))
#           z.append(float(xline.split()[2]))
#           phix.append(float(xline.split()[3]))
#           phiy.append(float(xline.split()[4]))
#           phiz.append(float(xline.split()[5]))
#       i=i+1
#f.close()
#
#x = np.asarray(x).reshape((177,2))
#y = np.asarray(y).reshape((177,2))
#z = np.asarray(z).reshape((177,2))
#phix = np.asarray(phix).reshape((177,2))
#phiy = np.asarray(phiy).reshape((177,2))
#phiz = np.asarray(phiz).reshape((177,2))
#mode_1 = phiz[:,0].reshape((177))
#print("surface mode shape:", mode_1.shape)
## fourth dimension - colormap
## create colormap according to x-value (can use any 50x50 array)
#color_dimension = phiz # change to desired fourth dimension
#minn, maxx = color_dimension.min(), color_dimension.max()
#norm = colors.Normalize(minn, maxx)
#m = cm.ScalarMappable(norm=norm, cmap='jet')
#m.set_array([])
#fcolors = m.to_rgba(color_dimension)
#
#
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#
#ax.plot_surface(x,y,z,rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
#fig.colorbar(m, shrink=0.5, aspect=10)
## Set an equal aspect ratio
##ax.set_aspect('equal')# Set an equal aspect ratio
#plt.show()
#
#fileName = "sanity_checkmod_2.dat"
#num_head_lines=2
#f=open(folderName+"/"+fileName,"r")
#lines=f.readlines()
#x,y,z = [],[],[]
#phix,phiy,phiz = [],[],[]
#
#i=1
#for xline in lines:
#       #if i>18 and i<5700:
#       if i>num_head_lines:
#           x.append(float(xline.split()[0]))
#           y.append(float(xline.split()[1]))
#           z.append(float(xline.split()[2]))
#           phix.append(float(xline.split()[3]))
#           phiy.append(float(xline.split()[4]))
#           phiz.append(float(xline.split()[5]))
#       i=i+1
#f.close()
#
#x = np.asarray(x).reshape((177,2))
#y = np.asarray(y).reshape((177,2))
#z = np.asarray(z).reshape((177,2))
#phix = np.asarray(phix).reshape((177,2))
#phiy = np.asarray(phiy).reshape((177,2))
#phiz = np.asarray(phiz).reshape((177,2))
#mode_2 = phiz[:,0].reshape((177))
#
## fourth dimension - colormap
## create colormap according to x-value (can use any 50x50 array)
#color_dimension = phiz # change to desired fourth dimension
#minn, maxx = color_dimension.min(), color_dimension.max()
#norm = colors.Normalize(minn, maxx)
#m = cm.ScalarMappable(norm=norm, cmap='jet')
#m.set_array([])
#fcolors = m.to_rgba(color_dimension)
#
#
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#
#ax.plot_surface(x,y,z,rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
#fig.colorbar(m, shrink=0.5, aspect=10)
## Set an equal aspect ratio
##ax.set_aspect('equal')# Set an equal aspect ratio
#plt.show()

















if True:    
    # for fixed grid: load mesh outside the loop
    
    
    
    
    
    
    
    
    
    
    for ifile in range(5): 
    #if True: 


        #imax, jmax, kmax, x, y, threed = read_grid(mesh_fName)
        #print(i, file)



        
        f0, g0, Pij, xi, eta =  calcArcLength(x_original, y_original)





        #x_surface = x_original[jst-1:jfn,0]
        #y_surface = y_original[jst-1:jfn,0]




        
        vmin = 0
        vmax = 0.2
        #lv = np.r_[vmin: vmax: 29j*5]
        #lv = np.arange(0.05, 0.9 + inc, inc)
        




        xs_pred = np.copy(xs_initial)
        ys_pred = np.copy(ys_initial)
        
        plt.figure()



        inc = 0.1
        #control_x_locations = np.r_[0.2:0.8:1j*7]
        control_x_locations = np.arange(0.2, 0.8 + inc, inc)
        perturb = np.zeros(len(control_x_locations))
        #print("Seed: {}".format(seed))
        print(control_x_locations)
        for i in range(len(perturb)):
            seed = random.randint(0, 2**32 - 1)
            np.random.seed(seed)
            perturb[i] = np.random.uniform(-0.5, 0.5) 
        
        morphable_x = xs_pred[88:] 
        #b = hickshenne(morphable_x, np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), 2, 0.05, np.asarray([0.1, .1, .1, .05, 0.05, 0.5, 0.5]))
        #def hickshenne(x, xmi, width, SF, mag):
        b = hickshenne(morphable_x, control_x_locations, 3, 0.05, perturb)
        plt.plot(morphable_x, b)



        print(len(morphable_x))
        for i in range(0,89): 
            ys_pred[i+88] += b[i]



        control_x_locations = np.arange(0.2, 0.8 + inc, inc)
        for i in range(len(perturb)):
            seed = random.randint(0, 2**32 - 1)
            np.random.seed(seed)
            perturb[i] = np.random.uniform(-0.5, 0.5) 
        
        morphable_x = xs_pred[:89] 
        print(len(morphable_x))
        #b = hickshenne(morphable_x, np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), 2, 0.05, np.asarray([0.1, .1, .1, .05, 0.05, 0.5, 0.5]))
        b = hickshenne(morphable_x, control_x_locations, 2, 0.05, perturb)
        plt.plot(morphable_x, b)

        for i in range(0,89): 
            ys_pred[i] += b[i]

        plt.show()


        if CHECK_AIRFOIL:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #ax = fig.subplot(1,2,1, aspect=1.)
            #ax.plot(x_surface, y_surface,"k.")
            ax.plot(xs_initial, ys_initial,"k.")
            ax.plot(xs_pred, ys_pred,"g-", lw=1.5)
            #ax.plot(xs_initial, ys_initial, color="pink", lw=0.2)
            ax.set_aspect('equal')
            fig.savefig('./hickshenne/airfoil_'+str(ifile)+'.png')
            #airfoil_coordinates_output = np.stack((x_surface, y_surface),axis=-1)
            #np.savez_compressed("./airfoils_npz/airfoil_"+file.split("_")[1]+".npz", a=airfoil_coordinates_output)
            plt.show()
            #plt.close()


        dP_i0_j0   = np.zeros((2))
        dP_te      = np.zeros((2))
        dP_te[0] = xs_pred[0] - xs_initial[0]
        dP_te[1] = ys_pred[0] - ys_initial[0]
        print("TE point deformation:", dP_te, ys_pred[0], ys_initial[0])
        #ax = fig.add_subplot(111)

        dE_wake_lower, dE_wake_upper = calcTFIEdge(dP_te, dP_i0_j0, f0, g0, xi, eta, wake_lower_jst, wake_lower_jfn, wake_upper_jst, wake_upper_jfn)
        dE_surface_x = xs_pred - xs_initial 
        dE_surface_y = ys_pred - ys_initial 


        #plt.figure()
        #plt.plot(xs_pred, dE_surface_x)
        #plt.figure()
        #plt.plot(xs_pred, dE_surface_y)
        #plt.show()
        dE_surface = np.concatenate((dE_surface_x.reshape(1,jfn-jst+1,1), dE_surface_y.reshape(1,jfn-jst+1,1)), axis=0)

        dE = np.concatenate((dE_wake_lower[:,0:-1,:], dE_surface[:,0:-1,:], dE_wake_upper) ,axis=1)








        
        Uij, Vij = calcUV(dE, f0, g0, xi, eta)
        print(Uij.shape, Vij.shape)
        dS = calcDeformation(dE, f0, g0, xi, eta, Uij, Vij)
        print(dE.shape, dS.shape)


        plt.subplot(1,1,1, aspect=1.)
        segs1 = np.stack((x_original+dS[0], y_original+dS[1]), axis=2)
        segs2 = segs1.transpose(1,0,2)
        plt.gca().add_collection(LineCollection(segs1, colors='green', linewidth=0.5))
        plt.gca().add_collection(LineCollection(segs2, colors='green', linewidth=0.5))

        segs1 = np.stack((x_original, y_original), axis=2)
        segs2 = segs1.transpose(1,0,2)
        plt.gca().add_collection(LineCollection(segs1, colors='black', linewidth=0.2))
        plt.gca().add_collection(LineCollection(segs2, colors='black', linewidth=0.2))


        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        #plt.title(r'TFI: $t$=%1.2f [s]' %((ifile+1)*0.01))
        plt.savefig('./hickshenne/tfi_'+str(ifile)+'.png')

        plt.show()
        plt.close()


 

