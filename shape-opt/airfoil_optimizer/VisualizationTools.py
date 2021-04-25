import os, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clr 

_CCDICT = {'blue' : (( 0.0,  1.0, 1.0),
                     ( 0.5,  0.0, 0.0),
                     ( 1.0,  0.0, 0.0)),

           'green': (( 0.0,  0.0, 0.0),
                     ( 0.5,  0.0, 0.0),
                     ( 0.6,  0.5, 0.5),
                     ( 1.0,  0.0, 0.0)),

           'red'  : (( 0.0,  0.0, 0.0),
                     ( 0.5,  0.0, 0.0),
                     ( 1.0,  1.0, 1.0))}

_EPSILON = 1e-1

def extractGradients(infile='./process.log', outfile='./2_gradients.log'):
    gradient_list = []
    with open(infile, 'r') as input:
        with open(outfile, 'w') as output:
            hitTheLine   = False
            gradientLine = ''
            iter    = 0
            output.write('Gradients of drag wrt parameters calculated during optimization:\n\n')
            for _, line in enumerate(input):
                if 'Gradient of parameter' in line:
                    hitTheLine    = True
                    gradientLine  = line
                    continue
                if hitTheLine:
                    if 'Updating param' in line:
                        hitTheLine    = False
                        gradientLine  = ''.join([l.strip() for l in gradientLine.split('\n')])
                        gradients     = gradientLine.split('[[')[1].split(']]')[0].strip().split(',')                    
                        gradient_list.append([float(s.strip()) for s in gradients])
                        output.write('Iteration: {} -->\t'.format(iter) + str(gradient_list[-1]) + '\n\n')
                        iter += 1
                    else:    
                        gradientLine += line
    return gradient_list

def visualizeGradients(maskFile, pressureFile, gradients, outfile, display=False):
    plt.clf(); plt.cla()
    binaryMask = np.load(maskFile)['a']
    pressure   = np.load(pressureFile)['a']
    invBinMask = np.where(binaryMask==1, 0, 1)
    max = np.max(pressure); min = np.min(pressure)
    pressure = ((pressure - min) / (max - min) * 2.0 - 1.0) * invBinMask
    indicies = np.where(binaryMask == 1)
    indicies = np.column_stack(indicies)
    firstRow = indicies[ 0][0]
    lastRow  = indicies[-1][0]
    for row in range(firstRow, lastRow+1, 2):
        gradientIDX = int((row - firstRow)/2)
        idxToSelect = np.where(indicies[:,0]==row)[0]
        binMaskIDXs = indicies[int(idxToSelect[0]):int(idxToSelect[-1])+1,:]
        if -_EPSILON < gradients[gradientIDX] < _EPSILON:
            binaryMask[row    ,binMaskIDXs[0][1]:binMaskIDXs[-1][1]+1] =  0
            binaryMask[row + 1,binMaskIDXs[0][1]:binMaskIDXs[-1][1]+1] =  0
        else:
            binaryMask[row    ,binMaskIDXs[0][1]:binMaskIDXs[-1][1]+1] = gradients[gradientIDX] / abs(gradients[gradientIDX])
            binaryMask[row + 1,binMaskIDXs[0][1]:binMaskIDXs[-1][1]+1] = gradients[gradientIDX] / abs(gradients[gradientIDX])
            plt.text(binMaskIDXs[binMaskIDXs.shape[0]//2][1], row+1.25, str(gradients[gradientIDX]), 
                     fontsize=5, fontweight='bold', horizontalalignment='center', 
                     bbox=dict(facecolor='white', alpha=0.2, pad=0.2))
    outImage   = binaryMask + pressure
    customMap  = clr.LinearSegmentedColormap('CustomMap', _CCDICT)
    plt.imshow(outImage, interpolation='nearest', cmap=customMap)
    plt.colorbar()
    plt.savefig(outfile, interpolation='nearest', cmap=customMap, dpi=600)
    if display:
        plt.show()

def gradientDebug(gradients, binaryMaskDir='./saved_figures/', pressureDir='./online_solver/data_pictures/', outdir='./grad_debug/'):
    if not os.path.exists(outdir): os.makedirs(outdir)
    binMaskFiles  = sorted([int(file.split('_')[1].split('.')[0]) for file in os.listdir(binaryMaskDir) if 'npz' in file])
    pressureFiles = sorted([int(file.split('_')[1].split('.')[0]) for file in os.listdir(pressureDir)   if 'pressure' in file and'npz' in file])
    assert len(binMaskFiles) == len(pressureFiles), "{} many binary mask and {} pressure file doesn't match one to one.".format(len(binMaskFiles), len(pressureFiles))
    for bm, p, idx in zip(binMaskFiles, pressureFiles, range(len(binMaskFiles))):
        bm = os.path.join(binaryMaskDir,'ImageAirfoil_' + str(bm) + '.npz')
        p  = os.path.join(pressureDir,  'pressure_'     + str(p)  + '.npz')
        visualizeGradients(bm, p, gradients[idx], outfile=os.path.join(outdir, 'grad_debug_' + str(idx) + '.png'))

def lossGraph(infile='./process.log', outfile='./1_loss.log', display=False):
    iter_list = []
    loss_list = []
    drag_pres_list = []
    drag_visc_list = []

    with open(infile, 'r') as input:
        with open(outfile, 'w') as output:
            output.write("Iteration \t\t Loss\n")
            output.write("--------- \t\t ----\n")
            for _, line in enumerate(input):
                if 'Design Loss' in line:
                    try:
                        iter = int(line.split('Design Loss:')[1].split('||')[1].replace('iter:', '').strip())
                        loss = float(line.split('Design Loss:')[1].split('||')[0].strip())
                        if iter < 10:
                            output.write("\t\t{}\t\t\t\t\t{}\n".format(iter, loss))
                        elif iter < 100:
                            output.write("\t {}\t\t\t\t\t{}\n".format(iter, loss))
                        else:
                            output.write("\t {}\t\t\t\t{}\n".format(iter, loss))
                        if isinstance(iter, int) and isinstance(loss, float):
                            iter_list.append(iter)
                            loss_list.append(loss)
                    except Exception as err:
                        print("Exception caught: {}".format(str(err)))

                iter = 0
                if 'Drag (pressure part, raw data)' in line:
                    try:
                        drag_pressure = float(line.split('Drag (pressure part, raw data): ')[1].strip().rstrip('.')) 
                        
                        if iter < 10:
                            output.write("\t\t{}\t\t\t\t\t{}\n".format(iter, drag_pressure))
                        elif iter < 100:
                            output.write("\t {}\t\t\t\t\t{}\n".format(iter, drag_pressure))
                        else:
                            output.write("\t {}\t\t\t\t{}\n".format(iter, drag_pressure))
                        if isinstance(drag_pressure, float):
                            drag_pres_list.append(drag_pressure)
                            iter = iter + 1
                    except Exception as err:
                        print("Exception caught: {}".format(str(err)))

                iter = 0
                if 'Drag (viscous part, raw data)' in line:
                    try:
                        drag_viscous = float(line.split('Drag (viscous part, raw data): ')[1].strip().rstrip('.')) 
                        
                        if iter < 10:
                            output.write("\t\t{}\t\t\t\t\t{}\n".format(iter, drag_viscous))
                        elif iter < 100:
                            output.write("\t {}\t\t\t\t\t{}\n".format(iter, drag_viscous))
                        else:
                            output.write("\t {}\t\t\t\t{}\n".format(iter, drag_viscous))
                        if isinstance(drag_viscous, float):
                            drag_visc_list.append(drag_viscous)
                            iter = iter + 1
                    except Exception as err:
                        print("Exception caught: {}".format(str(err)))                        

                        
            npList   = np.array(loss_list)
            indecies = npList.argsort()
            npList_drag_pressure = np.array(drag_pres_list)
            npList_drag_viscous  = np.array(drag_visc_list)

            try:
                min1,  min2,   min3  = npList[indecies][0:3].tolist()
                iter1, iter2,  iter3 = indecies[0:3].tolist()
                output.write('''\n\n\t Least three drag 
                                {} \t {}, 
                                {} \t {}, 
                                {} \t {},
                            '''.format(min1, iter1, min2, iter2, min3, iter3))
            except:
                pass
    plt.plot(iter_list, loss_list, 'g^', iter_list, loss_list, 'k')
    #plt.ylim(0.2, 0.5)
    plt.savefig('Loss.png')
    plt.clf()
    
    fig, ax = plt.subplots()
    ax.plot(iter_list, npList_drag_pressure, '-b', label='Pressure')
    ax.plot(iter_list, npList_drag_viscous,  '-r', label='Viscous')
    leg = ax.legend()
    plt.savefig('pressure_and_viscous.png')

    if display: plt.show()

def main(): 
    lossGraph()
    #gradients = extractGradients()
    #gradientDebug(gradients)

if __name__ == '__main__':
    main()
