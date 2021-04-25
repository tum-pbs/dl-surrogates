################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Helpers for data generation
#
################

import os
import numpy as np
from PIL import Image
from matplotlib import cm
from random import choice, randint

def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)

def imageOut(filename, outputs_param, targets_param, saveTargets=False):
    outputs = np.copy(outputs_param)
    targets = np.copy(targets_param)

    for i in range(3):
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        outputs[i] -= min_value
        targets[i] -= min_value
        max_value -= min_value
        outputs[i] /= max_value
        targets[i] /= max_value

        suffix = ""
        if i==0:
            suffix = "_pressure"
        elif i==1:
            suffix = "_velX"
        else:
            suffix = "_velY"

        im = Image.fromarray(cm.magma(outputs[i], bytes=True))
        im = im.resize((512,512))
        im.save(filename + suffix + "_pred.png")

        if saveTargets:
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            im = im.resize((512,512))
            im.save(filename + suffix + "_target.png")

def saveAsImage(filename, field_param):
    field = np.copy(field_param)
    field = np.flipud(field.transpose())

    min_value = np.min(field)
    max_value = np.max(field)
    field -= min_value
    max_value -= min_value
    field /= max_value

    im = Image.fromarray(cm.magma(field, bytes=True))
    im = im.resize((512, 512))
    im.save(filename)

def random_signal(num_superpos = 10, max_frequency = 20, sampling = 64):
    x = np.linspace(1e-1, 2*np.pi - 1e-1, sampling)
    y = np.sin(x)

    for _ in range(2,num_superpos + 1):
        omega = randint(1, max_frequency)
        trig_function = choice([np.sin, np.cos, np.tan])

        if trig_function == np.tan:
            y_temp = np.clip(trig_function(omega), -1, 1)
            y += y_temp
        else:
            y += trig_function(omega*x)

    y = np.interp(y, (y.min(), y.max()), (-1, +1))
    y[y < 0] *= -1

    return y