import os
import sys
import shutil 
import numpy as np
import matplotlib.pyplot as plt
import math
from random import sample 

root_dir = os.getcwd()
data_dir = os.path.join(os.getcwd(), "train")
out_dir  = os.path.join(os.getcwd(), "test")

os.chdir(data_dir)
files = [f for f in os.listdir(".") if os.path.isfile(f)]
random_files = sample(files, 79)

idx=1
for f in random_files:
    print("Processing {}/{}".format(idx, len(random_files)))
    shutil.move(os.path.join(data_dir, f), os.path.join(out_dir, f) )
    idx += 1

os.chdir(root_dir)

