#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from scipy.ndimage.filters import uniform_filter as uf
# from utils import *

def select_pixels(index, thr=5, vfilter=False):
    """
    Read and threshold a clipped luminosity image.
    """

    # -- set the data directory
    dpath = os.path.join("..", "output", "clips_luminosity")

    # -- get the file list
    flist = sorted(glob.glob(os.path.join(dpath, "*.npy")))

    # -- return the image
    if vfilter:
        imgLcc = np.load(flist[index]) > thr
        imgLcc[uf(1.0 * imgLcc, (100, 0)) == 1] = False
        return imgLcc        
    else:
        return np.load(flist[index]) > thr

