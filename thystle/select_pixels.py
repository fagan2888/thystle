#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
# from utils import *

def select_pixels(index, thr=5):
    """
    Read and threshold a clipped luminosity image.
    """

    # -- set the data directory
    dpath = os.path.join("..", "output", "clips_luminosity")

    # -- get the file list
    flist = sorted(glob.glob(os.path.join(dpath, "*.npy")))

    # -- return the image
    print("reading clips luminosity file: {0}".format(flist[index]))
    return np.load(flist[index]) > thr

