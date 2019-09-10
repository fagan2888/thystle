#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import hyss
imps = ["utils.py", "select_pixels.py"]
for ii in imps:
    exec(open(ii).read())


# -- initialize timer
t00 = time.time()

# -- set the scan index number
snum = 160
# snum = 159

# -- get the file list
flist = get_file_list()
print("working on scan: {0}".format(flist[snum]))

# -- get the source pixels
imgLcc = select_pixels(snum)

# -- get the spectra
print("reading spectra...")
t0    = time.time()
cube  = read_hyper(flist[snum])
specs = cube.data[:, imgLcc].astype(float).T
elapsed_time(t0)

# -- read in noaa, remove correlated spectra, and interpolate
noaa = hyss.HyperNoaa()
noaa.remove_correlated()
noaa.interpolate(cube.waves)

# -- standardize both spects
stand_noaa  = standardize(noaa.irows)
stand_specs = standardize(specs)

# -- correlate
print("correlating spectra with NOAA...")
t0 = time.time()
cc = np.dot(stand_specs, stand_noaa.T) / stand_specs.shape[1]
elapsed_time(t0)

# -- total time
elapsed_time(t00, "TOTAL ")
