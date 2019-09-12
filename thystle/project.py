#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import hyss
import numpy as np
import pickle as pkl
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

# -- these should be turned into imports
imps = ["utils.py", "select_pixels.py"]
for ii in imps:
    exec(open(ii).read())


# -- defaults
kmcluster = False

# -- initialize timer
t00 = time.time()

# -- set the scan index number
snum = 159

# -- get the file list
flist = get_file_list()
print("working on scan: {0}".format(flist[snum]))

# -- get the source pixels
imgLcc = select_pixels(snum, vfilter=True)

# -- get the spectra
print("reading spectra...")
t0    = time.time()
cube  = read_hyper_clean(flist[snum])
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

# -- cluster spectra
if kmcluster:
    print("K-Means clustering...")
    t0 = time.time()
    km = KMeans(n_clusters=15, n_jobs=16)
    km.fit(stand_specs)
    elapsed_time(t0)

# -- read in the 2016 clusters
kmname  = os.path.join("..", "data", "km_cluster.pkl")
wname   = os.path.join("..", "data", "vnir_waves.npy")
km16    = pkl.load(open(kmname, "rb"), encoding="latin1")
waves16 = np.load(wname)

# -- interpolate specs onto 2016 wavelengths and re-standardize
print("interpolating 2018 spectra onto 2016 wavelengths...")
t0           = time.time()
interp_specs = interp1d(cube.waves, specs, axis=1, fill_value="extrapolate")
specs18      = interp_specs(waves16)
specs18      = standardize(specs18)
elapsed_time(t0)

# -- assign 2018 spectra to 2016 K-Means clusters
labs18 = km16.predict(specs18)

# -- total time
elapsed_time(t00, "TOTAL ")
