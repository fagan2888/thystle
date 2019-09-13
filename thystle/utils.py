#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob
import numpy as np
from scipy.ndimage import measurements as spm
from scipy.ndimage.filters import gaussian_filter as gf

def read_header(hdrfile, verbose=True):
    """
    Read a Middleton header file.

    Parameters
    ----------
    hdrfile : str
        Name of header file.
    verbose : bool, optional
        If True, alert the user.

    Returns
    -------
    dict : dict
        A dictionary continaing the number of rows, columns, and wavelengths
        as well as an array of band centers.
    """

    # -- alert
    if verbose:
        print("reading and parsing {0}...".format(hdrfile))

    # -- open the file and read in the records
    recs = [rec for rec in open(hdrfile)]

    # -- parse for samples, lines, bands, and the start of the wavelengths
    for irec, rec in enumerate(recs):
        if 'samples' in rec:
            samples = int(rec.split("=")[1])
        elif 'lines' in rec:
            lines = int(rec.split("=")[1])
        elif 'bands' in rec:
            bands = int(rec.split("=")[1])
        elif "Wavelength" in rec:
            w0ind = irec+1

    # -- parse for the wavelengths
    waves = np.array([float(rec.split(",")[0]) for rec in 
                      recs[w0ind:w0ind+bands]])

    # -- return a dictionary
    return {"nrow":samples, "ncol":lines, "nwav":bands, "waves":waves}


def read_raw(rawfile, shape, hyper=False, verbose=True):
    """
    Read a Middleton raw file.

    Parameters
    ----------
    rawfile : str
        The name of the raw file.
    shape : tuple
        The output shape of the data cube (nwav, nrow, ncol).
    hyper : bool, optional
        Set this flag to read a hyperspectral image.
    verbose : bool, optional
        Alert the user.

    Returns
    -------
    memmap : memmap
        A numpy memmap of the datacube.
    """

    # -- alert
    if verbose:
        print("reading {0}...".format(rawfile))

    # -- read either broadband or hyperspectral image
    if hyper:
        return np.memmap(rawfile, np.uint16, mode="r") \
            .reshape(shape[2], shape[0], shape[1])[:, :, ::-1] \
            .transpose(1, 2, 0)
    else:
        return np.memmap(rawfile, np.uint8, mode="r") \
            .reshape(shape[1], shape[2], shape[0])[:, :, ::-1]


def read_hyper(fpath, fname=None, full=True):
    """
    Read a full hyperspectral scan (raw and header file).

    Parameters
    ----------
    fpath : str
        Either the full name+path of the raw file or the path of the raw file.
        If the latter, fname must be supplied.
    fname : str, optional
        The name of the raw file (required if fpath is set to a path).
    full : bool, optional
        If True, output a class containing data and supplementary information.
        If False, output only the data.

    Returns
    -------
    output or memmap : class or memmap
        If full is True, a class containing data plus supplementary 
        information.  If full is False, a memmap array of the data.
    """

    # -- set up the file names
    if fname is not None:
        fpath = os.path.join(fpath, fname)

    # -- read the header
    hdr = read_header(fpath.replace("raw", "hdr"))
    sh  = (hdr["nwav"], hdr["nrow"], hdr["ncol"])

    # -- if desired, only output data cube
    if not full:
        return read_raw(fpath, sh, hyper=True)

    # -- output full structure
    class output():
        def __init__(self, fpath):
            self.filename = fpath
            self.data     = read_raw(fpath, sh, hyper=True)
            self.waves    = hdr["waves"]
            self.nwav     = sh[0]
            self.nrow     = sh[1]
            self.ncol     = sh[2]

    return output(fpath)


def get_file_list(clean=True):

    # -- get file list
    dpath = os.environ["HSI1_DATA"]
    flist = sorted([i for i in glob.glob(os.path.join(dpath, "night_*.raw"))])

    # -- only return viable files if desired
    if clean:
        return [i for i in flist if "00056" not in i and "00461" not in i]
    else:
        return flist


def get_clipsL_list():

    # -- get file list
    dpath = os.path.join("..", "output", "clips_luminosity")
    return sorted([i for i in glob.glob(os.path.join(dpath, "*.npy"))])


def write_log(lout, msg, flush=True, stdout=False):

    # -- write to stdout if desired
    if stdout:
        print(msg)
        return
    
    # -- write the message to file
    lout.write(msg + "\n")

    # -- flush if desired
    if flush:
        lout.flush()

    return


def read_clean(fpath, fname=None):

    # -- read in the hyperspectral cube
    cube = read_hyper(fpath, fname)

    # -- rebinning data
    print("data must be spectrally rebinned by a factor of 4 for cleaning...")
    t0 = time.time()
    fac = 4
    cube.dbin = cube.data.reshape(cube.nwav // fac, fac, cube.nrow, cube.ncol)\
                        .mean(1)
    print("  {0}s".format(time.time() - t0))
    
    # -- read in the median offset
    rname    = os.path.join("..", "output", "sigma_clips",
                            os.path.split(cube.filename)[1] \
                            .replace(".raw", "_rows_md.npy"))
    cname_up = rname.replace("_rows_", "_cols_up_")
    cname_lo = rname.replace("_rows_", "_cols_lo_")

    # -- subtract from binned data
    print("cleaning rows...")
    t0 = time.time()
    cube.dbin -= np.load(rname)
    print("  {0}s".format(time.time() - t0))
    print("cleaning upper columns...")
    t0 = time.time()
    cube.dbin[:, :800] -= np.load(cname_up)
    print("  {0}s".format(time.time() - t0))
    print("cleaning lower columns...")
    t0 = time.time()
    cube.dbin[:, 800:] -= np.load(cname_lo)
    print("  {0}s".format(time.time() - t0))
    
    return cube.dbin


def read_clean2(fpath, fname=None):

    # -- read in the hyperspectral cube
    cube = read_hyper(fpath, fname)

    # -- set the file names for the offsets
    cname = os.path.join("..", "output", "scan_offsets", 
                         os.path.split(cube.filename)[1] \
                         .replace(".raw", "_cmed.npy"))
    rname = cname.replace("cmed", "rmed")
 
    # -- subtract the offsets
    print("  removing offsets...")
    return cube.data - np.load(cname) - np.load(rname)


def select_small_sources(imgL, thr, lo, hi):

    # -- clip the luminosity image
    imgLc = imgL > thr

    # -- select the sources
    labs = spm.label(imgLc)
    nsrc = labs[1]

    # -- get the source sizes
    lsz = spm.sum(imgLc, labs[0], range(1, nsrc + 1))

    # -- get sources within some range
    ind = (lsz >= lo) & (lsz < hi)

    # -- return the labels, sizes, and indices
    return labs, lsz, ind


def elapsed_time(t0, prefix="  "):
    """ Helper function for elapse time. """

    # -- round to the nearest millisecond and alert
    dt = round(time.time() - t0, 3)
    print(prefix + "elapsed time: {0}s".format(dt))

    return


def standardize(arr, axis=1):
    """ Standardize a 2D array (assumes float).  Set to 0 when standard 
    deviation is 0. """

    # -- find where standard deviation is 0
    sig   = arr.std(axis, keepdims=True)
    coeff = sig == 0

    # -- return standardized array
    return ~coeff * (arr - arr.mean(axis, keepdims=True)) / (sig + coeff)


def read_hyper_clean(fpath, fname=None):
    """ Read a clean hyperspectral scan. """

    # -- read in the scan
    print("reading and converting to float...")
    t0        = time.time()
    cube      = read_hyper(fpath, fname)
    cube.data = cube.data.astype(float)
    elapsed_time(t0)

    # -- read in the offset
    oname = os.path.split(cube.filename)[-1].replace(".raw", "_off.npy")
    opath = os.path.join("..", "output", "scan_offsets", oname)
    try:
        off = np.load(opath)
    except:
        print("offset file {0} not found!!!\n  Generating...".format(opath))
        t0  = time.time()
        off = np.median(gf(cube.data, (0, 1, 1)), 2, keepdims=True)
        np.save(opath, off)
        elapsed_time(t0)

    # -- remove offset
    print("removing offset...")
    t0         = time.time()
    cube.data -= off
    elapsed_time(t0)

    return cube
