#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def lighting_tech(labs, km, imgL):
    
    # -- set the colors, types, and KM indices
    clrs  = ['#E24A33', '#8EBA42', '#348ABD', '#988ED5', '#FBC15E', '#FFB5B8']
    types = ['High\nPressure\nSodium', 'LED', 'Fluorescent', 'Metal\nHalide',
             'LED', 'LED']
    kinds = [np.array([10, 1, 8, 14]), np.array([2]), np.array([3, 12]),
             np.array([5, 4]), np.array([0]), np.array([7])]

    # -- get the x/y positions of all active pixels
    pos      = np.arange(imgL.size).reshape(imgL.shape)[imgL]
    xpos_all = pos % imgL.shape[1]
    ypos_all = pos // imgL.shape[1]

    # -- get the positions for each type
    xpos = []
    ypos = []
    for ii in range(len(kinds)):
        txpos = np.hstack([xpos_all[labs == kind] for kind in kinds[ii]])
        typos = np.hstack([ypos_all[labs == kind] for kind in kinds[ii]])
        xpos.append(txpos)
        ypos.append(typos)

    # -- plot utilities
    # stamp   = imgL[600:850,:250]
    stamp   = imgL[750:1250, 500:1000]
    xs      = 1.0 * 6.5
    asp     = 0.45
    xoff    = 0.05
    wid_top = 1.0 - 2.0 * xoff
    wid_bot = (wid_top - 0.5 * xoff) * 0.5
    rat_top = 1600. / 1560.
    rat_bot = 250. / 250.
    ys      = 2.5 * xoff * xs + wid_top * rat_top * asp * xs + \
              wid_bot * rat_bot * asp * xs
    hgt_top = wid_top * rat_top * asp * xs / ys
    ysep    = 0.5 * xoff * xs / ys
    yoff    = xoff * xs / ys
    hgt_bot = wid_bot * rat_bot * asp * xs / ys

    # -- initialize the plot
    plt.close("all")
    fig     = plt.figure(figsize=(xs, ys))
    ax_top  = fig.add_axes((xoff, yoff + 0.5 * yoff + hgt_bot, wid_top,
                            hgt_top))
    ax_botl = fig.add_axes((xoff, yoff, wid_bot, hgt_bot))
    ax_botr = fig.add_axes((0.5 + 0.25 * xoff, yoff, wid_bot, hgt_bot))

    # -- add the lighting tags
    for ii in range(len(kinds)):
        ax_top.scatter(xpos[ii], ypos[ii], 1, clrs[ii], ".", lw=0)
        ax_botr.scatter(xpos[ii], ypos[ii], 4, clrs[ii], ".", lw=0)

    im_top = ax_top.imshow(imgL, "bone", clim=[0, 1], aspect=0.45)
    im_bot = ax_botr.imshow(imgL, "bone", clim=[0, 1], aspect=0.45)
    ax_top.axis("off")
    ax_botr.axis("off")
    # ax_botr.set_xlim(0, 250)
    # ax_botr.set_ylim(850, 600)
    ax_botr.set_xlim(500, 1000)
    ax_botr.set_ylim(1350, 850)

    # -- label the top and bottom images
    txtsz = 10
    yr = ax_top.get_ylim()
    xr = ax_top.get_xlim()
    ax_top.text(xr[0], yr[1] - 0.03 * (yr[0] - yr[1]),
                "New York City Lighting Technologies", ha="left",
                va="center", fontsize=txtsz)
    # ax_botr.text(250, 850 + 0.08 * (850 - 600), "Manhattan Bridge Region",
    #              ha="right", va="center", fontsize=txtsz)
    ax_botr.text(1000, 1350 + 0.08 * (1350 - 850), "Manhattan Bridge Region",
                 ha="right", va="center", fontsize=txtsz)

    # -- plot the spectra
    ax_botl.set_ylim(-1, 8)
    ax_botl.set_xlim(0, 6 * (waves16[-1] - waves16[0]))
    for ii in range(6):
        twaves = waves16 - waves16[0] + ii * (waves16[-1] - waves16[0])
        tkmc   = km.cluster_centers_[kinds[ii][0]]
        ax_botl.plot(twaves, tkmc - tkmc.min(), color=clrs[ii], lw=0.5)
        ax_botl.text(0.5 * (twaves.max() + twaves.min()), 7.9, types[ii],
                     fontsize=5, ha="center", va="top")

    ax_botl.set_xticks([ii*(waves16[-1] - waves16[0]) for ii in range(6)])
    ax_botl.set_xticklabels("")
    ax_botl.set_yticklabels("")
    ax_botl.xaxis.grid(1)
    ax_botl.set_ylabel("intensity [arb units]")
    ax_botl.set_xlabel("wavelength [range: 0.4-1.0 microns]")
    fig.canvas.draw()

    return



def lighting_tech2(labs, km, imgL):

    # -- set colors
    clrs = np.array([to_rgb(i) for i in ['#000000', '#FBC15E',
                                         '#E24A33', '#8EBA42', '#348ABD',
                                         '#988ED5', '#988ED5', '#000000',
                                         '#FFB5B8', '#E24A33', '#000000',
                                         '#E24A33', '#000000', '#348ABD',
                                         '#000000', '#E24A33']])

    # -- make map
    lmap = np.zeros(imgL.shape, dtype=int) - 1
    lmap[imgL] = labs
    lmap += 1
    tech = clrs[lmap]
    
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(tech[::10, ::10], aspect=0.45)
    fig.canvas.draw()

    return


def lighting_tech3(labs, km, imgLc, imgL, outfile=None, aspect=1, xlim=None,
                   ylim=None):
    
    # -- set the colors, types, and KM indices
    clrs  = ['#E24A33', '#8EBA42', '#348ABD', '#988ED5', '#FBC15E', '#FFB5B8']
    types = ['High\nPressure\nSodium', 'LED', 'Fluorescent', 'Metal\nHalide',
             'LED', 'LED']
    kinds = [np.array([10, 1, 8, 14]), np.array([2]), np.array([3, 12]),
             np.array([5, 4]), np.array([0]), np.array([7])]

    # -- get the x/y positions of all active pixels
    pos      = np.arange(imgL.size).reshape(imgL.shape)[imgLc]
    xpos_all = pos % imgL.shape[1]
    ypos_all = pos // imgL.shape[1]

    # -- get the positions for each type
    xpos = []
    ypos = []
    for ii in range(len(kinds)):
        txpos = np.hstack([xpos_all[labs == kind] for kind in kinds[ii]])
        typos = np.hstack([ypos_all[labs == kind] for kind in kinds[ii]])
        xpos.append(txpos)
        ypos.append(typos)

    # -- initialize figure
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.subplots_adjust(0, 0, 1, 1)

    # -- add the lighting tags
    for ii in range(len(kinds)):
        ax.scatter(xpos[ii], ypos[ii], 1, clrs[ii], ".", lw=0)
        ax.scatter(xpos[ii], ypos[ii], 4, clrs[ii], ".", lw=0)

    # -- display the luminosity image
    ax.imshow(imgL, "gist_gray", aspect=aspect)

    # -- clean up axis
    ax.axis("off")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    fig.canvas.draw()

    # -- save figure if desired
    if outfile:
        fig.savefig(outfile, clobber=True)

    return



def lighting_tech_specs(waves, km):
    
    # -- set the colors, types, and KM indices
    clrs  = ['#E24A33', '#8EBA42', '#348ABD', '#988ED5', '#FBC15E', '#FFB5B8']
    types = ['High Pressure\nSodium', 'LED', 'Fluorescent', 'Metal\nHalide',
             'LED', 'LED']
    inds  = [10, 2, 3, 5, 0, 7]

    fig, ax = plt.subplots(1, 6, figsize=(10, 2.5), sharey=True)
    fig.subplots_adjust(0.06, 0.22, 0.94, 0.8, 0.0)

    for ii in range(len(clrs)):
        ax[ii].plot(waves, km.cluster_centers_[inds[ii]], color=clrs[ii])
        ax[ii].set_title(types[ii])

    ax[0].set_ylabel("intensity\n[arb units]")
    fig.text(0.5, 0.02, "wavelength [nm]", ha="center")
    fig.canvas.draw()

    fig.savefig("../output/lighting_tech_specs.png", clobber=True)

    return
