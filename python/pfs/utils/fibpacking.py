# JEG C converted to python with minimal changes beyond a bugfix
#
# This function takes a fiber number and converts it to x,y on the
# ferrule face, assuming 191.5 micron fiber `circles' and a 2181 micron
# hex, and the silly numbering scheme in the SuNSS document.

import numpy as np

__all__ = ["fibxy"]

# to do this, we need some arrays which take a row number on the ferrule,
# ir = 0 -> 12

# number of fibers in row
ns = np.array([7, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 7], dtype=int)
# direction of fiber number increase in row
direct = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
# first fiber number in row + ficticious end fiber */
fbase = np.array([1, 8, 16, 25, 35, 46, 58, 71, 83, 94, 104, 113, 121, 128])
# fiber number of central fiber in row; interpolated if no central fiber
xcar = np.array([4, 11.5, 20, 29.5, 40, 51.5, 64, 76.5, 88, 98.5, 108, 116.5, 124])

N_PER_FERRULE = 127                     # number of fibres per ferrule
FIBER_RADIUS = 95.75                    # fiber radius, microns
CORE_RADIUS = 64                        # fiber core radius, microns

def fibxy(nf):
    """takes a fiber number and returns the x,y coordinates in microns of the fiber center
    """
    # find row
    ir = -1                             # row number
    fb = 1
    for i in range(len(ns) + 1):
        if nf < fb:
            ir = i - 1
            break
        fb += ns[i]

    if ir < 0:
        raise RuntimeError(f"Invalid fiberID {nf}")

    if False:
        print("Row number for fiber %d: %d, fbase=%d" % (nf,ir,fbase[ir]))


    y = (6 - ir)*np.sqrt(3)*FIBER_RADIUS                   # y, microns
    x = (nf - xcar[ir])*direct[ir]*2*FIBER_RADIUS          # x, microns

    return (x, y)


