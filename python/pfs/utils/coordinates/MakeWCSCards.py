#!/usr/bin/env python

import os,sys,re
import scipy
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from . import DistortionCoefficients

def WCSParameters(mode,cent,rot,alt,az):

    c=DistortionCoefficients.Coeff(mode)

    rot *= np.pi/180.
    w = wcs.WCS(naxis=2)

    # pixel -> intermediate pixel
    w.wcs.crpix = [cent[0],cent[1]]
    w.wcs.pc=[[np.cos(rot), -1.*np.sin(rot)], [np.sin(rot), np.cos(rot)]]

    # intermediate pixel to intermediate worlds
    w.wcs.cdelt = np.array([0.0032*c.rsc[0], 0.0032*c.rsc[0]])

    # Simple Imaging Polynomial
    if mode=="mcs_pfi":
        order=7
        a=np.zeros((order+1,order+1))
        a[3][0]=c.rsc[1]/c.rsc[0] ; a[5][0]=c.rsc[2]/c.rsc[0] ; a[7][0]=c.rsc[3]/c.rsc[0]
        ap=np.zeros((order+1,order+1))
    else:
        order=5
        a=np.zeros((order+1,order+1))
        a[3][0]=c.rsc[1]/c.rsc[0] ; a[5][0]=c.rsc[2]/c.rsc[0]
        ap=np.zeros((order+1,order+1))

    b=np.swapaxes(a,0,1)
    bp=np.swapaxes(ap,0,1)
    s = wcs.Sip(a, b, ap, bp, w.wcs.crpix)

    # intermeriate worlds to worlds
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["PIXEL", "PIXEL"]
    #w.wcs.set_pv([(2, 1, 45.0)])

    # Others
    w.wcs.cunit=['mm','mm']

    return w,s

if __name__ == '__main__':

    # center position in pixel
    cent=[4480,2889]
    # roator angle in degree
    rot=0.
    # mode
    mode="mcs_pfi"
    # alt, az in degree
    alt=90. ; az=0.

    w,s=WCSParameters(mode,cent,rot, alt, az)

    print(w, s.a, s.b)
