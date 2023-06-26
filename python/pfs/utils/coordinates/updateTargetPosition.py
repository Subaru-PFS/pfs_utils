#!/usr/bin/env python

from locale import dcgettext
import os
import logging
import numpy as np
from scipy import interpolate as ipol

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astroplan import Observer
import astropy.coordinates as ascor

from scipy.spatial.transform import Rotation as R


from .CoordTransp import CoordinateTransform
from . import Subaru_POPT2_PFS


mypath = os.path.dirname(os.path.abspath(__file__))+'/'

def update_target_position(radec, pa, cent, pm, par, obstime, epoch=2016.0):

    """ Update target position at the time of observation

    Parameters
    ----------
    radec : `np.ndarray`, (2, N)
        Input coordinates (Ra, Dec) in pfsDesgin (ra, dec).
    pa : `float`
        Position angle in unit of degree in pfsDesign (posAng)
    cent : `np.ndarray`, (2, 1)
        The boresight (Ra, Dec) in pfsDesign (raBoresight, decBoresight).
    pm: `np.ndarray`, (2, N)
        The proper motion of the targets. The unit is mas/yr .
    par: `np.ndarray`, (1, N), optional
        The parallax of the cordinatess. The unit is mas.
    obstime : `str`
        Observation time UTC in format of %Y-%m-%d %H:%M:%S
    epoch : `float`, optional
        Reference epoch of the sky catalogue. Default is 2016 (Gaia DR3)
        pfsDesign

    Returns
    -------
    ra_now : `np.ndarray`, (1, N)
        RA at the time of observation. Unit is degree
    dec_now : `np.ndarray`, (1, N)
        Dec at the time of obserbation. Unit is degree
    pfi_now_x : `np.ndarray`, (1, N)
        RA at the time of observation. Unit is mm
    pfi_now_y : `np.ndarray`, (1, N)
        Dec at the time of obserbation. Unit is mm
    """

    subaru = Subaru_POPT2_PFS.Subaru()
    ra_now, dec_now = subaru.radec2radecplxpm(epoch, radec[0,:], radec[1,:],
                                      par, pm[0, :], pm[1, :], obstime)
    '''
    d, d, d, ra_now, dec_now = radec_to_subaru(radec[0,:], radec[1,:], pa, obstime,
                                               epoch, pm[0,:], pm[1,:], par, 
                                               inr=None, returnRaDec=True)
    '''

    xyout = CoordinateTransform(radec, 'sky_pfi', pa=pa, cent=cent,
                                pm=pm, par=par, time=obstime)

    pfi_now_x, pfi_now_y = xyout[0,:], xyout[1,:]

    return ra_now, dec_now, pfi_now_x, pfi_now_y
