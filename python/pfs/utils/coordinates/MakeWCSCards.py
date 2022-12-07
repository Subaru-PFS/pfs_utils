#!/usr/bin/env python

import numpy as np
from astropy import wcs
# from astropy.io import fits
# from astropy import units as u
# from astropy.time import Time
# from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from . import DistortionCoefficients as disco
from . import CoordTransp as coord


def get_wcs_mcs(cent, rot):
    """Get WCS parameters For PFSC (MCS) file (WCS on PFI coordinates)

    Parameters
    ----------
    cent : `np.ndarray`, (2, 1)
        The FoV center, i.e., telescope pointing center, (ra, dec) in unit
        of degree.
    rot : `float`
        Telescope instrument rotator angle in unit of degree.

    Returns
    -------
    w: `astropy.wcs.WCS`
        registered wcs keywords
    s: `astropy.wcs.Sip`
        registered SIP keywords
    """

    c = disco.Coeff('mcs_pfi')
    rot *= np.pi/180.
    w = wcs.WCS(naxis=2)

    # pixel -> intermediate pixel
    w.wcs.crpix = [cent[0], cent[1]]
    w.wcs.pc = [[np.cos(rot), -1.*np.sin(rot)], [np.sin(rot), np.cos(rot)]]

    # intermediate pixel to intermediate worlds
    w.wcs.cdelt = np.array([disco.mcspixel*c.rsc[0], disco.mcspixel*c.rsc[0]])

    # Simple Imaging Polynomial (no inverse func is defined)
    # Here, r~x~y is assumed
    order = 7
    a = np.zeros((order+1, order+1))
    a[3][0] = c.rsc[1]/c.rsc[0]
    a[5][0] = c.rsc[2]/c.rsc[0]
    a[7][0] = c.rsc[3]/c.rsc[0]
    ap = np.zeros((order+1, order+1))

    b = np.swapaxes(a, 0, 1)
    bp = np.swapaxes(ap, 0, 1)
    s = wcs.Sip(a, b, ap, bp, w.wcs.crpix)

    # intermeriate worlds to worlds
    w.wcs.crval = [0., 0.]
    w.wcs.ctype = ["PIXEL", "PIXEL"]
    # w.wcs.set_pv([(2, 1, 45.0)])

    # Others
    w.wcs.cunit = ['mm', 'mm']

    return w, s


def WCSParameters(mode, cent, rot, alt, az):

    w, s = get_wcs_mcs(cent, rot)

    return w, s


def get_wcs_ag(cent, rot, alt, az, pa=0., icam=0, time='2020-01-01 10:00:00'):
    """Get WCS parameters For PFSD (AG) file (WCS on sky coordinates)

    Parameters
    ----------
    cent : `np.ndarray`, (2, 1)
        The FoV center, i.e., telescope pointing center, (ra, dec) in unit
        of degree.
    rot : `float`
        Telescope instrument rotator angle in unit of degree.
    alt: `float`
        Telescope altitude in unit of degree.
    az: `float`
        Telescope azimuth in unit of degree.
    pa: `float`, optional
        Telescope Position Angle in unit of degree.
        Default = 0.
    icam: `int`, optional
        AG camera ID. (i.e.,  GuideStars.agId)
        Default = 0
    time : `str`, optional
        Observation time UTC in format of %Y-%m-%d %H:%M:%S
        Defalt is 2020-01-01 00:00:00

    Returns
    -------
    w: `astropy.wcs.WCS`
        created wcs keywords
    """
    c = disco.Coeff('pfi_sky')

    rot = (icam*-60. - pa)*np.pi/180.
    w = wcs.WCS(naxis=2)

    # pixel -> intermediate pixel
    w.wcs.crpix = [536.5, 521.5]
    w.wcs.pc = [[np.cos(rot), -1.*np.sin(rot)], [np.sin(rot), np.cos(rot)]]

    # intermediate pixel to intermediate worlds
    xpfi, ypfi = coord.ag_pixel_to_pfimm(icam, 511.5, 511.5)
    xyin = np.array([[xpfi], [ypfi]])
    # print(xyin)
    # pixel scale in sky
    scale_pix = c.rsc[0] * disco.agpixel

    # intermediate worlds to worlds
    raise RuntimeError("get_wcs_ag is disabled waiting for bugfix")
    xyout = coord.CoordinateTransform(xyin, "pfi_sky", za=0., time=time,
                                      cent=cent, pa=pa)
    # print(xyout)
    w.wcs.cdelt = np.array([-1.*scale_pix, -1.*scale_pix])

    w.wcs.crval = [xyout[0, 0], xyout[1, 0]]

    w.wcs.ctype = ['RA', 'DEC']
    w.wcs.cunit = ['deg', 'deg']

    return w


if __name__ == '__main__':

    # center position in pixel
    cent = [4480, 2889]
    # roator angle in degree
    rot = 0.
    # mode
    mode = "mcs_pfi"
    # alt, az in degree
    alt = 90.
    az = 0.

    w, s = WCSParameters(mode, cent, rot, alt, az)

    print(w, s.a, s.b)
