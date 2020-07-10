#!/usr/bin/env python

import os
import sys
import math as mt
import numpy as np
from scipy import interpolate as ipol

# Dictionary keys ( argument name is mode)
# sky_pfi : sky to F3C
# sky_pfi_hsc : sky to hsc focal plane
# sky_mcs : sky to MCS
# pfi_mcs : F3C to MCS
# pfi_mcs_wofe : F3C to MCS (w/o Field Element)
# mcs_pfi : MCS to F3C

from . import DistortionCoefficients as DCoeff

mypath = os.path.dirname(os.path.abspath(__file__))+'/'


# input x,y point list, zenith angle, mode, rotator angle, centerposition
def CoordinateTransform(xyin, za, mode, inr=0., cent=np.array([[0.], [0.]])):
    """ Transform Coordinates given za and inr

    Parameters
    ----------
    xyin : `np.ndarray`, (N, 2)
        Input coordinates.
        Unit is degree for sky, mm for PFI, and pixel for MCS.
    za : `float`
        Zenith angle in degree.
    mode : `str`
        Transformation mode. Available mode is "sky_pfi", "sky_pfi_hsc"
        "pfi_mcs", "pfi_mcs_wofe", "mcs_pfi"
    inr : `float`, optional
        Instrument rotator andle in degree. Default is 0.
    cent : `np.ndarray`, (1, 2), optional
        The center of input coordinates. Unit is the same as xyin.
        Defaut is x=0. , y=0.

    Returns
    -------
    xyout : `np.ndarray`, (N, 8)
        Output coordinates etc.. The first two rows are the coordinates.
        Unit is degree for sky, mm for PFI, and pixel for MCS.
    """

    c = DCoeff.Coeff(mode)

    # convert pixel to mm: mcs_pfi and mcs_pfi_asrd
    if (c.mode == 'mcs_pfi') or (c.mode == 'mcs_pfi_wofe'):
        xyin = Pixel2mm(xyin, inr, cent,
                        pix=DCoeff.mcspixel, invx=1., invy=-1.)
    elif c.mode == 'mcs_pfi_asrd':
        xyin = Pixel2mm(xyin, inr, cent,
                        pix=DCoeff.mcspixel_asrd, invx=-1., invy=1.)

    arg = np.array([np.arctan2(j, i) for i, j in zip(*xyin)])
    # MCS and PFI : xy is flipped
    if c.mode == 'mcs_pfi' or c.mode == 'mcs_pfi_wofe':
        arg = arg+np.pi

    # PFI to MCS: input argument depends on rotator angle
    if c.mode == 'pfi_mcs' or c.mode == 'pfi_mcs_wofe':
        arg = arg+np.deg2rad(inr)+np.pi

    # Scale conversion
    print("Scaling", file=sys.stderr)
    scale = ScalingFactor(xyin, c)

    # deviation
    # base
    if c.skip1_off:
        offx1 = np.zeros(xyin.shape[1])
        offy1 = np.zeros(xyin.shape[1])
    else:
        print("Offset 1", file=sys.stderr)
        offx1, offy1 = OffsetBase(xyin, c)

    if c.skip2_off:
        offx2 = np.zeros(xyin.shape[1])
        offy2 = np.zeros(xyin.shape[1])
    else:
        # z-dependent
        print("Offset 2", file=sys.stderr)
        offx2, offy2 = DeviationZenithAngle(xyin, za, c)

    xx = scale*np.cos(arg)+offx1+offx2
    yy = scale*np.sin(arg)+offy1+offy2

    # convert pixel to mm: mcs_pfi
    if c.mode == 'pfi_mcs' or c.mode == 'pfi_mcs_wofe':
        xx, yy = mm2Pixel(xx, yy, cent)

    # Rotation to PFI coordinates
    if c.mode == 'sky_pfi' or c.mode == 'sky_pfi_hsc':
        xx, yy = Rotation(xx, yy, -1.*inr)

    xyout = np.array([xx, yy, scale, arg, offx1, offy1, offx2, offy2])

    return xyout


# differential : z
def DeviationZenithAngle(xyin, za, c):
    """ Calculate displacement at a given zenith angle

    Parameters
    ----------
    xyin : `np.ndarray`, (N, 2)
        Input coordinates.
        Unit is degree for sky, mm for PFI, and pixel for MCS
    za : `float`
        Zenith angle in degree
    c : `DCoeff` class
       Distortion Coefficients

    Returns
    -------
    offx : `np.ndarray`, (N, 1)
        Displacement in x-axis
    offy : `np.ndarray`, (N, 1)
        Displacement in y-axis
    """

    if c.mode == 'mcs_pfi' or c.mode == 'mcs_pfi_wofe':
        # Reference: zenith angle 60 deg
        coeffzx = DiffCoeff(za, c, axis='x')/DiffCoeff(60., c, axis='x')
        coeffzy = DiffCoeff(za, c, axis='y')/DiffCoeff(60., c, axis='y')
    else:
        # Reference: zenith angle 30 deg
        coeffz = DiffCoeff(za, c)/DiffCoeff(30., c)

    # y : slope cy5(z) * y
    za_a = [0., 30., 60.]
    sl_a = c.slp

    sl_itrp = ipol.splrep(za_a, sl_a, k=2, s=0)
    cy5 = ipol.splev(za, sl_itrp)

    if c.mode == 'mcs_pfi' or c.mode == 'mcs_pfi_wofe':
        tarr = np.array([RotationPattern(za, c, x, y) for x, y in zip(*xyin)])
        rotxy = tarr.transpose()

        offx = np.array([coeffzx*(c.dev_pattern_x(x, y)) 
                         for x, y in zip(*rotxy)])
        offy = np.array([coeffzy*(c.dev_pattern_y(x, y)) + cy5 * y
                         for x, y in zip(*rotxy)])
    else:
        offx = np.array([coeffz*(c.dev_pattern_x(x, y))
                         for x, y in zip(*xyin)])
        offy = np.array([coeffz * (c.dev_pattern_y(x, y)) + cy5 * y
                         for x, y in zip(*xyin)])

    return offx, offy


def DiffCoeff(za, c, axis='x'):
    """ Calculate coefficients of displacement at a given zenith angle

    Parameters
    ----------
    za : `float`
        Zenith angle in degree.
    c : `DCoeff` class
        Distortion Coefficients.
    axix : `str`, optional
        Axis to calculate.

    Returns
    -------
    `float`
        Coefficianct.
    """

    za *= np.pi/180.
    if (c.mode == 'mcs_pfi' or c.mode == 'mcs_pfi_wofe') and (axis == 'y'):
        return c.dsc[2]*(c.dsc[3]*np.sin(za)+(1-np.cos(za)))
    else:
        return c.dsc[0]*(c.dsc[1]*np.sin(za)+(1-np.cos(za)))


def RotationPattern(za, c, x, y):
    """ Calculate rotate displacement at a given angle

    Parameters
    ----------
    za : `float`
        Zenith angle in degree.
    c : `DCoeff` class
        Distortion Coefficients.
    axix : `str`, optional
        Axis to calculate. Default is x-axis.
    x: `float`
        Position in x-axis.
    y: `float`
        Position in y-axis.

    Returns
    -------
    rx: `float`
        Position in x-axis.
    ry: `float`
        Position in y-axis.

    Note
    ----
    This function is implemented from stdy in Sep. 2018. However,
    Recent stuty in 2019 and 2020 found that rotation of displacement
    is not needed. This function will be removed in the near future.
    """

    ra = -1.*(60.-za)

    rx, ry = Rotation(x, y, ra)

    return rx, ry


def OffsetBase(xyin, c):
    """ Derive Displacement at the zenith

    Parameters
    ----------
    xyin : `np.ndarray`, (N, 2)
        Input coordinates.
        Unit is degree for sky, mm for PFI, and pixel for MCS.
    c : `DCoeff` class
        Distortion Coefficients.

    Returns
    -------
    offsetx : `np.ndarray`, (N, 1)
        Displacement in x-axis.
    offsety : `np.ndarray`, (N, 1)
        Displacement in y-axis.
    """

    # sky-x sky-y off-x off-y
    dfile = mypath+"data/offset_base_"+c.mode+".dat"
    IpolD = np.loadtxt(dfile).T

    x_itrp = ipol.SmoothBivariateSpline(IpolD[0, :], IpolD[1, :], IpolD[2, :],
                                        kx=5, ky=5, s=1)
    y_itrp = ipol.SmoothBivariateSpline(IpolD[0, :], IpolD[1, :], IpolD[3, :],
                                        kx=5, ky=5, s=1)

    print("Interpol Done.", file=sys.stderr)

    offsetx = np.array([x_itrp.ev(i, j) for i, j in zip(*xyin)])
    offsety = np.array([y_itrp.ev(i, j) for i, j in zip(*xyin)])

    return offsetx, offsety


def ScalingFactor(xyin, c):
    """ Derive Axi-symmetric scaling factor
        It consifts of polynomial component and additional component
        by iterpolation.

    Parameters
    ----------
    xyin : `np.ndarray`, (N, 2)
        Input coordinates.
        Unit is degree for sky, mm for PFI, and pixel for MCS.
    c : `DCoeff` class
        Distortion Coefficients.

    Returns
    -------
    scale : `float`
        Scaling factor.
    """

    dist = [mt.sqrt(i*i+j*j) for i, j in zip(*xyin)]

    # at ASRD
    if c.mode == 'mcs_pfi_asrd':
        dist1 = [mt.sqrt((i * c.rsc[0] - DCoeff.distc_x_asrd) *
                         (i * c.rsc[0] - DCoeff.distc_x_asrd) +
                         (j * c.rsc[0] - DCoeff.distc_y_asrd) *
                         (j * c.rsc[0] - DCoeff.distc_y_asrd))
                 for i, j in zip(*xyin)]
        scale = [r * c.rsc[0] - (c.rsc[1] +
                                 c.rsc[2] * s +
                                 c.rsc[3] * s * s +
                                 c.rsc[4] * np.power(s, 3.))
                 for r, s in zip(dist, dist1)]

    # at the Summit
    else:
        # scale1 : rfunction
        # scale2 : interpolation
        scale1 = [c.scaling_factor_rfunc(r) for r in dist]
        if c.mode != 'pfi_mcs' and c.mode != 'pfi_mcs_wofe':
            # Derive Interpolation function
            sc_intf = ScalingFactor_Inter(c)
            scale2 = ipol.splev(dist, sc_intf)

        if c.mode == 'pfi_mcs' or c.mode == 'pfi_mcs_wofe':
            scale = scale1
        else:
            scale = np.array([x+y for x, y in zip(scale1, scale2)])

    return scale


def ScalingFactor_Inter(c):
    """ Calculate additional component of the scaling factor
        Using interpolation

    Parameters
    ----------
    c : `DCoeff` class
        Distortion Coefficients

    Returns
    -------
    r_itrp : `float`
        Scaling factor (additional component)
    """

    dfile = mypath+"data/scale_interp_"+c.mode+".dat"
    IpolD = np.loadtxt(dfile)

    r_itrp = ipol.splrep(IpolD[:, 0], IpolD[:, 1], s=0)

    return r_itrp


def Pixel2mm(xyin, inr, cent, pix=1., invx=1., invy=1.):
    """ Convert MCS Unit from pixel to mm

    Parameters
    ----------
    xyin : `np.ndarray`, (N, 2)
        Input coordinates in pixel.
    inr : `float`, optional
        Instrument rotator andle in degree. Default is 0.
    cent : `np.ndarray`, (1, 2), optional
        The center of input coordinates.
    pix : `float`, optional
        pixel scale in mm/pix
    invx : `float`, optional
        Invert x axis (-1.) or not (1.). Default is No (1.).
    invy : `float`, optional
        Invert y axis (-1.) or not (1.). Default is No (1.).

    Returns
    -------
    xyin : `np.ndarray`, (N, 2)
        Output coordinates in mm.
    """

    offxy = xyin - cent

    xymm = []
    for x, y in zip(*offxy):
        rx, ry = Rotation(x, y, inr)
        rx *= pix
        ry *= pix
        xymm.append([invx*rx, invy*ry])

    xymm = np.swapaxes(np.array(xymm, dtype=float), 0, 1)

    return xymm


def mm2Pixel(x, y, cent):
    """ Convert MCS Unit from mm to pixel.

    Parameters
    ----------
    x : `float`
        Input coordinates in x-axis in mm.
    y : `float`
        Input coordinates in y-axis in mm.
    cent : `np.ndarray`, (1, 2), optional
        The center of onput coordinates.

    Returns
    -------
    sx : `float`
        Coordinates in x-axis in pixel.
    sy : `float`
        Coordinates in y-axis in pixel.
    """

    sx = x/DCoeff.mcspixel + cent[0]
    sy = (-1.)*y/DCoeff.mcspixel + cent[1]

    return sx, sy


def Rotation(x, y, rot):
    """ Rotate position

    Parameters
    ----------
    x : `float`
        Input coordinates in x-axis.
    y : `float`
        Input coordinates in y-axis.
    rot : `float`
        Rotation angle in degree.

    Returns
    -------
    rx : `float`
        Coordinates in x-axis.
    ry : `float`
        Coordinates in y-axis.
    """

    ra = np.deg2rad(rot)

    rx = np.cos(ra)*x-np.sin(ra)*y
    ry = np.sin(ra)*x+np.cos(ra)*y

    return rx, ry
