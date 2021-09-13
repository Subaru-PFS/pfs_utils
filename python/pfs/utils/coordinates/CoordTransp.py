#!/usr/bin/env python

import os
import logging
import numpy as np
from scipy import interpolate as ipol

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, SkyOffsetFrame
from astroplan import Observer
import astropy.coordinates as ascor

from scipy.spatial.transform import Rotation as R

# Dictionary keys ( argument name is mode)
# sky_pfi : sky to F3C
# sky_pfi_hsc : sky to hsc focal plane
# sky_mcs : sky to MCS
# pfi_mcs : F3C to MCS
# pfi_mcs_wofe : F3C to MCS (w/o Field Element)
# mcs_pfi : MCS to F3C

from . import DistortionCoefficients as DCoeff

mypath = os.path.dirname(os.path.abspath(__file__))+'/'


def CoordinateTransform(xyin, za, mode, inr=0., pa=0.,
                        cent=np.array([[0.], [0.]]),
                        time='2020-01-01 10:00:00'):
    """Transform Coordinates with given observing conditions.

    Parameters
    ----------
    xyin : `np.ndarray`, (2, N)
        Input coordinates. Namely. (Ra, Dec) in unit of degree for sky, 
        (X, Y) in unit of mm for PFI, and (X, Y) in unit of pixel for MCS.
    za : `float`
        Zenith angle in unit of degree.
    mode : `str`
        Transformation mode. Available mode is "sky_pfi", "sky_pfi_hsc"
        "pfi_mcs", "pfi_mcs_wofe", "mcs_pfi"
    inr : `float`, optional
        Instrument rotator angle in unit of degree. Default is 0. Note that
        this value is automatically calculated and overwritten by this routine
        for sky_pfi* transformation.
    pa : `float`, optional
        Position angle in unit of degree for sky_pfi* transformation. Default
        is 0.
    cent : `np.ndarray`, (2, 1), optional
        The center of input coordinates in the same unit as xyin.
        Default is x=0. , y=0.
    time : `str`, optional
        Observation time UTC in format of %Y-%m-%d %H:%M:%S
        Defalt is 2020-01-01 00:00:00

    Returns
    -------
    xyout : `np.ndarray`, (8, N)
        Output coordinates etc.. The first two rows are the coordinates.
        Unit is degree for sky, mm for PFI, and pixel for MCS.
    """

    c = DCoeff.Coeff(mode)

    # Transform iput coordinates to those the same as WFC as-built model
    xyin, inr, za1 = convert_in_position(xyin, za, inr, pa, c, cent, time)
    if (mode == 'sky_pfi') and (za1 != za):
        logging.info("Zenith angle for your field should be %s", za1)
        za = za1

    # Calculate Argument
    arg = calc_argument(xyin, inr, c)

    # Scale conversion
    logging.info("Scaling")
    scale = c.scaling_factor(xyin)

    # deviation
    # base
    logging.info("Offset 1, at zenith")
    offx1, offy1 = c.offset_base(xyin)

    if c.skip2_off:
        offx2 = np.zeros(xyin.shape[1])
        offy2 = np.zeros(xyin.shape[1])
    else:
        # z-dependent
        logging.info("Offset 2, from zenith")
        offx2, offy2 = deviation_zenith_angle(xyin, za, c)

    xx = scale*np.cos(arg)+offx1 + offx2
    yy = scale*np.sin(arg)+offy1 + offy2

    xx, yy = convert_out_position(xx, yy, inr, c, cent, time)

    xyout = np.array([xx, yy, scale, arg, offx1, offy1, offx2, offy2])

    return xyout


def convert_out_position(x, y, inr, c, cent, time):
    """convert outputs position on WFC-as built model to those on the PFS
    coordinates.
    Parameters
    ----------
    x : `float`,
       input position in x-axis
    y : `float`,
       input position in y-axis
    inr : `float`
        Instrument rotator angle in unit of degree.
    c : `DCoeff` class
       Distortion Coefficients
    cent : `np.ndarray`, (2, 1)
        The center of input coordinates in the same unit of as xyin.
    time : `str`
        Observation time UTC in format of %Y-%m-%d %H:%M:%S

    Returns
    -------
    xx : `float`,
       converted position in x-axis
    yy : `float`,
       converted position in y-axis
    """
    # convert pixel to mm: mcs_pfi
    if c.mode == 'pfi_mcs' or c.mode == 'pfi_mcs_wofe':
        xx, yy = mm_to_pixel(x, y, cent)
    # Rotation to PFI coordinates
    elif c.mode == 'sky_pfi' or c.mode == 'sky_pfi_hsc':
        xx, yy = rotation(x, y, -1.*inr)
    elif c.mode == 'pfi_sky':  # WFC to Ra-Dec
        # Set Observation Site (Subaru)
        tel = EarthLocation.of_site('Subaru')
        obs_time = Time(time)

        aref_file = mypath+'data/Refraction_data_635nm.txt'
        atm_ref = np.loadtxt(aref_file)
        atm_interp = ipol.splrep(atm_ref[:, 0], atm_ref[:, 1], s=0)

        # Ra-Dec to Az-El (Center)
        coord_cent = SkyCoord(cent[0], cent[1], unit=u.deg)
        altaz_cent = coord_cent.transform_to(AltAz(obstime=obs_time,
                                                   location=tel))

        az0 = altaz_cent.az.deg
        el0 = altaz_cent.alt.deg

        # offser frame in WFC
        center = SkyCoord(0., 0., unit=u.deg)
        aframe = center.skyoffset_frame()
        coord = SkyCoord(x, y, frame=aframe, unit=u.deg,
                         obstime=obs_time, location=tel)

        logging.info("(%s %s)", az0, el0)

        r = R.from_euler('ZYZ', [az0, -1*el0, 0.], degrees=True)
        xc, yc, zc = ascor.spherical_to_cartesian(1., np.deg2rad(y),
                                                  np.deg2rad(x))
        xyz = np.vstack((xc, yc, zc)).T
        logging.info("(%s)", xyz.shape)
        logging.info("(%s)", r.as_rotvec())
        azel = r.apply(xyz)
        rs, lats, lons = ascor.cartesian_to_spherical(azel[:, 0], azel[:, 1],
                                                      azel[:, 2])

        az = np.array(np.rad2deg(lons))
        el = np.array(np.rad2deg(lats))

        # eld = el - ipol.splev(90.-el, atm_interp)/3600.

        # Az-El to Ra-Dec (Targets)
        coord = SkyCoord(alt=el, az=az, frame='altaz', unit=u.deg,
                         obstime=obs_time, location=tel)
        radec = coord.transform_to('icrs')
        xx = radec.ra.deg
        yy = radec.dec.deg
    else:
        xx = x
        yy = y

    return xx, yy


def convert_in_position(xyin, za, inr, pa, c, cent, time):
    """convert input position to those on the same coordinates as
        the WFC as-built model.
    Parameters
    ----------
    xyin : `np.ndarray`, (2, N)
        Input coordinates, in unit of degree for sky, mm for PFI,
        and pixel for MCS
    za : `float`
        Zenith angle in unit of degree
    inr : `float`
        Instrument rotator angle in unit of degree.
    pa : `float`
        Position angle in unit of degree.
    c : `DCoeff` class
       Distortion Coefficients
    cent : `np.ndarray`, (2, 1)
        The center of input coordinates in the same unit as xyin.
    time : `str`
        Observation time UTC in format of %Y-%m-%d %H:%M:%S

    Returns
    -------
    xyconv : `np.ndarray`, (2, N)
       converted xy position in the format of WFC as-built model
    """

    # convert pixel to mm: mcs_pfi and mcs_pfi_asrd
    if (c.mode == 'mcs_pfi') or (c.mode == 'mcs_pfi_wofe'):
        xyconv = pixel_to_mm(xyin, inr, cent,
                             pix=DCoeff.mcspixel, invx=1., invy=-1.)
    elif c.mode == 'mcs_pfi_asrd':
        xyconv = pixel_to_mm(xyin, inr, cent,
                             pix=DCoeff.mcspixel_asrd, invx=-1., invy=1.)
    elif (c.mode == 'sky_pfi') or (c.mode == 'sky_pfi_hsc'):
        # Set Observation Site (Subaru)
        tel = EarthLocation.of_site('Subaru')
        tel2 = Observer.at_site("Subaru", timezone="US/Hawaii")
        obs_time = Time(time)

        aref_file = mypath+'data/Refraction_data_635nm.txt'
        atm_ref = np.loadtxt(aref_file)
        atm_interp = ipol.splrep(atm_ref[:, 0], atm_ref[:, 1], s=0)

        # Ra-Dec to Az-El (Center)
        coord_cent = SkyCoord(cent[0], cent[1], unit=u.deg)
        altaz_cent = coord_cent.transform_to(AltAz(obstime=obs_time,
                                                   location=tel))

        # Instrument rotator angle
        paa = tel2.parallactic_angle(obs_time, coord_cent).deg
        lat = tel2.location.lat.deg
        dc = coord_cent.dec.deg
        if dc > lat:
            inr = paa + pa - 180.
        else:
            inr = paa - pa

        az0 = altaz_cent.az.deg
        el0 = altaz_cent.alt.deg
        za = 90. - el0
        eld0 = el0 + ipol.splev(za, atm_interp)/3600.
        try:
            za = za[0]
        except IndexError:
            pass

        # define WFC frame
        center = SkyCoord(az0, eld0, unit=u.deg)
        aframe = center.skyoffset_frame()
        logging.info("FoV center: Ra,Dec=(%s %s) is Az,El,InR=(%s %s %s)",
                     cent[0], cent[1], az0, eld0, inr)

        # Ra-Dec to Az-El (Targets)
        coord = SkyCoord(xyin[0, :], xyin[1, :], unit=u.deg)
        altaz = coord.transform_to(AltAz(obstime=obs_time, location=tel))
        el = altaz.alt.deg
        az = altaz.az.deg

        eld = el + ipol.splev(90.-el, atm_interp)/3600.

        # Az-El to offset angle from the center (Targets)
        target = SkyCoord(az, eld, unit=u.deg)
        off = target.transform_to(aframe)
        xyconv = np.vstack((off.lon.deg, off.lat.deg))
    elif c.mode == 'pfi_sky':  # Rotate PFI to WFC
        # Set Observation Site (Subaru)
        tel = EarthLocation.of_site('Subaru')
        tel2 = Observer.at_site("Subaru", timezone="US/Hawaii")
        obs_time = Time(time)

        # Ra-Dec to Az-El (Center)
        coord_cent = SkyCoord(cent[0], cent[1], unit=u.deg)

        # Instrument rotator angle
        paa = tel2.parallactic_angle(obs_time, coord_cent).deg
        lat = tel2.location.lat.deg
        dc = coord_cent.dec.deg
        if dc > lat:
            inr = paa + pa - 180.
        else:
            inr = paa - pa

        xx, yy = rotation(xyin[0, :], xyin[1, :], inr)
        xyconv = np.vstack((xx, yy))
    else:
        xyconv = xyin

    return xyconv, inr, za


# differential : z
def deviation_zenith_angle(xyin, za, c):
    """Calculate displacement at a given zenith angle
    Parameters
    ----------
    xyin : `np.ndarray`, (2, N)
        Input coordinates in unit of degree for sky, mm for PFI,
        and pixel for MCS
    za : `float`
        Zenith angle in unit of degree
    c : `DCoeff` class
       Distortion Coefficients

    Returns
    -------
    offx : `np.ndarray`, (1, N)
        Displacement in x-axis
    offy : `np.ndarray`, (1, N)
        Displacement in y-axis
    """

    if c.mode == 'mcs_pfi' or c.mode == 'mcs_pfi_wofe':
        coeffzx, coeffzy = c.diff_coeff(za, 60.)
    else:
        # Reference: zenith angle 30 deg
        coeffzx, coeffzy = c.diff_coeff(za, 30.)

    # y : slope cy5(z) * y
    za_a = [0., 30., 60.]
    sl_a = c.slp

    sl_itrp = ipol.splrep(za_a, sl_a, k=2, s=0)
    cy5 = ipol.splev(za, sl_itrp)

    if c.mode == 'mcs_pfi' or c.mode == 'mcs_pfi_wofe':
        tarr = np.array([rotation_pattern(za, x, y) for x, y in zip(*xyin)])
        rotxy = tarr.transpose()
        offx = np.array([coeffzx*(c.dev_pattern_x(x, y))
                         for x, y in zip(*rotxy)])
        offy = np.array([coeffzy*(c.dev_pattern_y(x, y)) + cy5 * y
                         for x, y in zip(*rotxy)])
    else:
        offx = np.array([coeffzx*(c.dev_pattern_x(x, y))
                         for x, y in zip(*xyin)])
        offy = np.array([coeffzy * (c.dev_pattern_y(x, y)) + cy5 * y
                         for x, y in zip(*xyin)])

    return offx, offy


def rotation_pattern(za, x, y):
    """Calculate rotate displacement at a given angle

    Parameters
    ----------
    za : `float`
        Zenith angle in unit of degree.
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
    This function is implemented from study in Sep. 2018. However,
    Recent study in 2019 and 2020 found that rotation of displacement
    is not needed. This function will be removed in the near future.
    """

    ra = -1.*(60.-za)

    rx, ry = rotation(x, y, ra)

    return rx, ry


def calc_argument(xyin, inr, c):
    """Calculate argument angle of input position
    Parameters
    ----------
    xyin : `np.ndarray`, (2, N)
        Input coordinates in  unit of degree for sky, mm for PFI,
        and pixel for MCS
    inr : `float`
        Instrument rotator angle in unit of degree.
    c : `DCoeff` class
       Distortion Coefficients

    Returns
    -------
    arg : `np.ndarray`, (1, N)
       argument angle of positions in unit of radian
    """

    arg = np.array([np.arctan2(j, i) for i, j in zip(*xyin)])
    # MCS and PFI : xy is flipped
    if c.mode == 'mcs_pfi' or c.mode == 'mcs_pfi_wofe':
        arg = arg+np.pi

    # PFI to MCS: input argument depends on rotator angle
    if c.mode == 'pfi_mcs' or c.mode == 'pfi_mcs_wofe':
        arg = arg+np.deg2rad(inr)+np.pi

    return arg


def pixel_to_mm(xyin, inr, cent, pix=1., invx=1., invy=1.):
    """Convert MCS Unit from pixel to mm

    Parameters
    ----------
    xyin : `np.ndarray`, (2, N)
        Input coordinates in unit of pixel.
    inr : `float`, optional
        Instrument rotator angle in unit of degree. Default is 0.
    cent : `np.ndarray`, (2, 1), optional
        The center of input coordinates.
    pix : `float`, optional
        pixel scale in unit of mm/pix
    invx : `float`, optional
        Invert x axis (-1.) or not (1.). Default is No (1.).
    invy : `float`, optional
        Invert y axis (-1.) or not (1.). Default is No (1.).

    Returns
    -------
    xyin : `np.ndarray`, (2, N)
        Output coordinates in unit of mm.
    """

    offxy = xyin - cent

    xymm = []
    for x, y in zip(*offxy):
        rx, ry = rotation(x, y, inr)
        rx *= pix
        ry *= pix
        xymm.append([invx*rx, invy*ry])

    xymm = np.swapaxes(np.array(xymm, dtype=float), 0, 1)

    return xymm


def mm_to_pixel(x, y, cent):
    """Convert MCS Unit from mm to pixel.

    Parameters
    ----------
    x : `float`
        Input coordinates in x-axis in unit of mm.
    y : `float`
        Input coordinates in y-axis in unit of mm.
    cent : `np.ndarray`, (2, 1), optional
        The center of input coordinates.

    Returns
    -------
    sx : `float`
        Coordinates in x-axis in unit of pixel.
    sy : `float`
        Coordinates in y-axis in unit of pixel.
    """

    sx = x/DCoeff.mcspixel + cent[0]
    sy = (-1.)*y/DCoeff.mcspixel + cent[1]

    return sx, sy


def ag_pfimm_to_pixel(icam, xpfi, ypfi):
    """Convert AG PFI coorrdinates to detector coordinates.

    Convert PFI Cartesian coordinates in mm to one of the AG detectors
    in pixel (1024 x 1024).
    Note that the pixel coordinates is on the imaging area of AG detecor.

    Parameters
    ----------
    icam : `int`
        The detector identifiers ([0, 5])
    xpfi : `float`
        The Cartesian coordinates x's of points on the focal plane in PFI
        coordinate system (mm)
    ypfi : `float`
        The Cartesian coordinates y's of points on the focal plane in PFI
        coordinate system (mm)

    Returns
    -------
    xag : `float`
        The Cartesian coordinates x's of points in AG detecor (pix)
    yag : `float`
        The Cartesian coordinates y's of points in AG detecor (pix)
    """

    x, y = rotation(xpfi, ypfi, icam*60.)
    xag = y/DCoeff.agpixel + 511.5
    yag = - (x - DCoeff.agcent)/DCoeff.agpixel + 511.5

    return xag, yag


def ag_pixel_to_pfimm(icam, xag, yag):
    """Convert AG detector coordinates to PFI coordinates.

    Convert AG detetor coordinates (1024 x 1024) in pixel to PFI Cartesian
    coordinates in mm.
    Note that the pixel coordinates is on the imaging area of AG detecor.

    Parameters
    ----------
    icam : `int`
        The detector identifiers ([0, 5])
    xag : `float`
        The Cartesian coordinates x's of points in AG detecor (pix)
    yag : `float`
        The Cartesian coordinates y's of points in AG detecor (pix)

    Returns
    -------
    xpfi : `float`
        The Cartesian coordinates x's of points on the focal plane in PFI
        coordinate system (mm)
    ypfi : `float`
        The Cartesian coordinates y's of points on the focal plane in PFI
        coordinate system (mm)
    """

    y = (xag - 511.5)*DCoeff.agpixel
    x = - (yag - 511.5)*DCoeff.agpixel + DCoeff.agcent
    xpfi, ypfi = rotation(x, y, icam*(-60.))

    return xpfi, ypfi


def rotation(x, y, rot):
    """Rotate position

    Parameters
    ----------
    x : `float`
        Input coordinates in x-axis.
    y : `float`
        Input coordinates in y-axis.
    rot : `float`
        Rotation angle in unit of degree.

    Returns
    -------
    rx : `float`
        Coordinates in x-axis.
    ry : `float`
        Coordinates in y-axis.
    """

    ra = np.deg2rad(rot)

    rx = np.cos(ra)*x - np.sin(ra)*y
    ry = np.sin(ra)*x + np.cos(ra)*y

    return rx, ry
