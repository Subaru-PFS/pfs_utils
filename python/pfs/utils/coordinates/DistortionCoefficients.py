#!/usr/bin/env python

import os
import logging
import numpy as np
from scipy import interpolate as ipol

"""
Distortion Coefficients
 ver 1.0 : based on spot data in May 2017
 ver 2.0 : based on spot data in September 2018
           only coefficients from MCS to F3C (mcs_pfi) were changed
 ver 3.0 :  (2019.06.20)
           add F3C to MCS (pfi_mcs and pfi_mcs_wofe) coefficients using data
           in September 2018
 ver 4.0 :  (2019.02.20)
           add temporary MCS to F3C at ASRD (mcs_pfi_asrd) coefficients using
           measurement at ASRD
 ver 5.0 :  (2019.02.23)
           add MCS to F3C withot Field Element (mcs_pfi_wofe) coefficients
           using data in January 2020
 ver 6.0 :  (2020.11.24)
           update scale function of sky-pfi coordinates with 7th order,
           and add scale function of pfi_sky transformation.
           Here, old data in May 2017 was used
"""

"""
Global Parameters
"""

mypath = os.path.dirname(os.path.abspath(__file__))+'/'

# MCS pixel scale: 3.2um/pixel
mcspixel = 3.2e-3
mcspixel_asrd = 3.1e-3

# Distortion Center of Temporary MCS on PFI coordinates
# distc_x_asrd=-15.5764 -(-1.568) ; distc_y_asrd = -2.62071 -(1.251)
distc_x_asrd = -15.5764
distc_y_asrd = -2.62071

# AG pixel scale : 13um/pixel
agpixel = 1.3e-2
agcent = 241.292  # mm

# PFI InR offset from the telescope telescope
inr_pfi = -90
inr_min = -180
inr_max = +270

class Coeff:
    """Distortion Coefficients and Methods

    Parameters
    ----------
    mode : `str`
        Coordinate transportation mode. Availables are
        sky_pfi : sky to F3C
        sky_pfi_hsc : sky to hsc focal plane
        sky_mcs : sky to MCS
        pfi_sky : F3C to sky
        pfi_mcs : F3C to MCS
        pfi_mcs_wofe : F3C to MCS w/o the field element
        mcs_pfi : MCS to F3C
        mcs_pfi_wofe : MCS to F3C w/o the field element
    """

    def __init__(self, mode):

        # Differential pattern
        # x : dx = c0*x*y or = c0*x*x + c1*y*y + c2*x*y +c3*x^3*y^3+c4
        dic_cx = {"sky_pfi": [0., 0., 0.0750466788561, 0., 0.],
                  "sky_pfi_hsc": [0., 0., 0.06333788071803437, 0., 0.],
                  "pfi_sky": np.nan,
                  "pfi_mcs": np.nan,
                  "pfi_mcs_wofe": np.nan,
                  "mcs_pfi": [-1.1708420586425107e-05,
                                   -2.38300051844927e-06,
                                   0.0006432312579453259,
                                   7.960541263988616e-08,
                                   0.0003132127638028303],
                  "mcs_pfi_wofe": [-1.1708420586425107e-05,
                                   -2.38300051844927e-06,
                                   0.0006432312579453259,
                                   7.960541263988616e-08,
                                   0.0003132127638028303],
                  "mcs_pfi_asrd": []}
        # ADC diff
        dic_cx2 = {"sky_pfi": 0.,
                   "sky_pfi_hsc": [-3.6093014110703635e-05,
                                   -3.5711582686051736e-05,
                                   0.036725289727567695,
                                   0.16118982538765803,
                                   2.2543088456011478e-05],
                   "pfi_mcs": np.nan,
                   "pfi_mcs_wofe": np.nan,
                   "mcs_pfi": [-3.219708114471702e-07,
                                    -2.428020741979396e-07,
                                    1.1*0.0002491242718225472,
                                    5.037758838480642e-08,
                                    2.4243867461498993e-05],
                   "mcs_pfi_wofe": [-3.219708114471702e-07,
                                    -2.428020741979396e-07,
                                    1.1*0.0002491242718225472,
                                    5.037758838480642e-08,
                                    2.4243867461498993e-05],
                   "mcs_pfi_asrd": []}
        # y : dy = c0*x^2 + c1*y^2 + c2*y^4 + c3*x^2*y^2 + c4
        dic_cy = {"sky_pfi": [0.045278345420470184,
                              0.,
                              0.094587722958699313,
                              0.063632568952111501,
                              0.074033633816049674,
                              -0.023152187620119963],
                  "sky_pfi_hsc": [0.03094074673276881,
                                  0.07592444641651051,
                                  0.04699568361118014,
                                  0.05799533522450577,
                                  -0.01739380213343191],
                  "pfi_sky": [],
                  "pfi_mcs": [],
                  "pfi_mcs_wofe": [],
                  "mcs_pfi": [0.0002830462227831256,
                                   7.280407331455539e-07,
                                   0.0008204841449211456,
                                   3.457009571551328e-06,
                                   4.215526845313614e-06,
                                   -0.02704375829046644],
                  "mcs_pfi_wofe": [0.0002830462227831256,
                                   7.280407331455539e-07,
                                   0.0008204841449211456,
                                   3.457009571551328e-06,
                                   4.215526845313614e-06,
                                   -0.02704375829046644],
                  "mcs_pfi_asrd": []}
        # ADC differential pattern
        dic_cy2 = {"sky_pfi": [],
                   "sky_pfi_hsc": [0.036348008722863305,
                                   0.017639216366374662,
                                   0.06305057726030829,
                                   0.05447186091101391,
                                   0.07318081631882754,
                                   -0.01734138863579538],
                   "pfi_mcs": [],
                   "pfi_mcs_wofe": [],
                   "mcs_pfi": [1.1*0.0002468250920671912, 
                                    8.156925097149555e-07,
                                    1.1*0.00042927311964389167,
                                    2.4798055201049686e-06,
                                    3.3884450932813327e-06,
                                    -0.017404771609892535],
                   "mcs_pfi_wofe": [1.1*0.0002468250920671912, 
                                    8.156925097149555e-07,
                                    1.1*0.00042927311964389167,
                                    2.4798055201049686e-06,
                                    3.3884450932813327e-06,
                                    -0.017404771609892535],
                   "mcs_pfi_asrd": []}
        # Whether to skip use offset base
        skip1_off = {"sky_pfi": False,
                     "sky_pfi_hsc": False,
                     "pfi_sky": False,
                     "pfi_mcs": True,
                     "pfi_mcs_wofe": True,
                     "mcs_pfi": False,
                     "mcs_pfi_wofe": False,
                     "mcs_pfi_asrd": False}
        # Whether to skip use differential pattern
        skip2_off = {"sky_pfi": False,
                     "sky_pfi_hsc": False,
                     "pfi_sky": True,
                     "pfi_mcs": True,
                     "pfi_mcs_wofe": True,
                     "mcs_pfi": False,
                     "mcs_pfi_wofe": False,
                     "mcs_pfi_asrd": True}

        # y : slope cy5(z) * y
        dic_slp = {"sky_pfi": [0.,
                               -0.00069803799551166809,
                               -0.0048414570302028372],
                   "sky_pfi_hsc": [0.,
                                   -0.00015428320350572582,
                                   -7.358026105602118e-05],
                   "pfi_sky": [],
                   "pfi_mcs": [],
                   "pfi_mcs_wofe": [],
                   "mcs_pfi": [0.,
                                    -3.717227234445596e-05,
                                    -1.2506301957697883e-05],
                   "mcs_pfi_wofe": [0.,
                                    -3.717227234445596e-05,
                                    -1.2506301957697883e-05],
                   "mcs_pfi_asrd": []}

        # Scaling factor of difference
        dic_dsc = {"sky_pfi": [0.995465,
                               1.741161],
                   "sky_pfi_hsc": [0.116418996679199,
                                   16.9113776962136],
                   "pfi_sky": [],
                   "pfi_mcs": [],
                   "pfi_mcs_wofe": [],
                   "mcs_pfi": [-15.919527343063823,
                                    -0.07526314764972708,
                                    -17.130921960367584,
                                    -0.06975537122267313],
                   "mcs_pfi_wofe": [-15.919527343063823,
                                    -0.07526314764972708,
                                    -17.130921960367584,
                                    -0.06975537122267313],
                   "mcs_pfi_asrd": []}

        # Scaling factor w.r.t radius
        # mcs_pfi_asrd :
        # scale factor + distortion (3rd order from distotion center)
        dic_rsc = {"sky_pfi": [319.8849288008787,
                               15.455002704868093,
                               2.521741084754467,
                               4.231324795167893],
                   "sky_pfi_hsc": [320.0107640966762,
                                   14.297123404641752,
                                   6.586014956556028,
                                   0.],
                   "pfi_sky": [0.003126049416499188,
                               -1.4608970079526195e-09,
                               -1.1573407216548699e-15,
                               -1.3201189804972364e-20],
                   "pfi_mcs": [0.03802405803382136,
                               1.7600554217409373e-08,
                               -1.5297627440776523e-14,
                               1.6143531343033776e-19],
                   "pfi_mcs_wofe": [0.038019509493361525,
                                    -1.7569114332967838e-08,
                                    -1.5633186312477447e-14,
                                    -1.5136575959136838e-19],
                   "mcs_pfi": [26.301766037195193,
                                    0.008484216901138097,
                                    1.0150811666775894e-05,
                                    1.0290642807331274e-07],
                   "mcs_pfi_wofe": [26.301766037195193,
                                    0.008484216901138097,
                                    1.0150811666775894e-05,
                                    1.0290642807331274e-07],
                   "mcs_pfi_asrd": [24.68498072,
                                    -0.0287145,
                                    0.001562543,
                                    -1.58048E-05,
                                    -4.11313E-08]}

        self.mode = mode
        self.cx = dic_cx[mode]
        self.cy = dic_cy[mode]
        self.cx2 = dic_cx2[mode]
        self.cy2 = dic_cy2[mode]
        self.skip1_off = skip1_off[mode]
        self.skip2_off = skip2_off[mode]
        self.slp = dic_slp[mode]
        self.dsc = dic_dsc[mode]
        self.rsc = dic_rsc[mode]

    def dev_pattern_x(self, x, y, adc=False):
        """Calc patterned deviation in x-axis

        Parameters
        ----------
        x : float
            position in x-axis
        y : float
            position in y-axis
        Returns
        -------
        dx : float
            deviation in x-axis
        """

        if adc:
            ccx = self.cx2
        else:
            ccx = self.cx

        dx = (ccx[0]*x*x +
              ccx[1]*y*y +
              ccx[2]*x*y +
              ccx[3]*x*x*x*y*y*y +
              ccx[4])

        return dx

    def dev_pattern_y(self, x, y, adc=False):
        """Calc patterned deviation in y-axis

        Parameters
        ----------
        x : float
            position in x-axis
        y : float
            position in y-axis
        Returns
        -------
        dy : float
            deviation in x-axis
        """

        if adc:
            ccy = self.cy2
        else:
            ccy = self.cy

        dy = (ccy[0]*x*x +
              ccy[1]*np.power(x, 4.) +
              ccy[2]*y*y +
              ccy[3]*np.power(y, 4.) +
              ccy[4]*x*x*y*y +
              ccy[5])

        return dy

    def diff_coeff(self, za, za0=60.):
        """Calculate coefficients of displacement at a given zenith angle

        Parameters
        ----------
        za : `float`
            Zenith angle in degree.
        za0 : `float`, optional
            Zenith angle in degree where Coeff is unity. Default is 60.

        Returns
        -------
        coeffx: `float`
            Coefficient in x-axis.
        coeffy: `float`
            Coefficient in y-axis.
        """

        za = np.deg2rad(za)
        if (self.mode == 'mcs_pfi' or self.mode == 'mcs_pfi_wofe'):
            coeffx = self.dsc[0]*(self.dsc[1]*np.sin(za) +
                                  (1-np.cos(za)))
            coeffy = self.dsc[2]*(self.dsc[3]*np.sin(za) +
                                  (1-np.cos(za)))
            coeffx0 = self.dsc[0]*(self.dsc[1]*np.sin(za0) +
                                   (1-np.cos(za0)))
            coeffy0 = self.dsc[2]*(self.dsc[3]*np.sin(za0) +
                                   (1-np.cos(za0)))
        else:
            coeffx = coeffy = \
                self.dsc[0]*(self.dsc[1]*np.sin(za)+(1-np.cos(za)))
            coeffx0 = coeffy0 = \
                self.dsc[0]*(self.dsc[1]*np.sin(za0)+(1-np.cos(za0)))

        coeffx /= coeffx0
        coeffy /= coeffy0

        return coeffx, coeffy

    def offset_base(self, xyin):
        """Derive Displacement at the zenith

        Parameters
        ----------
        xyin : `np.ndarray`, (N, 2)
            Input coordinates.
            Unit is degree for sky, mm for PFI, and pixel for MCS.

        Returns
        -------
        offsetx : `np.ndarray`, (N, 1)
            Displacement in x-axis.
        offsety : `np.ndarray`, (N, 1)
            Displacement in y-axis.
        """

        if self.skip1_off:
            logging.info("------ Skipped.")
            offsetx = offsety = np.zeros(xyin.shape[1])
        else:
            # sky-x sky-y off-x off-y
            dfile = mypath+"data/offset_base_"+self.mode+".dat"
            IpolD = np.loadtxt(dfile).T

            x_itrp = ipol.SmoothBivariateSpline(IpolD[0, :], IpolD[1, :],
                                                IpolD[2, :], kx=5, ky=5, s=1)
            y_itrp = ipol.SmoothBivariateSpline(IpolD[0, :], IpolD[1, :],
                                                IpolD[3, :], kx=5, ky=5, s=1)

            logging.info("Interpolated the base offset")

            offsetx = np.array([x_itrp.ev(i, j) for i, j in zip(*xyin)])
            offsety = np.array([y_itrp.ev(i, j) for i, j in zip(*xyin)])

        return offsetx, offsety

    def scaling_factor(self, xyin):
        """Derive axi-symmetric scaling factor
            It consifts of polynomial component and additional component
            by interpolation.

        Parameters
        ----------
        xyin : `np.ndarray`, (N, 2)
            Input coordinates.
            Unit is degree for sky, mm for PFI, and pixel for MCS.

        Returns
        -------
        scale : `float`
            Scaling factor.
        """

        dist = [np.sqrt(i*i+j*j) for i, j in zip(*xyin)]

        # at ASRD
        if self.mode == 'mcs_pfi_asrd':
            dist1 = [np.sqrt((i * self.rsc[0] - distc_x_asrd) *
                             (i * self.rsc[0] - distc_x_asrd) +
                             (j * self.rsc[0] - distc_y_asrd) *
                             (j * self.rsc[0] - distc_y_asrd))
                     for i, j in zip(*xyin)]
            scale = [r * self.rsc[0] - (self.rsc[1] +
                                        self.rsc[2] * s +
                                        self.rsc[3] * s * s +
                                        self.rsc[4] * np.power(s, 3.))
                     for r, s in zip(dist, dist1)]

        # at the Summit
        else:
            # scale1 : r-function
            # scale2 : interpolation
            scale1 = [self.scaling_factor_rfunc(r) for r in dist]

            if self.mode == 'pfi_mcs' or self.mode == 'pfi_mcs_wofe':
                scale2 = np.zeros(len(scale1))
            else:
                # Derive Interpolation function
                sc_intf = self.scaling_factor_inter()
                scale2 = ipol.splev(dist, sc_intf)

            scale = np.array([x+y for x, y in zip(scale1, scale2)])

        return scale

    # Scaling Factor: function of r
    def scaling_factor_rfunc(self, r):
        """Calculate polynomial component of the scaling factor

        Parameters
        ----------
        r : `float`
            Distance from the coordinate center

        Returns
        -------
        sf : `float`
            Scaling factor (polynomial component)
        """

        sf = (self.rsc[0]*r +
              self.rsc[1]*np.power(r, 3.) +
              self.rsc[2]*np.power(r, 5.) +
              self.rsc[3]*np.power(r, 7.))

        return sf

    def scaling_factor_inter(self):
        """Calculate additional component of the scaling factor
            Using interpolation

        Parameters
        ----------

        Returns
        -------
        r_itrp : `float`
            Scaling factor (additional component)
        """

        dfile = mypath+"data/scale_interp_"+self.mode+".dat"
        IpolD = np.loadtxt(dfile)

        r_itrp = ipol.splrep(IpolD[:, 0], IpolD[:, 1], s=0)

        return r_itrp
