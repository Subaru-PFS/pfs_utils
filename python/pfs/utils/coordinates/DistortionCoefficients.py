#!/usr/bin/env python

import os
import logging
import numpy as np
from scipy import interpolate as ipol

from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, FK5, Distance, SkyOffsetFrame
try:
    from astropy.coordinates import TETE
except ImportError:
    import astropy
    logging.warning("Unable to load TETE (old version of astropy? (%s)); using FK5", astropy.__version__)

    def TETE(obstime, location):
        return FK5(equinox=obstime.jyear_str)

from astroplan import Observer
import astropy.coordinates as ascor

from . import Subaru_POPT2_PFS

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
 ver 7.0 :  (2021.11.01)
           update with the latest version og spot calculation (only PFI<->MCS)
 ver 7.1 :  (2021.11.15)
           update with the latest version og spot calculation (Sky<->PFI)
 ver 8.0 : (2022.05)
           update based on 2021 Nov-run's analysis, implemented Kawanomoto's 
           library as default for sky-pfi
 ver 9.0 : (2022.19)
           update based on 2022 June & Sep run's analysis, implemented Kawanomoto's 
           library as default for sky-pfi
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

# AG camera parameters
# AG pixel scale : 13um/pixel
agpixel = 1.3e-2
agcent = 241.292  # mm
# AG centre position and rotation (measured in Nov 2021)
# dy[mm] dx[mm] dt[deg] for each camera in PFI coordinates
agcent_off = [[-0.405, 0.668, -0.253368], [-0.055, -0.081, 0.234505],
              [-0.357, -0.180, 0.329449], [0.270, -0.357, 0.416894],
              [0.444, -0.138, 0.0589071], [0.067, 0.077, 0.234977]]

# PFI InR offset from the telescope telescope
inr_pfi = -90  # deg
inr_min = -180  # deg
inr_max = +180  # deg
pfi_offx = -1.855  # mm
pfi_offy = -0.998  # mm
pfi_offrot = 0.765507  # deg
pfi_diffscale = 0.999232042

# correction during the Sep 2022 run
inr_tel_offset = 0.12  # deg (0.08 + 0.03 + 0.01)
pfi_x_offset = -0.115  # -0.09 -0.025  mm
pfi_y_offset = 0.03  # 0.01 + 0.02  mm

# Wavelength used in AG
wl_ag = 0.62


class Coeff:
    """Distortion Coefficients and Methods

    Parameters
    ----------
    mode : `str`
        Coordinate transportation mode. Availables are
        sky_pfi : sky to F3C (Kawanomoto's ver)
        sky_pfi_old : sky to F3C (Moritani's ver)
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
        """
        dic_cx = {"sky_pfi_old": [-0.0017327476322776293,
                              -   0.00035441489584601635,
                                  0.09508424178479086,
                                  0.25940659835632474,
                                  0.00031404409450775506],
        """
        dic_cx = {"sky_pfi_old": [0.,
                                  0.,
                                  0.,
                                  0.,
                                  0.],
                  "sky_pfi_hsc": [-0.0019537856994430326,
                                  -0.0003633426407904909,
                                  0.09744667192075979,
                                  0.2676659451936487,
                                  0.0003463495414575713],
                  "pfi_sky": np.nan,
                  "pfi_mcs": [-3.803415379857743e-10,
                              -1.242550157492245e-10,
                              -1.4789849377620123e-09,
                              1.3074411474901438e-18,
                              7.398678492518167e-06],
                  "pfi_mcs_wofe": np.nan,
                  "mcs_pfi": [-1.1700047234309595e-05,
                              -2.392062531716983e-06,
                              0.0006422692238954877,
                              7.987925712881243e-08,
                              0.000313912344805486],
                  "mcs_pfi_wofe": [-1.1708420586425107e-05,
                                   -2.38300051844927e-06,
                                   0.0006432312579453259,
                                   7.960541263988616e-08,
                                   0.0003132127638028303],
                  "mcs_pfi_asrd": []}
        # ADC diff
        dic_cx2 = {"sky_pfi_old": [-3.584787066577977e-05,
                                   -3.5799605301681134e-05,
                                   0.03659745888281277,
                                   0.16051286109719687,
                                   2.247351405551246e-05],
                   "sky_pfi_hsc": [-3.6093014110703635e-05,
                                   -3.5711582686051736e-05,
                                   0.036725289727567695,
                                   0.16118982538765803,
                                   2.2543088456011478e-05],
                   "pfi_sky": [],
                   "pfi_mcs": [-2.4899491113979943e-11,
                               -1.778441766247285e-11,
                               2.8833885926174088e-09,
                               -2.2080960521984518e-18,
                               5.921418946211664e-07],
                   "pfi_mcs_wofe": np.nan,
                   "mcs_pfi": [-2.4124629770100625e-07,
                               -2.414388469053491e-07,
                               0.00024720479272183967,
                               4.9442381536775584e-08,
                               2.244197954251821e-05],
                   "mcs_pfi_wofe": [-3.219708114471702e-07,
                                    -2.428020741979396e-07,
                                    1.1*0.0002491242718225472,
                                    5.037758838480642e-08,
                                    2.4243867461498993e-05],
                   "mcs_pfi_asrd": []}
        # y : dy = c0*x^2 + c1*y^2 + c2*y^4 + c3*x^2*y^2 + c4
        """
        dic_cy = {"sky_pfi_old [0.041746387732980325,
                              0.01608173945524268,
                              0.12094239424626228,
                              0.07654433799914458,
                              0.09301814497785414,
                              -0.027067648260184846],
        """
        dic_cy = {"sky_pfi_old": [0.,
                                  0.,
                                  0.,
                                  0.,
                                  0.,
                                  0.],
                  "sky_pfi_hsc": [0.04173443438068433,
                                  0.01619031569246855,
                                  0.12274785098042126,
                                  0.07863042407051903,
                                  0.09526105421465705,
                                  -0.027426594145212868],
                  "pfi_sky": [],
                  "pfi_mcs": [-1.6388064512086219e-09,
                              9.103026627957999e-15,
                              -4.256053569982049e-09,
                              4.3141135569462175e-14,
                              5.135716437956087e-14,
                              5.564416989761432e-05],
                  "pfi_mcs_wofe": [],
                  "mcs_pfi": [0.00028201389900424996,
                              7.325933341912196e-07,
                              0.000817008523730211,
                              3.488785431966954e-06,
                              4.24138148573218e-06,
                              -0.027067261140898205],
                  "mcs_pfi_wofe": [0.0002830462227831256,
                                   7.280407331455539e-07,
                                   0.0008204841449211456,
                                   3.457009571551328e-06,
                                   4.215526845313614e-06,
                                   -0.02704375829046644],
                  "mcs_pfi_asrd": []}
        # ADC differential pattern
        dic_cy2 = {"sky_pfi_old": [0.03623304902406039,
                                   0.0175673111666987,
                                   0.06285186439429795,
                                   0.05423970484228492,
                                   0.07287287217681873,
                                   -0.017282746333435548],
                   "sky_pfi_hsc": [0.036348008722863305,
                                   0.017639216366374662,
                                   0.06305057726030829,
                                   0.05447186091101391,
                                   0.07318081631882754,
                                   -0.01734138863579538],
                   "pfi_sky": [],
                   "pfi_mcs": [2.554808743825899e-09,
                               -1.6457667382905435e-14,
                               7.193759108156046e-09,
                               -7.315204126820892e-14,
                               -9.10019247701946e-14,
                               -8.973422557327273e-05],
                   "pfi_mcs_wofe": [],
                   "mcs_pfi": [0.0002447633276481989,
                               8.005351813232036e-07,
                               0.00042454305093323744,
                               2.473384758171295e-06,
                               3.324026612698092e-06,
                               -0.017282281242905196],
                   "mcs_pfi_wofe": [1.1*0.0002468250920671912,
                                    8.156925097149555e-07,
                                    1.1*0.00042927311964389167,
                                    2.4798055201049686e-06,
                                    3.3884450932813327e-06,
                                    -0.017404771609892535],
                   "mcs_pfi_asrd": []}
        # Whether to skip use offset base
        skip1_off = {"sky_pfi_old": False,
                     "sky_pfi_hsc": False,
                     "pfi_sky": False,
                     "pfi_mcs": False,
                     "pfi_mcs_wofe": True,
                     "mcs_pfi": False,
                     "mcs_pfi_wofe": False,
                     "mcs_pfi_asrd": False}
        # Whether to skip use differential pattern
        skip2_off = {"sky_pfi_old": False,
                     "sky_pfi_hsc": False,
                     "pfi_sky": True,
                     "pfi_mcs": False,
                     "pfi_mcs_wofe": True,
                     "mcs_pfi": False,
                     "mcs_pfi_wofe": False,
                     "mcs_pfi_asrd": True}
        # Whether to execute offset base 2
        do_off2 = {"sky_pfi": False,
                   "sky_pfi_hsc": False,
                   "pfi_sky": False,
                   "pfi_mcs": False,
                   "pfi_mcs_wofe": False,
                   "mcs_pfi": True,
                   "mcs_pfi_wofe": True,
                   "mcs_pfi_asrd": False}

        # y : slope cy5(z) * y
        dic_slp = {"sky_pfi_old": [0.,
                                   -0.00011423970952475685,
                                   0.00013767881534632514],
                   "sky_pfi_hsc": [0.,
                                   -0.00015428320350572582,
                                   -7.358026105602118e-05],
                   "pfi_sky": [],
                   "pfi_mcs": [],
                   "pfi_mcs_wofe": [],
                   "mcs_pfi": [0.,
                               9.048282335676767e-06,
                               -1.1893652243911522e-05],
                   "mcs_pfi_wofe": [0.,
                                    -3.717227234445596e-05,
                                    -1.2506301957697883e-05],
                   "mcs_pfi_asrd": []}

        # Scaling factor of difference
        dic_dsc = {"sky_pfi_old": [-0.07532961367738047,
                                   -15.905990388393876,
                                   -0.06990604717680293,
                                   -17.095242283006968],
                   "sky_pfi_hsc": [-0.06754143189515537,
                                   -17.673531175920985,
                                   -0.060597493431458524,
                                   -19.63260277208938],
                   "pfi_sky": [],
                   "pfi_mcs": [-0.31313751818329066,
                               -4.2648692387414116,
                               0.14923479607779552,
                               7.160124962641096],
                   "pfi_mcs_wofe": [],
                   "mcs_pfi": [-0.07532961367738047,
                               -15.905990388393876,
                               -0.06990604717680293,
                               -17.095242283006968],
                   "mcs_pfi_wofe": [-0.07526314764972708,
                                    -15.919527343063823,
                                    -0.06975537122267313,
                                    -17.130921960367584],
                   "mcs_pfi_asrd": []}

        # Scaling factor w.r.t radius
        # mcs_pfi_asrd :
        # scale factor + distortion (3rd order from distotion center)
        dic_rsc = {"sky_pfi_old": [319.88636084359314,
                                   15.416834272793494,
                                   2.688157915137708,
                                   4.028757396154106],
                   "sky_pfi_hsc": [319.91928151401225,
                                   15.432336769183166,
                                   2.711106670089066,
                                   4.009570559486747],
                   "pfi_sky": [0.003126041437325755,
                               -1.4582612651458824e-09,
                               -1.2674580781874949e-15,
                               -1.1943114562699083e-20],
                   "pfi_mcs": [0.0380323375150029,
                               -1.759579156066903e-08,
                               -1.5796484298836056e-14,
                               -1.5449737416030818e-19],
                   "pfi_mcs_wofe": [0.038019509493361525,
                                    -1.7569114332967838e-08,
                                    -1.5633186312477447e-14,
                                    -1.5136575959136838e-19],
                   "mcs_pfi": [26.29282427147882,
                               0.00849357480319668,
                               9.962945899388842e-06,
                               1.0586816061308735e-07],
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
        if mode != 'sky_pfi':
            self.cx = dic_cx[mode]
            self.cy = dic_cy[mode]
            self.cx2 = dic_cx2[mode]
            self.cy2 = dic_cy2[mode]
            self.skip1_off = skip1_off[mode]
            self.skip2_off = skip2_off[mode]
            self.do_off2 = do_off2[mode]
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
        za0 = np.deg2rad(za0)
        coeffx = self.dsc[0]*(self.dsc[1]*np.sin(za) + (1-np.cos(za)))
        coeffy = self.dsc[2]*(self.dsc[3]*np.sin(za) + (1-np.cos(za)))
        coeffx0 = self.dsc[0]*(self.dsc[1]*np.sin(za0) + (1-np.cos(za0)))
        coeffy0 = self.dsc[2]*(self.dsc[3]*np.sin(za0) + (1-np.cos(za0)))

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
            IpolD = IpolD[:, ~np.any(IpolD == -99999., axis=0)]

            x_itrp = ipol.SmoothBivariateSpline(IpolD[0, :], IpolD[1, :],
                                                IpolD[2, :], kx=5, ky=5, s=1.)
            y_itrp = ipol.SmoothBivariateSpline(IpolD[0, :], IpolD[1, :],
                                                IpolD[3, :], kx=5, ky=5, s=1.)

            logging.info("Interpolated the base offset")

            offsetx = np.array([x_itrp.ev(i, j) for i, j in zip(*xyin)])
            offsety = np.array([y_itrp.ev(i, j) for i, j in zip(*xyin)])

            if self.do_off2:
                # ADC=0
                dfile2 = mypath+"data/offset2_base_"+self.mode+"_map.dat"
                IpolD2 = np.loadtxt(dfile2).T
                IpolD2 = IpolD2[:, ~np.any(IpolD2 == -99999., axis=0)]

                x_itrp2 = ipol.SmoothBivariateSpline(IpolD2[0, :], IpolD2[1, :],
                                                     IpolD2[2, :], kx=5, ky=5, s=1)
                y_itrp2 = ipol.SmoothBivariateSpline(IpolD2[0, :], IpolD2[1, :],
                                                     IpolD2[3, :], kx=5, ky=5, s=1)

            logging.info("Interpolated the base offset")

            if self.do_off2:
                offsetx = np.array([x_itrp.ev(i, j)+x_itrp2.ev(i, j)
                                    for i, j in zip(*xyin)])
                offsety = np.array([y_itrp.ev(i, j)+y_itrp2.ev(i, j)
                                    for i, j in zip(*xyin)])
                rin = np.array([np.sqrt(i*i+j*j) for i, j in zip(*xyin)])
                # print(rin)
                # Out of Grid
                offsetx[rin > 9.46] = 0.
                offsety[rin > 9.46] = 0.
            else:
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

            if self.mode == 'pfi_mcs_wofe':
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

        r_itrp = ipol.splrep(IpolD[:, 0], IpolD[:, 1], s=1)

        return r_itrp


# General functions
def calc_m3pos(za):

    # for now, just use fixed position

    return 3.0


def calc_adc_position(za, id=106):

    if id == 106:  # Filter ID=106 (wave 9)
        e0 = 0.0027
        e1 = 12.6755
        e2 = -0.0992
        e3 = 0.2141
        e4 = -0.0901
    else:
        logging.info("Invalid mode. Set ADC=0.")
        e0 = e1 = e2 = e3 = e4 = 0.

    t = np.tan(np.deg2rad(za))
    adc = e0 + e1*t + e2*t*t + e3*t*t*t + e4*t*t*t*t

    return adc


def calc_atmospheric_refraction(za, wl=0.575):

    # parameters
    b = 0.00130
    # Maunakea
    p = 450    # mmHg
    g = 978.627  # gal
    t = 0.  # degC
    f = 1.  # mmHg (RH=20%)
    # function of wavelength [um]
    rho = 6432.8 + 2949810./(146. - 1./wl/wl) + 25540./(41. - 1./wl/wl)
    p1 = p*980.665/g
    rho2 = rho*p1*(1. + (1.049 - 0.0157*t)*0.000001*p1)/(720.883*(1. + 0.003661*t))
    rho3 = 0.00000001*(rho2 - f*(6.24 - 0.0680*wl*wl)/(1. + 0.003661*t))
    a0 = rho3 - rho3*b + 2.*rho3*b*b - 0.5*rho3*rho3*b 
    a1 = 0.5*rho3*rho3 + rho3*rho3*rho3/6. - rho3*b - 2.75*rho3*rho3*b + 5.*rho3*b*b
    a2 = 0.5*rho3*rho3*rho3 -2.25*rho3*rho3*b + 3.*rho3*b*b
    tanz = np.tan(np.deg2rad(za))
    pr = a0*tanz + a1*tanz*tanz*tanz + a2*tanz*tanz*tanz*tanz*tanz

    return np.rad2deg(pr)


def radec_to_subaru(ra, dec, pa, time, epoch, pmra, pmdec, par, inr=None, log=True):

    # Set Observation Site (Subaru)
    tel = EarthLocation.of_site('Subaru')
    tel2 = Observer.at_site("Subaru", timezone="US/Hawaii")
    # Observation time
    obs_time = Time(time)
    # The equinox of the catalogue
    obs_jtime = Time(epoch, format='decimalyear')
    logging.info(obs_jtime)
    d_yr = TimeDelta(obs_time.jd-obs_jtime.jd, format='jd').to(u.yr)
    logging.debug(d_yr)

    # Atmospheric refraction
    # aref_file = mypath+'data/Refraction_data_635nm.txt'
    # atm_ref = np.loadtxt(aref_file)
    # atm_interp = ipol.splrep(atm_ref[:, 0], atm_ref[:, 1], s=0)

    pmra = pmra*u.mas/u.yr
    pmdec = pmdec*u.mas/u.yr
    par = par*u.mas

    logging.debug(par)
    coord1 = SkyCoord(ra*u.deg, dec*u.deg, distance=Distance(parallax=par),
                      pm_ra_cosdec=pmra, pm_dec=pmdec,
                      frame='fk5', obstime=obs_jtime, equinox='J2000.000')
    coord2 = coord1.apply_space_motion(dt=d_yr.to(u.yr))
    coord3 = coord2.transform_to(FK5(equinox=obs_time.jyear_str))
    # coord4 = coord3.transform_to(TETE(obstime=obs_time, location=tel))
    if log:
        logging.info("Ra Dec = (%s %s) : original",
                     coord1.ra.deg, coord1.dec.deg)
        logging.info("PM = (%s %s)",
                     coord1.pm_ra_cosdec, coord1.pm_dec)
        logging.info("Ra Dec = (%s %s) : applied proper motion",
                     coord2.ra.deg, coord2.dec.deg)
        logging.info("Ra Dec = (%s %s) : applied presession",
                     coord3.ra.deg, coord3.dec.deg)
        # logging.info("Ra Dec = (%s %s) : applied earth motion",
        #              coord4.ra.deg, coord4.dec.deg)

    altaz = coord3.transform_to(AltAz(obstime=obs_time,
                                location=Subaru_POPT2_PFS.Lsbr))

    az = altaz.az.deg
    el = altaz.alt.deg
    za = 90. - el
    # el = el + calc_atmospheric_refraction(za, wl=wl_ag)
    # eld0 = el0 + ipol.splev(za, atm_interp)/3600.
    try:
        za = za[0]
    except (IndexError, TypeError) as e:
        pass

    # Instrument rotator angle
    # Use Subaru_POPT2_PFS to have commonality (disabled: 2022.09.20)

    if inr is None:
        paa = tel2.parallactic_angle(obs_time, coord3).deg
        # lat = tel2.location.lat.deg
        dc = coord3.dec.deg
        logging.debug(dc)
        inr = paa + pa

    # check inr range is within +/- 180 degree
    if inr <= -180.:
        logging.info("InR will exceed the lower limit (-180 deg)")
        inr = inr + 360.
    elif inr >= +180:
        logging.info("InR will exceed the upper limit (+180 deg)")
        inr = inr - 360.
    # Disable for the time being.
    # subaru = Subaru_POPT2_PFS.Subaru()
    # inr = subaru.radec2inr(coord3.ra, coord3.dec, obs_time)

    return az, el, inr
