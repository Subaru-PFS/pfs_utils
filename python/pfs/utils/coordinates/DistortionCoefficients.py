#!/usr/bin/env python

import os,sys,re
import math as mt
import numpy as np
import scipy
from scipy import interpolate as ipol

"""
Distortion Coefficients
 ver 1.0 : based on spot data in May 2017
 ver 2.0 : based on spot data in September 2018
           only coefficients from MCS to F3C (mcs_pfi) were changed
 ver 3.0 :  (2019.06.20) 
           add F3C to MCS (pfi_mcs and pfi_mcs_wofe) coefficients using data in September 2018    
 ver 4.0 :  (2019.02.20) 
           add temporary MCS to F3C at ASRD (mcs_pfi_asrd) coefficients using measurement at ASRD    
 ver 5.0 :  (2019.02.23) 
           add MCS to F3C withot Field Element (mcs_pfi_wofe) coefficients using data in January 2020    
"""

## Dictionary keys ( argument name is mode)
# sky_pfi : sky to F3C
# sky_pfi_hsc : sky to hsc focal plane
# sky_mcs : sky to MCS
# pfi_mcs : F3C to MCS
# mcs_pfi : MCS to F3C

class Coeff:

    def __init__(self, mode):

        # Differential pattern
        # x : dx = c0*x*y
        dic_cx={"sky_pfi":0.0750466788561,
                "sky_pfi_hsc":0.06333788071803437,
                "pfi_mcs":np.nan,
                "pfi_mcs_wofe":np.nan,
                "mcs_pfi":[-0.00021573669065476386, -7.173175426293252e-05, 0.0003774856829608848, 0.0058948766457507734],
                "mcs_pfi_wofe":[-0.00021578492309190278 , -7.065020050370169e-05 , 0.00037240779097299495 , 0.005816579868913723],
                "mcs_pfi_asrd":[]}
        # y : dy = c0*x^2 + c1*y^2 + c2*y^4 + c3*x^2*y^2 + c4
        dic_cy={"sky_pfi":[0.045278345420470184, 0.094587722958699313, 0.063632568952111501, 0.074033633816049674, -0.023152187620119963],
                "sky_pfi_hsc":[0.03094074673276881, 0.07592444641651051, 0.04699568361118014, 0.05799533522450577, -0.01739380213343191],
                "pfi_mcs":[],
                "pfi_mcs_wofe":[],
                "mcs_pfi":[0.0003810242955129456, 0.0005690616536446459, 3.0281647084446334e-06, 3.0439743645414486e-06, -0.02291378662107904],
                "mcs_pfi_wofe":[0.0003789711867797638  , 0.0005782721715957847 , 2.790396278217492e-06 , 3.3437676720357216e-06 , -0.02266398882916199],
                "mcs_pfi_asrd":[]}
        # Whether to skip use offset base
        skip1_off={"sky_pfi":False,
                 "sky_pfi_hsc":False,
                 "pfi_mcs":True,
                 "pfi_mcs_wofe":True,
                 "mcs_pfi":False,
                 "mcs_pfi_wofe":False,
                 "mcs_pfi_asrd":False}
        # Whether to skip use differential pattern
        skip2_off={"sky_pfi":False,
                 "sky_pfi_hsc":False,
                 "pfi_mcs":True,
                 "pfi_mcs_wofe":True,
                 "mcs_pfi":False,
                 "mcs_pfi_wofe":False,
                 "mcs_pfi_asrd":True}

        # y : slope cy5(z) * y
        dic_slp={"sky_pfi":[0.,-0.00069803799551166809, -0.0048414570302028372],
                 "sky_pfi_hsc":[0., -0.00015428320350572582, -7.358026105602118e-05],
                 "pfi_mcs":[],
                "pfi_mcs_wofe":[],
                 "mcs_pfi":[0.,4.83653998826754e-05,0.0004251103128037524],
                 "mcs_pfi_wofe":[0.,-3.317782365886957e-05,2.691010630701391e-06],
                 "mcs_pfi_asrd":[]}

        # Scaling factor of difference
        dic_dsc={"sky_pfi":[0.995465, 1.741161],
                 "sky_pfi_hsc":[0.116418996679199, 16.9113776962136],
                 "pfi_mcs":[],
                "pfi_mcs_wofe":[],
                 "mcs_pfi":[-0.304865874392984, -4.36491928024528, 2.05697592456737, -0.0159919544966072],
                 "mcs_pfi_wofe":[-0.4455515583151641, -3.6721472387615224, -129.54246106865378, -0.01263007442575775],
                 "mcs_pfi_asrd":[]}

        # Scaling factor w.r.t radius
        # mcs_pfi_asrd : scale factor + distortion (3rd order from distotion center)
        dic_rsc={"sky_pfi":[319.97094550870497, 14.307946849934524, 6.6009649162879214, 0.],
                 "sky_pfi_hsc":[320.0107640966762, 14.297123404641752, 6.586014956556028, 0.],
                 "pfi_mcs":[0.03802405803382136, 1.7600554217409373e-08, -1.5297627440776523e-14, 1.6143531343033776e-19],
                 "pfi_mcs_wofe":[0.038019509493361525, -1.7569114332967838e-08, -1.5633186312477447e-14, -1.5136575959136838e-19],
                 "mcs_pfi":[26.298537331963416, 0.008504875348648966, 9.749117230839488e-06, 1.0827269091018277e-07],
                 "mcs_pfi_wofe":[26.301766037195193, 0.008484216901138097, 1.0150811666775894e-05, 1.0290642807331274e-07],
                 "mcs_pfi_asrd":[24.68498072,-0.0287145 , 0.001562543 , -1.58048E-05 , -4.11313E-08]}

        self.mode=mode
        self.cx=dic_cx[mode]
        self.cy=dic_cy[mode]
        self.skip1_off=skip1_off[mode]
        self.skip2_off=skip2_off[mode]
        self.slp=dic_slp[mode]
        self.dsc=dic_dsc[mode]
        self.rsc=dic_rsc[mode]


# MCS pixel scale: 3.2um/pixel
mcspixel=3.2e-3
mcspixel_asrd=3.1e-3

# Distortion Center of Temporary MCS on PFI coordinates
#distc_x_asrd=-15.5764 -(-1.568) ; distc_y_asrd = -2.62071 -(1.251)
distc_x_asrd=-15.5764 ; distc_y_asrd = -2.62071