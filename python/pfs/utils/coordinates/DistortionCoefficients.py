#!/usr/bin/env python

import os,sys,re
import math as mt
import numpy as np
import scipy
from scipy import interpolate as ipol

"""
Distortion Coefficients
 ver 1.0 : based on spot data in May 2017
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
                "mcs_pfi":0.000503107811479}
        # y : dy = c0*x^2 + c1*y^2 + c2*y^4 + c3*x^2*y^2 + c4
        dic_cy={"sky_pfi":[0.045278345420470184, 0.094587722958699313, 0.063632568952111501, 0.074033633816049674, -0.023152187620119963],
                "sky_pfi_hsc":[0.03094074673276881, 0.07592444641651051, 0.04699568361118014, 0.05799533522450577, -0.01739380213343191],
                "mcs_pfi":[0.00030369690700435151, 0.00063595572244913589, 2.8397096204379468e-06, 3.3134621802542193e-06, -0.022829824902876293]}
        # y : slope cy5(z) * y
        dic_slp={"sky_pfi":[0.,-0.00069803799551166809, -0.0048414570302028372],
                 "sky_pfi_hsc":[0., -0.00015428320350572582, -7.358026105602118e-05],
                 "mcs_pfi":[0.,6.4670295752240791e-07,0.00028349223770956881]}

        # Scaling factor of difference
        dic_dsc={"sky_pfi":[0.995465, 1.741161],
                 "sky_pfi_hsc":[0.116418996679199, 16.9113776962136],
                 "mcs_pfi":[0.995339, 1.741417]}

        # Scaling factor w.r.t radius
        dic_rsc={"sky_pfi":[319.97094550870497, 14.307946849934524, 6.6009649162879214],
                 "sky_pfi_hsc":[320.0107640966762, 14.297123404641752, 6.586014956556028],
                 "mcs_pfi":[26.209255623259196, 0.0077741519133454062, 2.4652054436469228e-05]}

        self.mode=mode
        self.cx=dic_cx[mode]
        self.cy=dic_cy[mode]
        self.slp=dic_slp[mode]
        self.dsc=dic_dsc[mode]
        self.rsc=dic_rsc[mode]
