# -*- coding: utf-8 -*-
import numpy as np

from astropy.io import fits as pyfits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Distance, EarthLocation, AltAz, get_sun
from astropy.coordinates import ICRS, FK5
from astropy.utils import iers
iers.conf.auto_download = False  

d_ra  = 1.0/3600.0
d_de  = 1.0/3600.0
d_inr = 0.01

### unknown scale factor
# Unknown_Scale_Factor = 1.0 + 2.22e-4
Unknown_Scale_Factor = 1.0

### constants proper to WFC optics
# wfc_scale_temp_coeff = -1.42e-05
wfc_scale_M2POS3_coeff = 1.01546e-4

### constants proper to PFS camera
pfs_inr_zero_offset        =  0.00 # in degree
pfs_detector_zero_offset_x =  0.00 # in mm
pfs_detector_zero_offset_y =  0.00 # in mm

class Misc():
    def diff_index_pbl1yold_bsl7y(self, wl):
        dn=0.0269+0.000983/(wl-0.118)**2 # used only unit conversion... precision is not very high.
        return dn

    def band_parameter(self, wl):
        # index of silica
        K1=6.961663e-01
        L1=4.679148e-03
        K2=4.079426e-01
        L2=1.351206e-02
        K3=8.974794e-01
        L3=9.793400e+01
        ns = np.sqrt(1+K1*wl**2/(wl**2-L1)+K2*wl**2/(wl**2-L2)+K3*wl**2/(wl**2-L3))
        return (ns-1.458)*100.0
    
class Subaru():
    def starSepZPA(self, tel_ra, tel_de, str_ra, str_de, wl, t):
        tel_ra = tel_ra*u.degree
        tel_de = tel_de*u.degree
        str_ra = str_ra*u.degree
        str_de = str_de*u.degree

        tel_coord = SkyCoord(ra=tel_ra, dec=tel_de, frame='fk5')
        str_coord = SkyCoord(ra=str_ra, dec=str_de, frame='fk5')

        sbr = EarthLocation.of_site('Subaru Telescope')
        frame_subaru = AltAz(obstime  = t, location = sbr,\
                             pressure = 600.5*u.hPa, obswl = wl*u.micron)
                             # pressure = 620*u.hPa, obswl = wl*u.micron)

        tel_altaz = tel_coord.transform_to(frame_subaru)
        str_altaz = str_coord.transform_to(frame_subaru)

        str_sep =  tel_altaz.separation(str_altaz).degree
        str_zpa = -tel_altaz.position_angle(str_altaz).degree

        return str_sep, str_zpa

    def starRADEC(self, tel_ra, tel_de, str_sep, str_zpa, wl, t):
        tel_ra  = tel_ra*u.degree
        tel_de  = tel_de*u.degree
        str_sep = str_sep*u.degree
        str_zpa = str_zpa*u.degree

        tel_coord = SkyCoord(ra=tel_ra, dec=tel_de, frame='fk5')

        sbr = EarthLocation.of_site('Subaru Telescope')
        frame_subaru = AltAz(obstime  = t, location = sbr,\
                             pressure = 600.5*u.hPa, obswl = wl*u.micron)
                             # pressure = 620*u.hPa, obswl = wl*u.micron)

        tel_altaz = tel_coord.transform_to(frame_subaru)

        str_altaz = tel_altaz.directional_offset_by(str_zpa, str_sep)
        
        str_coord = str_altaz.transform_to('fk5')
        ra = str_coord.ra.degree
        de = str_coord.dec.degree

        #print(ra,de)
        return ra,de
        
    def radec2radecplxpm(self, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t):
        str_plx[np.where(str_plx<0.00001)]=0.00001
        
        str_coord = SkyCoord(ra=str_ra*u.degree, dec=str_de*u.degree,
                             distance=Distance(parallax=str_plx * u.mas, allow_negative=False),             
                             pm_ra_cosdec=str_pmRA * u.mas/u.yr,
                             pm_dec= str_pmDE * u.mas/u.yr,
                             obstime=Time(2015.5, format='decimalyear'),
                             frame='icrs')

        str_coord_obstime = str_coord.apply_space_motion(Time(t))
        
        ra = str_coord_obstime.ra.degree
        de = str_coord_obstime.dec.degree

        return ra,de

class POPT2():
    def telStatusfromText(self, textfile):
        array = np.loadtxt(textfile,\
                           dtype={\
                               'names': ('DATE-OBS', 'UT-STR', 'RA2000', 'DEC2000', 'ADC-STR', 'INR-STR', 'M2-POS3', 'WAVELEN'),\
                               'formats':('<U10', '<U12', '<U12', '<U12', '<U12', '<U12', '<U12', '<U12')})
        dateobs = str(array['DATE-OBS'])
        utstr   = str(array['UT-STR'])
        ra2000  = str(array['RA2000'])
        dec2000 = str(array['DEC2000'])
        adcstr  = float(array['ADC-STR'])
        inrstr  = float(array['INR-STR'])
        m2pos3  = float(array['M2-POS3'])
        wl      = float(array['WAVELEN'])

        datetime= dateobs+"T"+utstr+"Z"
        coord = SkyCoord(ra=ra2000, dec=dec2000, unit=(u.hourangle, u.deg),
                         frame='fk5')
        tel_ra = coord.ra.degree
        tel_de = coord.dec.degree

        return tel_ra, tel_de, datetime, adcstr, inrstr, m2pos3, wl

    def telStatus(self, dateobs, utstr, ra2000, dec2000, adcstr, inrstr, m2pos3, wl):
        datetime= dateobs+"T"+utstr+"Z"
        coord = SkyCoord(ra=ra2000, dec=dec2000, unit=(u.hourangle, u.deg),
                         frame='fk5')
        tel_ra = coord.ra.degree
        tel_de = coord.dec.degree

        return tel_ra, tel_de, datetime, adcstr, inrstr, m2pos3, wl
        
    def celestial2focalplane(self, sep, zpa, adc, m2pos3, wl, flag):
        f = flag.astype(int)
        
        xfp_wisp, yfp_wisp = POPT2.celestial2focalplane_wisp(self, sep, zpa, adc, m2pos3, wl)
        xfp_wosp, yfp_wosp = POPT2.celestial2focalplane_wosp(self, sep, zpa, adc, m2pos3, wl)
        
        xfp = xfp_wosp*f + xfp_wisp*(1-f)
        yfp = yfp_wosp*f + yfp_wisp*(1-f)

        return xfp,yfp

    def focalplane2celestial(self, xt, yt, adc, m2pos3, wl, flag):
        f = flag.astype(int)
        
        r_wisp, t_wisp = POPT2.focalplane2celestial_wisp(self, xt, yt, adc, m2pos3, wl)
        r_wosp, t_wosp = POPT2.focalplane2celestial_wosp(self, xt, yt, adc, m2pos3, wl)

        r = r_wosp*f + r_wisp*(1-f)
        t = t_wosp*f + t_wisp*(1-f)

        return r,t

    def additionaldistortion(self, xt, yt):
        x = xt / 270.0
        y = yt / 270.0

        # data from HSC-PFS test observation on 2020/10/23, 7 sets of InR rotating data.
        adx =  5.96462898e-05 +2.42107822e-03*(2*(x**2+y**2)-1) +3.81085098e-03*(x**2-y**2) -2.75632544e-03*(2*x*y) -1.35748905e-03*((3*(x**2+y**2)-2)*x) +2.12301241e-05*((3*(x**2+y**2)-2)*y) -9.23710861e-05*(6*(x**2+y**2)**2-6*(x**2+y**2)+1)
        ady = -2.06476363e-04 -3.45168908e-03*(2*(x**2+y**2)-1) +4.09091198e-03*(x**2-y**2) +3.41002899e-03*(2*x*y) +1.63045902e-04*((3*(x**2+y**2)-2)*x) -1.01811037e-03*((3*(x**2+y**2)-2)*y) +2.07395905e-04*(6*(x**2+y**2)**2-6*(x**2+y**2)+1)

        return adx,ady

    def celestial2focalplane_wisp(self, sep, zpa, adc, m2pos3, wl):
        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)
        
        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans =  np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        x = tanx/0.014
        y = tany/0.014

        cx2 =  2.62676925e+02 +(-1.87221607e-06 +2.92222089e-06*D)*adc**2
        cx7 =  3.35307347e+00 +(-7.81712660e-09 -1.05819104e-09*D)*adc**2
        cx14=  2.38166168e-01 +(-2.03764107e-08 -7.38480616e-07*D)*adc**2
        cx23=  2.36440726e-02 +( 5.96839761e-09 -1.07538100e-06*D)*adc**2
        cx34= -2.62961444e-03 +( 8.86909851e-09 -9.14461129e-07*D)*adc**2
        cx47=  2.20267977e-03 +(-4.30901437e-08 +1.07839312e-06*D)*adc**2
        cx6 =                  ( 5.03828902e-04 -3.02018655e-02*D)*adc
        cx13=                  ( 1.62039347e-04 -5.25199388e-03*D)*adc
        cx1 =  0.00000000e+00
        cx4 =  0.00000000e+00
        cx9 =  0.00000000e+00

        cy3 =  2.62676925e+02 +(-8.86455750e-06 +3.15626958e-06*D)*adc**2
        cy8 =  3.35307347e+00 +(-6.26053110e-07 -2.95393149e-07*D)*adc**2
        cy15=  2.38166168e-01 +(-1.50434841e-07 -1.29919261e-06*D)*adc**2
        cy24=  2.36440726e-02 +(-1.29277787e-07 -1.83675834e-06*D)*adc**2
        cy35= -2.62961444e-03 +(-5.72922063e-08 -1.27892174e-06*D)*adc**2
        cy48=  2.20267977e-03 +(-1.11704310e-08 +1.53690637e-06*D)*adc**2
        cy5 =                 +(-4.99201211e-04 +3.00941960e-02*D)*adc
        cy12=                 +(-1.14622807e-04 +4.07269286e-03*D)*adc
        cy1 =                 +( 2.26420481e-02 -7.33125190e-01*D)*adc
        cy4 =                 +( 9.65017704e-04 -3.01504932e-02*D)*adc
        cy9 =                 +( 1.76747572e-04 -4.84455318e-03*D)*adc

        telx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tely = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        telx = telx * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor
        tely = tely * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor

        adtelx,adtely = POPT2.additionaldistortion(self, telx,tely)
        telx = telx + adtelx
        tely = tely + adtely
        
        return telx,tely

    def focalplane2celestial_wisp(self, xt, yt, adc, m2pos3, wl):
        adtelx,adtely = POPT2.additionaldistortion(self, xt, yt)
        xt = xt - adtelx
        yt = yt - adtely

        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor
        y = y / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor

        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)

        cx2 =  1.43733003e-02 +( 8.72090395e-11 +0.00000000e+00*D)*adc**2
        cx7 = -1.88237022e-04 +(-7.82457037e-12 +1.21758496e-11*D)*adc**2
        cx14= -6.78700730e-06 +( 1.66205167e-12 +1.81935028e-11*D)*adc**2
        cx23= -3.85793541e-07 +( 8.07898607e-13 +2.19246566e-11*D)*adc**2
        cx34=  2.67691166e-07 +( 1.73923452e-13 +1.67246499e-11*D)*adc**2
        cx47= -1.20869185e-07 +( 9.09157007e-13 -2.89301337e-11*D)*adc**2
        cx6 =                 +( 2.29093928e-08 +8.45286748e-09*D)*adc
        cx13=                 +(-2.12409671e-09 +3.88359241e-08*D)*adc
        cx1 =  0.00000000e+00
        cx4 =  0.00000000e+00
        cx9 =  0.00000000e+00
        
        cy3 =  1.43733003e-02 +( 4.61943503e-10 +0.00000000e+00*D)*adc**2
        cy8 = -1.88237022e-04 +( 1.26225040e-11 +8.53001285e-12*D)*adc**2
        cy15= -6.78700730e-06 +( 8.50721539e-12 +2.59588237e-11*D)*adc**2
        cy24= -3.85793541e-07 +( 7.98095083e-12 +6.92047432e-11*D)*adc**2
        cy35=  2.67691166e-07 +( 1.36129621e-12 +8.45438550e-11*D)*adc**2
        cy48= -1.20869185e-07 +( 4.48176515e-13 -7.66773681e-11*D)*adc**2
        cy5 =                 +(-2.31614663e-08 -3.71361360e-09*D)*adc
        cy12=                 +( 3.43468003e-10 +2.70608246e-10*D)*adc
        cy1 =                 +(-1.19076885e-06 +3.85419528e-05*D)*adc
        cy4 =                 +(-2.61964016e-09 +5.85399688e-09*D)*adc
        cy9 =                 +(-2.25081043e-09 +2.51123479e-08*D)*adc

        tantx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tanty = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))
        
        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)
        
        s = np.rad2deg(s)
        t = np.rad2deg(t)

        return s, t

    def celestial2focalplane_cobra(self, sep, zpa, adc, m2pos3, wl):
        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)
        L = Misc.band_parameter(self, wl)
        
        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans =  np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        x = tanx/0.014
        y = tany/0.014

        cx2  = (+2.62789699e+02)+(+1.41436855e-02*L)+(+6.58982982e-03*L**2)+(-9.48051092e-03*L**3) + (-1.00694635e-06 +(-2.48248836e-05*D))*adc**2
        cx7  = (+3.36215747e+00)+(+3.69187393e-03*L)+(-6.74817524e-03*L**2)+(+1.54801942e-03*L**3) + (+7.61247574e-09 +(-6.33418826e-07*D))*adc**2
        cx14 = (+2.35198776e-01)+(+9.42825888e-04*L)+(+9.90958196e-04*L**2)+(-3.17796838e-05*L**3) + (+1.42425375e-08 +(-1.72148497e-06*D))*adc**2
        cx23 = (+1.46655737e-02)+(-1.79003541e-03*L)+(-1.55308635e-03*L**2)+(+1.13125337e-03*L**3) + (+2.15230267e-08 +(-1.40990705e-06*D))*adc**2
        cx34 = (-5.52214359e-03)+(-6.97491826e-04*L)+(-9.14390111e-04*L**2)+(+5.80494577e-04*L**3) + (-7.18716293e-09 +(-1.08374155e-06*D))*adc**2
        cx47 = (+2.71674871e-03)+(+1.09560963e-03*L)+(+4.82675939e-04*L**2)+(-1.53464406e-04*L**3) + (-1.67044496e-08 +(+5.65443437e-07*D))*adc**2
        cx6  =                                                                                     + (+1.14778526e-04 +(-1.75018873e-02*D))*adc
        cx13 =                                                                                     + (+8.13032252e-06 +(-1.58749659e-04*D))*adc
        cx1  =   0.00000000e+00
        cx4  =   0.00000000e+00
        cx9  =   0.00000000e+00

        cy3  = (+2.62789699e+02)+(+1.41436855e-02*L)+(+6.58982982e-03*L**2)+(-9.48051092e-03*L**3) + (-4.18883241e-06 +(-1.47780317e-04*D))*adc**2
        cy8  = (+3.36215747e+00)+(+3.69187393e-03*L)+(-6.74817524e-03*L**2)+(+1.54801942e-03*L**3) + (-3.63808492e-07 +(-9.59230438e-06*D))*adc**2
        cy15 = (+2.35198776e-01)+(+9.42825888e-04*L)+(+9.90958196e-04*L**2)+(-3.17796838e-05*L**3) + (-2.93764283e-08 +(-4.75333128e-06*D))*adc**2
        cy24 = (+1.46655737e-02)+(-1.79003541e-03*L)+(-1.55308635e-03*L**2)+(+1.13125337e-03*L**3) + (+8.30835081e-08 +(-8.68707778e-06*D))*adc**2
        cy35 = (-5.52214359e-03)+(-6.97491826e-04*L)+(-9.14390111e-04*L**2)+(+5.80494577e-04*L**3) + (+2.58088187e-08 +(-4.49063408e-06*D))*adc**2
        cy48 = (+2.71674871e-03)+(+1.09560963e-03*L)+(+4.82675939e-04*L**2)+(-1.53464406e-04*L**3) + (-3.73048239e-08 +(+2.68719899e-06*D))*adc**2
        cy5  =                                                                                     + (-1.14688856e-04 +(+1.75492667e-02*D))*adc
        cy12 =                                                                                     + (-3.46582624e-06 +(+3.65925894e-04*D))*adc
        cy1  =                                                                                     + (+2.17560716e-04 +(-6.31969179e-03*D))*adc
        cy4  =                                                                                     + (+2.29182718e-04 +(-6.21806715e-03*D))*adc
        cy9  =                                                                                     + (+3.44748694e-05 +(-1.52943095e-04*D))*adc

        telx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tely = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        telx = telx * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor
        tely = tely * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor

        adtelx,adtely = POPT2.additionaldistortion(self, telx,tely)
        telx = telx + adtelx
        tely = tely + adtely
        
        return telx,tely
    
    def celestial2focalplane_wosp(self, sep, zpa, adc, m2pos3, wl):
        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)
        
        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans =  np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        x = tanx/0.014
        y = tany/0.014

        cx2 =  2.62647800e+02 +(-1.87011330e-06 +2.77589523e-06*D)*adc**2
        cx7 =  3.35086900e+00 +(-7.57865771e-09 +5.53973051e-08*D)*adc**2
        cx14=  2.38964699e-01 +(-1.94709286e-08 -7.73207447e-07*D)*adc**2
        cx23=  2.60230125e-02 +( 6.60545302e-09 -1.16130987e-06*D)*adc**2
        cx34= -1.83104099e-03 +( 1.57806929e-08 -1.02698829e-06*D)*adc**2
        cx47=  2.04091566e-03 +(-4.93364523e-08 +1.22931907e-06*D)*adc**2
        cx6 =                  ( 5.04570462e-04 -3.02466062e-02*D)*adc
        cx13=                  ( 1.68473464e-04 -5.48576870e-03*D)*adc
        cx1 =  0.00000000e+00
        cx4 =  0.00000000e+00
        cx9 =  0.00000000e+00

        cy3 =  2.62647800e+02 +(-8.86916339e-06 +3.23008109e-06*D)*adc**2
        cy8 =  3.35086900e+00 +(-6.08839423e-07 -5.65253675e-07*D)*adc**2
        cy15=  2.38964699e-01 +(-1.41970867e-07 -1.67270348e-06*D)*adc**2
        cy24=  2.60230125e-02 +(-1.41280083e-07 -1.52144095e-06*D)*adc**2
        cy35= -1.83104099e-03 +(-5.94064581e-08 -1.16255139e-06*D)*adc**2
        cy48=  2.04091566e-03 +(-9.97915337e-09 +1.46853795e-06*D)*adc**2
        cy5 =                  (-4.99779792e-04 +3.01309493e-02*D)*adc
        cy12=                  (-1.17250417e-04 +4.18415867e-03*D)*adc
        cy1 =                  ( 2.26486871e-02 -7.33054030e-01*D)*adc
        cy4 =                  ( 9.65729204e-04 -3.01924555e-02*D)*adc
        cy9 =                  ( 1.81933708e-04 -5.03118775e-03*D)*adc

        telx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tely = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        telx = telx * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor
        tely = tely * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor

        adtelx,adtely = POPT2.additionaldistortion(self, telx,tely)
        telx = telx + adtelx
        tely = tely + adtely
        
        return telx,tely

    def focalplane2celestial_wosp(self, xt, yt, adc, m2pos3, wl):
        adtelx,adtely = POPT2.additionaldistortion(self, xt, yt)
        xt = xt - adtelx
        yt = yt - adtely

        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor
        y = y / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor

        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)

        cx2 =  1.43748020e-02 +( 8.66331254e-11 +1.93388878e-11*D)*adc**2
        cx7 = -1.88232442e-04 +(-7.82853227e-12 +8.59386158e-12*D)*adc**2
        cx14= -6.87397752e-06 +( 1.61715989e-12 +2.00354352e-11*D)*adc**2
        cx23= -5.21300344e-07 +( 7.29677304e-13 +2.69615671e-11*D)*adc**2
        cx34=  2.39570945e-07 +(-2.82067547e-13 +2.51776050e-11*D)*adc**2
        cx47= -1.08241906e-07 +( 1.28605646e-12 -3.83590010e-11*D)*adc**2
        cx6 =                  ( 2.29520266e-08 +9.04783592e-09*D)*adc
        cx13=                  (-2.24317632e-09 +4.37694338e-08*D)*adc
        cx1 =  0.00000000e+00
        cx4 =  0.00000000e+00
        cx9 =  0.00000000e+00
        
        cy3 =  1.43748020e-02 +( 4.61979705e-10 +3.65991127e-12*D)*adc**2
        cy8 = -1.88232442e-04 +( 1.16038555e-11 +2.66431645e-11*D)*adc**2
        cy15= -6.87397752e-06 +( 8.12842568e-12 +4.41948555e-11*D)*adc**2
        cy24= -5.21300344e-07 +( 8.66566778e-12 +4.92555793e-11*D)*adc**2
        cy35=  2.39570945e-07 +( 1.33811721e-12 +8.16466379e-11*D)*adc**2
        cy48= -1.08241906e-07 +( 4.16136826e-13 -7.37253146e-11*D)*adc**2
        cy5 =                  (-2.32140319e-08 -3.87365915e-09*D)*adc
        cy12=                  ( 3.16034703e-10 -6.95499653e-11*D)*adc
        cy1 =                  (-1.19123300e-06 +3.85419594e-05*D)*adc
        cy4 =                  (-2.58356822e-09 +6.26527802e-09*D)*adc
        cy9 =                  (-2.32047805e-09 +2.81735279e-08*D)*adc

        tantx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tanty = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))
        
        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)
        
        s = np.rad2deg(s)
        t = np.rad2deg(t)

        return s, t

    def focalplane2celestial_cobra(self, xt, yt, adc, m2pos3, wl):
        adtelx,adtely = POPT2.additionaldistortion(self, xt, yt)
        xt = xt - adtelx
        yt = yt - adtely

        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor
        y = y / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor

        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)
        L = Misc.band_parameter(self, wl)

        cx2  = ( 1.43674845e-02)+(-7.43646874e-07*L)+(-3.16130067e-07*L**2)+( 4.80328267e-07*L**3) + ( 5.05131730e-11 +( 1.17283146e-09*D))*adc**2
        cx7  = (-1.88289474e-04)+(-1.56460268e-07*L)+( 3.80620067e-07*L**2)+(-1.19205317e-07*L**3) + (-4.84876314e-12 +(-7.61093281e-11*D))*adc**2
        cx14 = (-6.46172751e-06)+(-2.66777511e-08*L)+(-6.33815835e-08*L**2)+(-7.74745171e-09*L**3) + (-9.62743893e-13 +( 9.64732206e-11*D))*adc**2
        cx23 = ( 1.23562185e-07)+( 9.24192386e-08*L)+( 9.39175516e-08*L**2)+(-6.74785592e-08*L**3) + (-6.67748446e-13 +( 6.60925128e-11*D))*adc**2
        cx34 = ( 3.68389899e-07)+( 1.03386621e-08*L)+( 3.47784649e-08*L**2)+(-2.36730251e-08*L**3) + ( 1.08679636e-12 +( 2.43537188e-11*D))*adc**2
        cx47 = (-1.63142810e-07)+(-4.53025978e-08*L)+(-2.49134365e-08*L**2)+( 1.03064829e-08*L**3) + ( 4.18052943e-13 +(-3.13541684e-11*D))*adc**2
        cx6  =                                                                                     + (-5.39147702e-09 +( 9.19945103e-07*D))*adc
        cx13 =                                                                                     + ( 2.77202779e-10 +(-4.32936212e-08*D))*adc
        cx1  =   0.00000000e+0
        cx4  =   0.00000000e+0
        cx9  =   0.00000000e+0

        cy3  = ( 1.43674845e-02)+(-7.43646874e-07*L)+(-3.16130067e-07*L**2)+( 4.80328267e-07*L**3) + ( 2.20715956e-10 +( 7.77431541e-09*D))*adc**2
        cy8  = (-1.88289474e-04)+(-1.56460268e-07*L)+( 3.80620067e-07*L**2)+(-1.19205317e-07*L**3) + ( 8.06546205e-12 +( 1.99216576e-10*D))*adc**2
        cy15 = (-6.46172751e-06)+(-2.66777511e-08*L)+(-6.33815835e-08*L**2)+(-7.74745171e-09*L**3) + (-6.15086757e-13 +( 2.97964659e-10*D))*adc**2
        cy24 = ( 1.23562185e-07)+( 9.24192386e-08*L)+( 9.39175516e-08*L**2)+(-6.74785592e-08*L**3) + (-4.06656270e-12 +( 4.65768165e-10*D))*adc**2
        cy35 = ( 3.68389899e-07)+( 1.03386621e-08*L)+( 3.47784649e-08*L**2)+(-2.36730251e-08*L**3) + ( 5.85562680e-13 +( 1.36403403e-10*D))*adc**2
        cy48 = (-1.63142810e-07)+(-4.53025978e-08*L)+(-2.49134365e-08*L**2)+( 1.03064829e-08*L**3) + ( 1.38694343e-12 +(-1.22884800e-10*D))*adc**2
        cy5  =                                                                                     + ( 5.38704437e-09 +(-9.23542499e-07*D))*adc
        cy12 =                                                                                     + (-4.39088156e-10 +( 3.02959958e-08*D))*adc
        cy1  =                                                                                     + (-1.18341815e-08 +( 3.31202350e-07*D))*adc
        cy4  =                                                                                     + (-1.18199424e-08 +( 2.99799379e-07*D))*adc
        cy9  =                                                                                     + (-9.77458744e-10 +(-1.95302432e-08*D))*adc
        
        tantx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tanty = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))
        
        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)
        
        s = np.rad2deg(s)
        t = np.rad2deg(t)

        return s, t

    def sourceFilter(self, agarray, maxellip, maxsize, minsize):
        ag_ccd  = agarray[:,0]
        ag_id   = agarray[:,1]
        ag_xc   = agarray[:,2]
        ag_yc   = agarray[:,3]
        ag_flx  = agarray[:,4]
        ag_smma = agarray[:,5]
        ag_smmi = agarray[:,6]
        ag_flag = agarray[:,7]
        
        # ellipticity condition
        cellip = (1.0-ag_smmi/ag_smma)    < maxellip
        # size condition (upper)
        csizeU = np.sqrt(ag_smmi*ag_smma) < maxsize
        # size condition (lower)
        csizeL = np.sqrt(ag_smmi*ag_smma) > minsize

        v = cellip*csizeU*csizeL
            
        vdatanum = np.sum(v)

        oarray = np.zeros((vdatanum,8))
        oarray[:,0] = ag_ccd[v]
        oarray[:,1] = ag_id[v]
        oarray[:,2] = ag_xc[v]
        oarray[:,3] = ag_yc[v]
        oarray[:,4] = ag_flx[v]
        oarray[:,5] = ag_smma[v]
        oarray[:,6] = ag_smmi[v]
        oarray[:,7] = ag_flag[v]
        
        return oarray
            
    # def RADECInRShift(self, obj_xdp, obj_ydp, obj_int, \
    #                   cat_xdp, cat_ydp, cat_mag,\
    #                   dxra,dyra,dxde,dyde,dxinr,dyinr):
    def RADECInRShift(self, obj_xdp, obj_ydp, obj_int, obj_flag, v0, v1):        
        cat_xdp_0 = v0[:,0]
        cat_ydp_0 = v0[:,1]
        cat_mag_0 = v0[:,2]
        dxra_0    = v0[:,3]
        dyra_0    = v0[:,4]
        dxde_0    = v0[:,5]
        dyde_0    = v0[:,6]
        dxinr_0   = v0[:,7]
        dyinr_0   = v0[:,8]
        
        cat_xdp_1 = v1[:,0]
        cat_ydp_1 = v1[:,1]
        cat_mag_1 = v1[:,2]
        dxra_1    = v1[:,3]
        dyra_1    = v1[:,4]
        dxde_1    = v1[:,5]
        dyde_1    = v1[:,6]
        dxinr_1   = v1[:,7]
        dyinr_1   = v1[:,8]

        dxra  = (dxra_0  + dxra_1 )/2.0
        dyra  = (dyra_0  + dyra_1 )/2.0
        dxde  = (dxde_0  + dxde_1 )/2.0
        dyde  = (dyde_0  + dyde_1 )/2.0
        dxinr = (dxinr_0 + dxinr_1)/2.0
        dyinr = (dyinr_0 + dyinr_1)/2.0

        flg = np.where(obj_flag==1.0)
                
        n_obj = (obj_xdp.shape)[0]
        xdiff_0 = np.transpose([obj_xdp])-cat_xdp_0
        ydiff_0 = np.transpose([obj_ydp])-cat_ydp_0
        xdiff_1 = np.transpose([obj_xdp])-cat_xdp_1
        ydiff_1 = np.transpose([obj_ydp])-cat_ydp_1

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[flg]=xdiff_1[flg]
        ydiff[flg]=ydiff_1[flg]

        dist  = np.sqrt(xdiff**2+ydiff**2)
        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj)),min_dist_index
        rCRA = np.median((xdiff[min_dist_indices]*dyde[min_dist_index]-ydiff[min_dist_indices]*dxde[min_dist_index])/(dxra[min_dist_index]*dyde[min_dist_index]-dyra[min_dist_index]*dxde[min_dist_index]))
        rCDE = np.median((xdiff[min_dist_indices]*dyra[min_dist_index]-ydiff[min_dist_indices]*dxra[min_dist_index])/(dxde[min_dist_index]*dyra[min_dist_index]-dyde[min_dist_index]*dxra[min_dist_index]))

        xdiff_0 = np.transpose([obj_xdp])-(cat_xdp_0+rCRA*dxra+rCDE*dxde)
        ydiff_0 = np.transpose([obj_ydp])-(cat_ydp_0+rCRA*dyra+rCDE*dyde)

        xdiff_1 = np.transpose([obj_xdp])-(cat_xdp_1+rCRA*dxra+rCDE*dxde)
        ydiff_1 = np.transpose([obj_ydp])-(cat_ydp_1+rCRA*dyra+rCDE*dyde)

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[flg]=xdiff_1[flg]
        ydiff[flg]=ydiff_1[flg]
        
        dist  = np.sqrt(xdiff**2+ydiff**2)

        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj)),min_dist_index

        f = dist[min_dist_indices]<2

        # print(f)
        
        match_obj_xdp  = obj_xdp[f]
        match_obj_ydp  = obj_ydp[f]
        match_obj_int  = obj_int[f]
        match_obj_flag = obj_flag[f]

        match_cat_xdp_0 = (cat_xdp_0[min_dist_index])[f]
        match_cat_ydp_0 = (cat_ydp_0[min_dist_index])[f]
        match_cat_mag_0 = (cat_mag_0[min_dist_index])[f]
        match_dxra_0    = (dxra_0[min_dist_index])[f]
        match_dyra_0    = (dyra_0[min_dist_index])[f]
        match_dxde_0    = (dxde_0[min_dist_index])[f]
        match_dyde_0    = (dyde_0[min_dist_index])[f]
        match_dxinr_0   = (dxinr_0[min_dist_index])[f]
        match_dyinr_0   = (dyinr_0[min_dist_index])[f]

        match_cat_xdp_1 = (cat_xdp_1[min_dist_index])[f]
        match_cat_ydp_1 = (cat_ydp_1[min_dist_index])[f]
        match_cat_mag_1 = (cat_mag_1[min_dist_index])[f]
        match_dxra_1    = (dxra_1[min_dist_index])[f]
        match_dyra_1    = (dyra_1[min_dist_index])[f]
        match_dxde_1    = (dxde_1[min_dist_index])[f]
        match_dyde_1    = (dyde_1[min_dist_index])[f]
        match_dxinr_1   = (dxinr_1[min_dist_index])[f]
        match_dyinr_1   = (dyinr_1[min_dist_index])[f]

        match_cat_xdp = np.copy(match_cat_xdp_0)
        match_cat_ydp = np.copy(match_cat_ydp_0)
        match_cat_mag = np.copy(match_cat_mag_0)
        match_dxra    = np.copy(match_dxra_0)
        match_dyra    = np.copy(match_dyra_0)
        match_dxde    = np.copy(match_dxde_0)
        match_dyde    = np.copy(match_dyde_0)
        match_dxinr   = np.copy(match_dxinr_0)
        match_dyinr   = np.copy(match_dyinr_0)

        flg = np.where(match_obj_flag==1.0)

        match_cat_xdp[flg] = match_cat_xdp_1[flg]
        match_cat_ydp[flg] = match_cat_ydp_1[flg]
        match_cat_mag[flg] = match_cat_mag_1[flg]
        match_dxra[flg]    = match_dxra_1[flg]
        match_dyra[flg]    = match_dyra_1[flg]
        match_dxde[flg]    = match_dxde_1[flg]
        match_dyde[flg]    = match_dyde_1[flg]
        match_dxinr[flg]   = match_dxinr_1[flg]
        match_dyinr[flg]   = match_dyinr_1[flg]

        dra  = np.concatenate([match_dxra,match_dyra])
        dde  = np.concatenate([match_dxde,match_dyde])
        dinr = np.concatenate([match_dxinr,match_dyinr])

        basis= np.stack([dra,dde,dinr]).transpose()

        errx = match_obj_xdp - match_cat_xdp 
        erry = match_obj_ydp - match_cat_ydp 
        err  = np.concatenate([errx,erry])

        M = np.zeros((3,3))
        b = np.zeros((3))
        for itr1 in range(3):
            for itr2 in range(3):
                M[itr1,itr2] = np.sum(basis[:,itr1]*basis[:,itr2])
            b[itr1] = np.sum(basis[:,itr1]*err)

        A = np.linalg.solve(M,b)
        
        ra_offset  = A[0] * d_ra
        de_offset  = A[1] * d_de
        inr_offset = A[2] * d_inr
        
        return ra_offset, de_offset, inr_offset
                
    def RADECShift(self, obj_xdp, obj_ydp, obj_int, obj_flag, v0, v1):        
        cat_xdp_0 = v0[:,0]
        cat_ydp_0 = v0[:,1]
        cat_mag_0 = v0[:,2]
        dxra_0    = v0[:,3]
        dyra_0    = v0[:,4]
        dxde_0    = v0[:,5]
        dyde_0    = v0[:,6]
        dxinr_0   = v0[:,7]
        dyinr_0   = v0[:,8]
        
        cat_xdp_1 = v1[:,0]
        cat_ydp_1 = v1[:,1]
        cat_mag_1 = v1[:,2]
        dxra_1    = v1[:,3]
        dyra_1    = v1[:,4]
        dxde_1    = v1[:,5]
        dyde_1    = v1[:,6]
        dxinr_1   = v1[:,7]
        dyinr_1   = v1[:,8]

        dxra  = (dxra_0  + dxra_1 )/2.0
        dyra  = (dyra_0  + dyra_1 )/2.0
        dxde  = (dxde_0  + dxde_1 )/2.0
        dyde  = (dyde_0  + dyde_1 )/2.0
        dxinr = (dxinr_0 + dxinr_1)/2.0
        dyinr = (dyinr_0 + dyinr_1)/2.0

        flg = np.where(obj_flag==1.0)
                
        n_obj = (obj_xdp.shape)[0]
        xdiff_0 = np.transpose([obj_xdp])-cat_xdp_0
        ydiff_0 = np.transpose([obj_ydp])-cat_ydp_0
        xdiff_1 = np.transpose([obj_xdp])-cat_xdp_1
        ydiff_1 = np.transpose([obj_ydp])-cat_ydp_1

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[flg]=xdiff_1[flg]
        ydiff[flg]=ydiff_1[flg]

        dist  = np.sqrt(xdiff**2+ydiff**2)
        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj)),min_dist_index
        rCRA = np.median((xdiff[min_dist_indices]*dyde[min_dist_index]-ydiff[min_dist_indices]*dxde[min_dist_index])/(dxra[min_dist_index]*dyde[min_dist_index]-dyra[min_dist_index]*dxde[min_dist_index]))
        rCDE = np.median((xdiff[min_dist_indices]*dyra[min_dist_index]-ydiff[min_dist_indices]*dxra[min_dist_index])/(dxde[min_dist_index]*dyra[min_dist_index]-dyde[min_dist_index]*dxra[min_dist_index]))

        xdiff_0 = np.transpose([obj_xdp])-(cat_xdp_0+rCRA*dxra+rCDE*dxde)
        ydiff_0 = np.transpose([obj_ydp])-(cat_ydp_0+rCRA*dyra+rCDE*dyde)

        xdiff_1 = np.transpose([obj_xdp])-(cat_xdp_1+rCRA*dxra+rCDE*dxde)
        ydiff_1 = np.transpose([obj_ydp])-(cat_ydp_1+rCRA*dyra+rCDE*dyde)

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[flg]=xdiff_1[flg]
        ydiff[flg]=ydiff_1[flg]
        
        dist  = np.sqrt(xdiff**2+ydiff**2)

        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj)),min_dist_index

        f = dist[min_dist_indices]<2

        # print(f)
        
        match_obj_xdp  = obj_xdp[f]
        match_obj_ydp  = obj_ydp[f]
        match_obj_int  = obj_int[f]
        match_obj_flag = obj_flag[f]

        match_cat_xdp_0 = (cat_xdp_0[min_dist_index])[f]
        match_cat_ydp_0 = (cat_ydp_0[min_dist_index])[f]
        match_cat_mag_0 = (cat_mag_0[min_dist_index])[f]
        match_dxra_0    = (dxra_0[min_dist_index])[f]
        match_dyra_0    = (dyra_0[min_dist_index])[f]
        match_dxde_0    = (dxde_0[min_dist_index])[f]
        match_dyde_0    = (dyde_0[min_dist_index])[f]
        match_dxinr_0   = (dxinr_0[min_dist_index])[f]
        match_dyinr_0   = (dyinr_0[min_dist_index])[f]

        match_cat_xdp_1 = (cat_xdp_1[min_dist_index])[f]
        match_cat_ydp_1 = (cat_ydp_1[min_dist_index])[f]
        match_cat_mag_1 = (cat_mag_1[min_dist_index])[f]
        match_dxra_1    = (dxra_1[min_dist_index])[f]
        match_dyra_1    = (dyra_1[min_dist_index])[f]
        match_dxde_1    = (dxde_1[min_dist_index])[f]
        match_dyde_1    = (dyde_1[min_dist_index])[f]
        match_dxinr_1   = (dxinr_1[min_dist_index])[f]
        match_dyinr_1   = (dyinr_1[min_dist_index])[f]

        match_cat_xdp = np.copy(match_cat_xdp_0)
        match_cat_ydp = np.copy(match_cat_ydp_0)
        match_cat_mag = np.copy(match_cat_mag_0)
        match_dxra    = np.copy(match_dxra_0)
        match_dyra    = np.copy(match_dyra_0)
        match_dxde    = np.copy(match_dxde_0)
        match_dyde    = np.copy(match_dyde_0)
        match_dxinr   = np.copy(match_dxinr_0)
        match_dyinr   = np.copy(match_dyinr_0)

        flg = np.where(match_obj_flag==1.0)

        match_cat_xdp[flg] = match_cat_xdp_1[flg]
        match_cat_ydp[flg] = match_cat_ydp_1[flg]
        match_cat_mag[flg] = match_cat_mag_1[flg]
        match_dxra[flg]    = match_dxra_1[flg]
        match_dyra[flg]    = match_dyra_1[flg]
        match_dxde[flg]    = match_dxde_1[flg]
        match_dyde[flg]    = match_dyde_1[flg]
        match_dxinr[flg]   = match_dxinr_1[flg]
        match_dyinr[flg]   = match_dyinr_1[flg]

        dra  = np.concatenate([match_dxra,match_dyra])
        dde  = np.concatenate([match_dxde,match_dyde])
        dinr = np.concatenate([match_dxinr,match_dyinr])

        basis= np.stack([dra,dde]).transpose()

        errx = match_obj_xdp - match_cat_xdp 
        erry = match_obj_ydp - match_cat_ydp 
        err  = np.concatenate([errx,erry])

        M = np.zeros((2,2))
        b = np.zeros((2))
        for itr1 in range(2):
            for itr2 in range(2):
                M[itr1,itr2] = np.sum(basis[:,itr1]*basis[:,itr2])
            b[itr1] = np.sum(basis[:,itr1]*err)

        A = np.linalg.solve(M,b)
        
        ra_offset  = A[0] * d_ra
        de_offset  = A[1] * d_de
        inr_offset = 0.0
        
        return ra_offset, de_offset, inr_offset
                
    # def RADECShift(self, obj_xdp, obj_ydp, obj_int, \
    #                cat_xdp, cat_ydp, cat_mag,\
    #                dxra,dyra,dxde,dyde,dxinr,dyinr):
    #     n_obj = (obj_xdp.shape)[0]
    #     xdiff = np.transpose([obj_xdp])-cat_xdp
    #     ydiff = np.transpose([obj_ydp])-cat_ydp
    #     dist  = np.sqrt(xdiff**2+ydiff**2)
    #     min_dist_index   = np.nanargmin(dist, axis=1)
    #     min_dist_indices = np.array(range(n_obj)),min_dist_index
    #     rCRA = np.median((xdiff[min_dist_indices]*dyde[min_dist_index]-ydiff[min_dist_indices]*dxde[min_dist_index])/(dxra[min_dist_index]*dyde[min_dist_index]-dyra[min_dist_index]*dxde[min_dist_index]))
    #     rCDE = np.median((xdiff[min_dist_indices]*dyra[min_dist_index]-ydiff[min_dist_indices]*dxra[min_dist_index])/(dxde[min_dist_index]*dyra[min_dist_index]-dyde[min_dist_index]*dxra[min_dist_index]))

    #     xdiff = np.transpose([obj_xdp])-(cat_xdp+rCRA*dxra+rCDE*dxde)
    #     ydiff = np.transpose([obj_ydp])-(cat_ydp+rCRA*dyra+rCDE*dyde)

    #     dist  = np.sqrt(xdiff**2+ydiff**2)

    #     min_dist_index   = np.nanargmin(dist, axis=1)
    #     min_dist_indices = np.array(range(n_obj)),min_dist_index

    #     f = dist[min_dist_indices]<2

    #     #print(f)
        
    #     match_obj_xdp = obj_xdp[f]
    #     match_obj_ydp = obj_ydp[f]
    #     match_obj_int = obj_int[f]
    #     match_cat_xdp = (cat_xdp[min_dist_index])[f]
    #     match_cat_ydp = (cat_ydp[min_dist_index])[f]
    #     match_cat_mag = (cat_mag[min_dist_index])[f]
    #     match_dxra  = (dxra[min_dist_index])[f]
    #     match_dyra  = (dyra[min_dist_index])[f]
    #     match_dxde  = (dxde[min_dist_index])[f]
    #     match_dyde  = (dyde[min_dist_index])[f]
    #     match_dxinr = (dxinr[min_dist_index])[f]
    #     match_dyinr = (dyinr[min_dist_index])[f]

    #     dra  = np.concatenate([match_dxra,match_dyra])
    #     dde  = np.concatenate([match_dxde,match_dyde])
    #     dinr = np.concatenate([match_dxinr,match_dyinr])

    #     basis= np.stack([dra,dde]).transpose()

    #     errx = match_obj_xdp - match_cat_xdp 
    #     erry = match_obj_ydp - match_cat_ydp 
    #     err  = np.concatenate([errx,erry])

    #     M = np.zeros((2,2))
    #     b = np.zeros((2))
    #     for itr1 in range(2):
    #         for itr2 in range(2):
    #             M[itr1,itr2] = np.sum(basis[:,itr1]*basis[:,itr2])
    #         b[itr1] = np.sum(basis[:,itr1]*err)

    #     A = np.linalg.solve(M,b)
        
    #     ra_offset  = A[0] * d_ra
    #     de_offset  = A[1] * d_de
        
    #     return ra_offset, de_offset, 0.0
                
    def makeBasis(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        sep0,zpa0 = Subaru.starSepZPA(self, tel_ra, tel_de, str_ra, str_de, wl, t)
        sep1,zpa1 = Subaru.starSepZPA(self, tel_ra+d_ra, tel_de, str_ra, str_de, wl, t)
        sep2,zpa2 = Subaru.starSepZPA(self, tel_ra, tel_de+d_de, str_ra, str_de, wl, t)

        z = np.zeros_like(sep0)
        o = np.ones_like(sep0)
        
        xfp0_0,yfp0_0 = POPT2.celestial2focalplane(self, sep0,zpa0,adc,m2pos3,wl,z)
        xfp1_0,yfp1_0 = POPT2.celestial2focalplane(self, sep1,zpa1,adc,m2pos3,wl,z)
        xfp2_0,yfp2_0 = POPT2.celestial2focalplane(self, sep2,zpa2,adc,m2pos3,wl,z)

        xfp0_1,yfp0_1 = POPT2.celestial2focalplane(self, sep0,zpa0,adc,m2pos3,wl,o)
        xfp1_1,yfp1_1 = POPT2.celestial2focalplane(self, sep1,zpa1,adc,m2pos3,wl,o)
        xfp2_1,yfp2_1 = POPT2.celestial2focalplane(self, sep2,zpa2,adc,m2pos3,wl,o)

        xfp0 = 0.5*(xfp0_0+xfp0_1)
        xfp1 = 0.5*(xfp1_0+xfp1_1)
        xfp2 = 0.5*(xfp2_0+xfp2_1)

        yfp0 = 0.5*(yfp0_0+yfp0_1)
        yfp1 = 0.5*(yfp1_0+yfp1_1)
        yfp2 = 0.5*(yfp2_0+yfp2_1)

        xdp0,ydp0 = PFS.fp2dp(self, xfp0,yfp0,inr)
        xdp1,ydp1 = PFS.fp2dp(self, xfp1,yfp1,inr)
        xdp2,ydp2 = PFS.fp2dp(self, xfp2,yfp2,inr)
        xdp3,ydp3 = PFS.fp2dp(self, xfp0,yfp0,inr+d_inr)

        xdp0_0,ydp0_0 = PFS.fp2dp(self, xfp0_0,yfp0_0,inr)
        xdp1_0,ydp1_0 = PFS.fp2dp(self, xfp1_0,yfp1_0,inr)
        xdp2_0,ydp2_0 = PFS.fp2dp(self, xfp2_0,yfp2_0,inr)
        xdp3_0,ydp3_0 = PFS.fp2dp(self, xfp0_0,yfp0_0,inr+d_inr)

        xdp0_1,ydp0_1 = PFS.fp2dp(self, xfp0_1,yfp0_1,inr)
        xdp1_1,ydp1_1 = PFS.fp2dp(self, xfp1_1,yfp1_1,inr)
        xdp2_1,ydp2_1 = PFS.fp2dp(self, xfp2_1,yfp2_1,inr)
        xdp3_1,ydp3_1 = PFS.fp2dp(self, xfp0_1,yfp0_1,inr+d_inr)

        dxdpdra = xdp1-xdp0
        dydpdra = ydp1-ydp0
        dxdpdde = xdp2-xdp0
        dydpdde = ydp2-ydp0
        dxdpdinr= xdp3-xdp0
        dydpdinr= ydp3-ydp0

        dxdpdra_0 = xdp1_0-xdp0_0
        dydpdra_0 = ydp1_0-ydp0_0
        dxdpdde_0 = xdp2_0-xdp0_0
        dydpdde_0 = ydp2_0-ydp0_0
        dxdpdinr_0= xdp3_0-xdp0_0
        dydpdinr_0= ydp3_0-ydp0_0

        dxdpdra_1 = xdp1_1-xdp0_1
        dydpdra_1 = ydp1_1-ydp0_1
        dxdpdde_1 = xdp2_1-xdp0_1
        dydpdde_1 = ydp2_1-ydp0_1
        dxdpdinr_1= xdp3_1-xdp0_1
        dydpdinr_1= ydp3_1-ydp0_1

        v_a = np.transpose(np.stack([xdp0,ydp0,dxdpdra,dydpdra,dxdpdde,dydpdde,dxdpdinr,dydpdinr]))
        v_0 = np.transpose(np.stack([xdp0_0,ydp0_0,dxdpdra_0,dydpdra_0,dxdpdde_0,dydpdde_0,dxdpdinr_0,dydpdinr_0]))
        v_1 = np.transpose(np.stack([xdp0_1,ydp0_1,dxdpdra_1,dydpdra_1,dxdpdde_1,dydpdde_1,dxdpdinr_1,dydpdinr_1]))

        # return xdp0,ydp0, dxdpdra,dydpdra, dxdpdde,dydpdde, dxdpdinr,dydpdinr
        return v_0,v_1
        
class PFS():
    def fp2dp(self, xt, yt, inr_deg):
        inr = np.deg2rad(inr_deg)
        x = (xt*np.cos(inr)+yt*np.sin(inr))+pfs_detector_zero_offset_x
        y = (xt*np.sin(inr)-yt*np.cos(inr))+pfs_detector_zero_offset_y

        return x,y

    def dp2fp(self, xc, yc, inr_deg):
        inr = np.deg2rad(inr_deg)
        x = (xc-pfs_detector_zero_offset_x)*np.cos(inr)+(yc-pfs_detector_zero_offset_y)*np.sin(inr)
        y = (xc-pfs_detector_zero_offset_x)*np.sin(inr)-(yc-pfs_detector_zero_offset_y)*np.cos(inr)

        return x,y

    def agarray2momentdifference(self, array, maxellip, maxsize, minsize):
        ##### array 
        ### ccdid objectid xcent[mm] ycent[mm] flx[counts] semimajor[pix] semiminor[pix] Flag[0 or 1]
        filtered_agarray = POPT2.sourceFilter(self, array, maxellip, maxsize, minsize)
        outarray=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        for ccdid in range(1,7):
            array = filtered_agarray[np.where(filtered_agarray[:,0]==ccdid)]
            array_wosp = array[np.where(array[:,7]==0)]
            array_wisp = array[np.where(array[:,7]==1)]

            moment_wosp = np.median((array_wosp[:,5]**2+array_wosp[:,6]**2)*4)
            moment_wisp = np.median((array_wisp[:,5]**2+array_wisp[:,6]**2)*4)

            outarray[ccdid-1]=moment_wosp-moment_wisp

        return outarray
    
    def momentdifference2focuserror(self, momentdifference):
        # momentdifference [pixel^2]
        # focuserror [mm]
        focuserror = momentdifference*0.0086-0.026

        return focuserror
    
###
if __name__ == "__main__":
    print('basic functions for Subaru telescope, POPT2 and PFS')
