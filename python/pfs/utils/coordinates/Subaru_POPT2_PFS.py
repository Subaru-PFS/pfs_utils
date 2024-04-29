# -*- coding: utf-8 -*-
import numpy as np

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac

### unknown scale factor
Unknown_Scale_Factor_AG = 1.0 + 6.2e-04 # focus offset glass added 20230421   # + 1.7e-04
Unknown_Scale_Factor_cobra = 1.0 - 5.0e-5

### constants proper to WFC optics
wfc_scale_M2POS3_coeff = 1.01546e-4

### Subaru location
sbr_lat =   +19.8255
sbr_lon =  +204.523972222
sbr_hei = +4163.0
Lsbr = ac.EarthLocation(lat=sbr_lat,lon=sbr_lon,height=sbr_hei)

### misc.
sbr_press  = 620.0

### constants proper to PFS camera
inr_axis_on_fp_x  = +0.00 # in mm, should be zero
inr_axis_on_fp_y  = +0.00 # in mm, should be zero
inr_zero_offset   = +0.00 # in degree

# inr_axis_on_dp_x  = +0.15 # in mm, from insrot observation on 2022/06
# inr_axis_on_dp_y  = +0.03 # in mm, from insrot observation on 2022/06

# inr_axis_on_dp_x  =  0.03 # in mm, from insrot observation on 2023/04
# inr_axis_on_dp_y  = -0.01 # in mm, from insrot observation on 2023/04

inr_axis_on_dp_x  =  0.00 # in mm
inr_axis_on_dp_y  =  0.00 # in mm

inr_axis_on_pfi_x = inr_axis_on_dp_y
inr_axis_on_pfi_y = inr_axis_on_dp_x

# pfs_inr_zero_offset        =  0.00 # in degree
# pfs_detector_zero_offset_x =  0.00 # in mm
# pfs_detector_zero_offset_y =  0.00 # in mm

### definition
# fp  : telescope coordinate, +xfp  = Opt,   -xfp  = Ir,    +yfp  = Rear,  -yfp  = Front  at any InR
# dp  : rotating coordinate,  +xdp  = Opt,   -xdp  = Ir,    +ydp  = Front, -ydp  = Rear   at InR=0
#                             +xdp  = Rear,  -xdp  = Front, +ydp  = Opt,   -ydp  = Ir     at InR=90
# pfi : rotating coordinate,  +xpfi = Front, -xpfi = Rear,  +ypfi = Opt,   -ypfi = Ir     at InR=0
#                             +xpfi = Opt,   -xpfi = Ir,    +ypfi = Rear   -ypfi = Front  at InR=90

### pfi parity (flip y)
pfi_parity = -1.0 # -1 or +1, 

class Subaru():
    def radec2inr(self, tel_ra, tel_de, t):
        pr  = 0.0   # Subaru InR ignore atmospheric refraction
        wl  = 0.62  # so wavelength is selected freely in visible light

        tel_coord = ac.SkyCoord(ra=tel_ra, dec=tel_de, unit=(au.deg, au.deg), frame='fk5',equinox='J2000.0')
        np_coord  = ac.SkyCoord(ra=0.0,    dec=90.0,   unit=(au.deg, au.deg), frame='fk5',equinox=t) # north pole at observing time
        frame_subaru = ac.AltAz(obstime  = t, location = Lsbr, \
                                pressure = pr*au.hPa, obswl = wl*au.micron)
        tel_altaz = tel_coord.transform_to(frame_subaru)
        np_altaz  = np_coord.transform_to(frame_subaru)
        inr_cal = (tel_altaz.position_angle(np_altaz).degree-180)%360-180
        return inr_cal

    def radec2azel(self, tel_ra, tel_de, wl, t):
        tel_coord = ac.SkyCoord(ra=tel_ra, dec=tel_de, unit=(au.deg, au.deg), frame='fk5',equinox='J2000.0')
        frame_subaru = ac.AltAz(obstime  = t, location = Lsbr, \
                                pressure = sbr_press*au.hPa, obswl = wl*au.micron)
        tel_altaz = tel_coord.transform_to(frame_subaru)
        return tel_altaz.az.degree,tel_altaz.alt.degree

    def starSepZPA(self, tel_ra, tel_de, str_ra, str_de, wl, t):
        tel_coord = ac.SkyCoord(ra=tel_ra, dec=tel_de, unit=(au.deg, au.deg), frame='fk5')
        str_coord = ac.SkyCoord(ra=str_ra, dec=str_de, unit=(au.deg, au.deg), frame='fk5')
        frame_subaru = ac.AltAz(obstime  = t, location = Lsbr,\
                                pressure = sbr_press*au.hPa, obswl = wl*au.micron)
        tel_altaz = tel_coord.transform_to(frame_subaru)
        str_altaz = str_coord.transform_to(frame_subaru)

        str_sep =  tel_altaz.separation(str_altaz).degree
        str_zpa = -tel_altaz.position_angle(str_altaz).degree

        return str_sep, str_zpa

    def starRADEC(self, tel_ra, tel_de, str_sep, str_zpa, wl, t):
        str_sep = str_sep*au.degree
        str_zpa = str_zpa*au.degree

        tel_coord = ac.SkyCoord(ra=tel_ra, dec=tel_de, unit=(au.deg, au.deg), frame='fk5')
        frame_subaru = ac.AltAz(obstime  = t, location = Lsbr,\
                                pressure = sbr_press*au.hPa, obswl = wl*au.micron)
        tel_altaz = tel_coord.transform_to(frame_subaru)
        str_altaz = tel_altaz.directional_offset_by(str_zpa, str_sep)
        str_coord = str_altaz.transform_to('fk5')
        ra = str_coord.ra.degree
        de = str_coord.dec.degree
        return ra, de

    def radec2radecplxpm(self, gaia_epoch, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t):
        str_plx[np.where(str_plx<0.00001)]=0.00001
        str_coord = ac.SkyCoord(ra=str_ra, dec=str_de, unit=(au.deg, au.deg),
                                distance=ac.Distance(parallax=str_plx * au.mas, allow_negative=False),             
                                pm_ra_cosdec = str_pmRA * au.mas/au.yr,
                                pm_dec = str_pmDE * au.mas/au.yr,
                                obstime=at.Time(gaia_epoch, format='decimalyear'),
                                frame='fk5')
        str_coord_obstime = str_coord.apply_space_motion(at.Time(t))
        ra = str_coord_obstime.ra.degree
        de = str_coord_obstime.dec.degree
        return ra, de

class POPT2():
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
    
    def coeffs_AG_c2f_wisp(self, wl, adc): # celestial to focal plane for AG with glass
        D = POPT2.diff_index_pbl1yold_bsl7y(self, wl)
        cx = np.zeros((11))
        cy = np.zeros((11))

        cx[0] =  2.62676925e+02 +(-1.87221607e-06 +2.92222089e-06*D)*adc**2
        cx[1] =  3.35307347e+00 +(-7.81712660e-09 -1.05819104e-09*D)*adc**2
        cx[2] =  2.38166168e-01 +(-2.03764107e-08 -7.38480616e-07*D)*adc**2
        cx[3] =  2.36440726e-02 +( 5.96839761e-09 -1.07538100e-06*D)*adc**2
        cx[4] = -2.62961444e-03 +( 8.86909851e-09 -9.14461129e-07*D)*adc**2
        cx[5] =  2.20267977e-03 +(-4.30901437e-08 +1.07839312e-06*D)*adc**2
        cx[6] =                  ( 5.03828902e-04 -3.02018655e-02*D)*adc
        cx[7] =                  ( 1.62039347e-04 -5.25199388e-03*D)*adc
        cx[8] =  0.00000000e+00
        cx[9] =  0.00000000e+00
        cx[10]=  0.00000000e+00

        cy[0] =  2.62676925e+02 +(-8.86455750e-06 +3.15626958e-06*D)*adc**2
        cy[1] =  3.35307347e+00 +(-6.26053110e-07 -2.95393149e-07*D)*adc**2
        cy[2] =  2.38166168e-01 +(-1.50434841e-07 -1.29919261e-06*D)*adc**2
        cy[3] =  2.36440726e-02 +(-1.29277787e-07 -1.83675834e-06*D)*adc**2
        cy[4] = -2.62961444e-03 +(-5.72922063e-08 -1.27892174e-06*D)*adc**2
        cy[5] =  2.20267977e-03 +(-1.11704310e-08 +1.53690637e-06*D)*adc**2
        cy[6] =                 +(-4.99201211e-04 +3.00941960e-02*D)*adc
        cy[7] =                 +(-1.14622807e-04 +4.07269286e-03*D)*adc
        cy[8] =                 +( 2.26420481e-02 -7.33125190e-01*D)*adc
        cy[9] =                 +( 9.65017704e-04 -3.01504932e-02*D)*adc
        cy[10]=                 +( 1.76747572e-04 -4.84455318e-03*D)*adc

        return cx, cy

    def coeffs_AG_f2c_wisp(self, wl, adc): # focal plane to celestial for AG with glass
        D = POPT2.diff_index_pbl1yold_bsl7y(self, wl)
        cx = np.zeros((11))
        cy = np.zeros((11))
        
        cx[0] =  1.43733003e-02 +( 8.72090395e-11 +0.00000000e+00*D)*adc**2
        cx[1] = -1.88237022e-04 +(-7.82457037e-12 +1.21758496e-11*D)*adc**2
        cx[2] = -6.78700730e-06 +( 1.66205167e-12 +1.81935028e-11*D)*adc**2
        cx[3] = -3.85793541e-07 +( 8.07898607e-13 +2.19246566e-11*D)*adc**2
        cx[4] =  2.67691166e-07 +( 1.73923452e-13 +1.67246499e-11*D)*adc**2
        cx[5] = -1.20869185e-07 +( 9.09157007e-13 -2.89301337e-11*D)*adc**2
        cx[6] =                 +( 2.29093928e-08 +8.45286748e-09*D)*adc
        cx[7] =                 +(-2.12409671e-09 +3.88359241e-08*D)*adc
        cx[8] =  0.00000000e+00
        cx[9] =  0.00000000e+00
        cx[10]=  0.00000000e+00
        
        cy[0] =  1.43733003e-02 +( 4.61943503e-10 +0.00000000e+00*D)*adc**2
        cy[1] = -1.88237022e-04 +( 1.26225040e-11 +8.53001285e-12*D)*adc**2
        cy[2] = -6.78700730e-06 +( 8.50721539e-12 +2.59588237e-11*D)*adc**2
        cy[3] = -3.85793541e-07 +( 7.98095083e-12 +6.92047432e-11*D)*adc**2
        cy[4] =  2.67691166e-07 +( 1.36129621e-12 +8.45438550e-11*D)*adc**2
        cy[5] = -1.20869185e-07 +( 4.48176515e-13 -7.66773681e-11*D)*adc**2
        cy[6] =                 +(-2.31614663e-08 -3.71361360e-09*D)*adc
        cy[7] =                 +( 3.43468003e-10 +2.70608246e-10*D)*adc
        cy[8] =                 +(-1.19076885e-06 +3.85419528e-05*D)*adc
        cy[9] =                 +(-2.61964016e-09 +5.85399688e-09*D)*adc
        cy[10]=                 +(-2.25081043e-09 +2.51123479e-08*D)*adc

        return cx, cy
        
    def coeffs_AG_c2f_wosp(self, wl, adc): # celestial to focal plane for AG without glass
        D = POPT2.diff_index_pbl1yold_bsl7y(self, wl)
        cx = np.zeros((11))
        cy = np.zeros((11))
        
        cx[0] =  2.62647800e+02 +(-1.87011330e-06 +2.77589523e-06*D)*adc**2
        cx[1] =  3.35086900e+00 +(-7.57865771e-09 +5.53973051e-08*D)*adc**2
        cx[2] =  2.38964699e-01 +(-1.94709286e-08 -7.73207447e-07*D)*adc**2
        cx[3] =  2.60230125e-02 +( 6.60545302e-09 -1.16130987e-06*D)*adc**2
        cx[4] = -1.83104099e-03 +( 1.57806929e-08 -1.02698829e-06*D)*adc**2
        cx[5] =  2.04091566e-03 +(-4.93364523e-08 +1.22931907e-06*D)*adc**2
        cx[6] =                  ( 5.04570462e-04 -3.02466062e-02*D)*adc
        cx[7] =                  ( 1.68473464e-04 -5.48576870e-03*D)*adc
        cx[8] =  0.00000000e+00
        cx[9] =  0.00000000e+00
        cx[10]=  0.00000000e+00

        cy[0] =  2.62647800e+02 +(-8.86916339e-06 +3.23008109e-06*D)*adc**2
        cy[1] =  3.35086900e+00 +(-6.08839423e-07 -5.65253675e-07*D)*adc**2
        cy[2] =  2.38964699e-01 +(-1.41970867e-07 -1.67270348e-06*D)*adc**2
        cy[3] =  2.60230125e-02 +(-1.41280083e-07 -1.52144095e-06*D)*adc**2
        cy[4] = -1.83104099e-03 +(-5.94064581e-08 -1.16255139e-06*D)*adc**2
        cy[5] =  2.04091566e-03 +(-9.97915337e-09 +1.46853795e-06*D)*adc**2
        cy[6] =                  (-4.99779792e-04 +3.01309493e-02*D)*adc
        cy[7] =                  (-1.17250417e-04 +4.18415867e-03*D)*adc
        cy[8] =                  ( 2.26486871e-02 -7.33054030e-01*D)*adc
        cy[9] =                  ( 9.65729204e-04 -3.01924555e-02*D)*adc
        cy[10]=                  ( 1.81933708e-04 -5.03118775e-03*D)*adc

        return cx, cy

    def coeffs_AG_f2c_wosp(self, wl, adc): # focal plane to celestial for AG without glass
        D = POPT2.diff_index_pbl1yold_bsl7y(self, wl)
        cx = np.zeros((11))
        cy = np.zeros((11))
        
        cx[0] =  1.43748020e-02 +( 8.66331254e-11 +1.93388878e-11*D)*adc**2
        cx[1] = -1.88232442e-04 +(-7.82853227e-12 +8.59386158e-12*D)*adc**2
        cx[2] = -6.87397752e-06 +( 1.61715989e-12 +2.00354352e-11*D)*adc**2
        cx[3] = -5.21300344e-07 +( 7.29677304e-13 +2.69615671e-11*D)*adc**2
        cx[4] =  2.39570945e-07 +(-2.82067547e-13 +2.51776050e-11*D)*adc**2
        cx[5] = -1.08241906e-07 +( 1.28605646e-12 -3.83590010e-11*D)*adc**2
        cx[6] =                  ( 2.29520266e-08 +9.04783592e-09*D)*adc
        cx[7] =                  (-2.24317632e-09 +4.37694338e-08*D)*adc
        cx[8] =  0.00000000e+00
        cx[9] =  0.00000000e+00
        cx[10]=  0.00000000e+00
        
        cy[0] =  1.43748020e-02 +( 4.61979705e-10 +3.65991127e-12*D)*adc**2
        cy[1] = -1.88232442e-04 +( 1.16038555e-11 +2.66431645e-11*D)*adc**2
        cy[2] = -6.87397752e-06 +( 8.12842568e-12 +4.41948555e-11*D)*adc**2
        cy[3] = -5.21300344e-07 +( 8.66566778e-12 +4.92555793e-11*D)*adc**2
        cy[4] =  2.39570945e-07 +( 1.33811721e-12 +8.16466379e-11*D)*adc**2
        cy[5] = -1.08241906e-07 +( 4.16136826e-13 -7.37253146e-11*D)*adc**2
        cy[6] =                  (-2.32140319e-08 -3.87365915e-09*D)*adc
        cy[7] =                  ( 3.16034703e-10 -6.95499653e-11*D)*adc
        cy[8] =                  (-1.19123300e-06 +3.85419594e-05*D)*adc
        cy[9] =                  (-2.58356822e-09 +6.26527802e-09*D)*adc
        cy[10]=                  (-2.32047805e-09 +2.81735279e-08*D)*adc

        return cx, cy
    
    def coeffs_COBRA_c2f(self, wl, adc): # celestial to focal plane for COBRA
        D = POPT2.diff_index_pbl1yold_bsl7y(self, wl)
        L = POPT2.band_parameter(self, wl)
        cx = np.zeros((11))
        cy = np.zeros((11))

        cx[0] = (+2.62789699e+02)+(+1.41436855e-02*L)+(+6.58982982e-03*L**2)+(-9.48051092e-03*L**3) + (-1.00694635e-06 +(-2.48248836e-05*D))*adc**2
        cx[1] = (+3.36215747e+00)+(+3.69187393e-03*L)+(-6.74817524e-03*L**2)+(+1.54801942e-03*L**3) + (+7.61247574e-09 +(-6.33418826e-07*D))*adc**2
        cx[2] = (+2.35198776e-01)+(+9.42825888e-04*L)+(+9.90958196e-04*L**2)+(-3.17796838e-05*L**3) + (+1.42425375e-08 +(-1.72148497e-06*D))*adc**2
        cx[3] = (+1.46655737e-02)+(-1.79003541e-03*L)+(-1.55308635e-03*L**2)+(+1.13125337e-03*L**3) + (+2.15230267e-08 +(-1.40990705e-06*D))*adc**2
        cx[4] = (-5.52214359e-03)+(-6.97491826e-04*L)+(-9.14390111e-04*L**2)+(+5.80494577e-04*L**3) + (-7.18716293e-09 +(-1.08374155e-06*D))*adc**2
        cx[5] = (+2.71674871e-03)+(+1.09560963e-03*L)+(+4.82675939e-04*L**2)+(-1.53464406e-04*L**3) + (-1.67044496e-08 +(+5.65443437e-07*D))*adc**2
        cx[6] =                                                                                     + (+1.14778526e-04 +(-1.75018873e-02*D))*adc
        cx[7] =                                                                                     + (+8.13032252e-06 +(-1.58749659e-04*D))*adc
        cx[8] =   0.00000000e+00
        cx[9] =   0.00000000e+00
        cx[10]=   0.00000000e+00

        cy[0] = (+2.62789699e+02)+(+1.41436855e-02*L)+(+6.58982982e-03*L**2)+(-9.48051092e-03*L**3) + (-4.18883241e-06 +(-1.47780317e-04*D))*adc**2
        cy[1] = (+3.36215747e+00)+(+3.69187393e-03*L)+(-6.74817524e-03*L**2)+(+1.54801942e-03*L**3) + (-3.63808492e-07 +(-9.59230438e-06*D))*adc**2
        cy[2] = (+2.35198776e-01)+(+9.42825888e-04*L)+(+9.90958196e-04*L**2)+(-3.17796838e-05*L**3) + (-2.93764283e-08 +(-4.75333128e-06*D))*adc**2
        cy[3] = (+1.46655737e-02)+(-1.79003541e-03*L)+(-1.55308635e-03*L**2)+(+1.13125337e-03*L**3) + (+8.30835081e-08 +(-8.68707778e-06*D))*adc**2
        cy[4] = (-5.52214359e-03)+(-6.97491826e-04*L)+(-9.14390111e-04*L**2)+(+5.80494577e-04*L**3) + (+2.58088187e-08 +(-4.49063408e-06*D))*adc**2
        cy[5] = (+2.71674871e-03)+(+1.09560963e-03*L)+(+4.82675939e-04*L**2)+(-1.53464406e-04*L**3) + (-3.73048239e-08 +(+2.68719899e-06*D))*adc**2
        cy[6] =                                                                                     + (-1.14688856e-04 +(+1.75492667e-02*D))*adc
        cy[7] =                                                                                     + (-3.46582624e-06 +(+3.65925894e-04*D))*adc
        cy[8] =                                                                                     + (+2.17560716e-04 +(-6.31969179e-03*D))*adc
        cy[9] =                                                                                     + (+2.29182718e-04 +(-6.21806715e-03*D))*adc
        cy[10]=                                                                                     + (+3.44748694e-05 +(-1.52943095e-04*D))*adc

        return cx, cy
    
    def coeffs_COBRA_f2c(self, wl, adc): # focal plane to celestial for COBRA
        D = POPT2.diff_index_pbl1yold_bsl7y(self, wl)
        L = POPT2.band_parameter(self, wl)
        cx = np.zeros((11))
        cy = np.zeros((11))

        cx[0] = ( 1.43674845e-02)+(-7.43646874e-07*L)+(-3.16130067e-07*L**2)+( 4.80328267e-07*L**3) + ( 5.05131730e-11 +( 1.17283146e-09*D))*adc**2
        cx[1] = (-1.88289474e-04)+(-1.56460268e-07*L)+( 3.80620067e-07*L**2)+(-1.19205317e-07*L**3) + (-4.84876314e-12 +(-7.61093281e-11*D))*adc**2
        cx[2] = (-6.46172751e-06)+(-2.66777511e-08*L)+(-6.33815835e-08*L**2)+(-7.74745171e-09*L**3) + (-9.62743893e-13 +( 9.64732206e-11*D))*adc**2
        cx[3] = ( 1.23562185e-07)+( 9.24192386e-08*L)+( 9.39175516e-08*L**2)+(-6.74785592e-08*L**3) + (-6.67748446e-13 +( 6.60925128e-11*D))*adc**2
        cx[4] = ( 3.68389899e-07)+( 1.03386621e-08*L)+( 3.47784649e-08*L**2)+(-2.36730251e-08*L**3) + ( 1.08679636e-12 +( 2.43537188e-11*D))*adc**2
        cx[5] = (-1.63142810e-07)+(-4.53025978e-08*L)+(-2.49134365e-08*L**2)+( 1.03064829e-08*L**3) + ( 4.18052943e-13 +(-3.13541684e-11*D))*adc**2
        cx[6] =                                                                                     + (-5.39147702e-09 +( 9.19945103e-07*D))*adc
        cx[7] =                                                                                     + ( 2.77202779e-10 +(-4.32936212e-08*D))*adc
        cx[8] =   0.00000000e+0
        cx[9] =   0.00000000e+0
        cx[10]=   0.00000000e+0

        cy[0] = ( 1.43674845e-02)+(-7.43646874e-07*L)+(-3.16130067e-07*L**2)+( 4.80328267e-07*L**3) + ( 2.20715956e-10 +( 7.77431541e-09*D))*adc**2
        cy[1] = (-1.88289474e-04)+(-1.56460268e-07*L)+( 3.80620067e-07*L**2)+(-1.19205317e-07*L**3) + ( 8.06546205e-12 +( 1.99216576e-10*D))*adc**2
        cy[2] = (-6.46172751e-06)+(-2.66777511e-08*L)+(-6.33815835e-08*L**2)+(-7.74745171e-09*L**3) + (-6.15086757e-13 +( 2.97964659e-10*D))*adc**2
        cy[3] = ( 1.23562185e-07)+( 9.24192386e-08*L)+( 9.39175516e-08*L**2)+(-6.74785592e-08*L**3) + (-4.06656270e-12 +( 4.65768165e-10*D))*adc**2
        cy[4] = ( 3.68389899e-07)+( 1.03386621e-08*L)+( 3.47784649e-08*L**2)+(-2.36730251e-08*L**3) + ( 5.85562680e-13 +( 1.36403403e-10*D))*adc**2
        cy[5] = (-1.63142810e-07)+(-4.53025978e-08*L)+(-2.49134365e-08*L**2)+( 1.03064829e-08*L**3) + ( 1.38694343e-12 +(-1.22884800e-10*D))*adc**2
        cy[6] =                                                                                     + ( 5.38704437e-09 +(-9.23542499e-07*D))*adc
        cy[7] =                                                                                     + (-4.39088156e-10 +( 3.02959958e-08*D))*adc
        cy[8] =                                                                                     + (-1.18341815e-08 +( 3.31202350e-07*D))*adc
        cy[9] =                                                                                     + (-1.18199424e-08 +( 2.99799379e-07*D))*adc
        cy[10]=                                                                                     + (-9.77458744e-10 +(-1.95302432e-08*D))*adc

        return cx, cy
    
    def ZX(self, x, y, cx):
        return \
            cx[0]*(x) +\
            cx[1]*((3*(x**2+y**2)-2)*x) +\
            cx[2]*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx[3]*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx[4]*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx[5]*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx[6]*(2*x*y) +\
            cx[7]*((4*(x**2+y**2)-3)*2*x*y) +\
            cx[8]*(1) +\
            cx[9]*(2*(x**2+y**2)-1) +\
            cx[10]*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

    def ZY(self, x, y, cy):
        return \
            cy[0]*(y) +\
            cy[1]*((3*(x**2+y**2)-2)*y) +\
            cy[2]*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy[3]*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy[4]*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy[5]*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy[6]*(x**2-y**2) +\
            cy[7]*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy[8]*(1)+\
            cy[9]*(2*(x**2+y**2)-1) +\
            cy[10]*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

    def celestial2focalplane(self, sep, zpa, adc, inr, el, m2pos3, wl, flag):
        f = np.atleast_1d(flag).astype(float)
        # f = flag.astype(float)
        f[np.where(f>1)]=0.5
        f[np.where(f<0)]=0.5
        xfp_wisp, yfp_wisp = POPT2.celestial2focalplane_wisp(self, sep, zpa, adc, inr, el, m2pos3, wl)
        xfp_wosp, yfp_wosp = POPT2.celestial2focalplane_wosp(self, sep, zpa, adc, inr, el, m2pos3, wl)
        xfp = xfp_wosp*(1-f) + xfp_wisp*(f)
        yfp = yfp_wosp*(1-f) + yfp_wisp*(f)
        return xfp,yfp

    def focalplane2celestial(self, xt, yt, adc, inr, el, m2pos3, wl, flag):
        f = np.atleast_1d(flag).astype(float)
        # f = flag.astype(float)
        f[np.where(f>1)]=0.5
        f[np.where(f<0)]=0.5
        r_wisp, t_wisp = POPT2.focalplane2celestial_wisp(self, xt, yt, adc, inr, el, m2pos3, wl)
        r_wosp, t_wosp = POPT2.focalplane2celestial_wosp(self, xt, yt, adc, inr, el, m2pos3, wl)
        r = r_wosp*(1-f) + r_wisp*(f)
        t = t_wosp*(1-f) + t_wisp*(f)
        return r,t

    def additionaldistortion(self, xt, yt):
        x = xt / 270.0
        y = yt / 270.0
        # data from HSC-PFS test observation on 2020/10/23, 7 sets of InR rotating data.
        adx = \
            +5.96462898e-05 \
            +2.42107822e-03*(2*(x**2+y**2)-1) \
            +3.81085098e-03*(x**2-y**2) \
            -2.75632544e-03*(2*x*y) \
            -1.35748905e-03*((3*(x**2+y**2)-2)*x) \
            +2.12301241e-05*((3*(x**2+y**2)-2)*y) \
            -9.23710861e-05*(6*(x**2+y**2)**2-6*(x**2+y**2)+1)
        ady = \
            -2.06476363e-04 \
            -3.45168908e-03*(2*(x**2+y**2)-1) \
            +4.09091198e-03*(x**2-y**2) \
            +3.41002899e-03*(2*x*y) \
            +1.63045902e-04*((3*(x**2+y**2)-2)*x) \
            -1.01811037e-03*((3*(x**2+y**2)-2)*y) \
            +2.07395905e-04*(6*(x**2+y**2)**2-6*(x**2+y**2)+1)

        return adx, ady

    def additionaldistortion2(self, xt, yt, inr, el):
        x = xt / 270.0
        y = yt / 270.0
        # data from PFS test observation on 2023/04-05
        xx = + x * np.cos(3*inr/180.0*np.pi) + y * np.sin(3*inr/180.0*np.pi)
        yy = - x * np.sin(3*inr/180.0*np.pi) + y * np.cos(3*inr/180.0*np.pi)

        a = 2.8e-02*np.cos(el/180.0*np.pi)
        adx = -a*yy
        ady = -a*xx

        return adx,ady

    def celestial2focalplane_wisp(self, sep, zpa, adc, inr, el, m2pos3, wl):
        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans =  np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        x = tanx/0.014
        y = tany/0.014

        cx,cy = POPT2.coeffs_AG_c2f_wisp(self, wl, adc)
        telx = POPT2.ZX(self, x,y,cx) * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor_AG
        tely = POPT2.ZY(self, x,y,cy) * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor_AG

        adtelx,adtely = POPT2.additionaldistortion(self, telx,tely)
        telx = telx + adtelx
        tely = tely + adtely

        adtelx2,adtely2   = POPT2.additionaldistortion2(self, telx,tely,inr,el)
        telx = telx + adtelx2
        tely = tely + adtely2

        return telx,tely

    def focalplane2celestial_wisp(self, xt, yt, adc, inr, el, m2pos3, wl):
        adtelx2,adtely2 = POPT2.additionaldistortion2(self, xt, yt, inr, el)
        xt = xt - adtelx2
        yt = yt - adtely2
        adtelx,adtely = POPT2.additionaldistortion(self, xt, yt)
        xt = xt - adtelx
        yt = yt - adtely
        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor_AG
        y = y / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor_AG

        cx,cy = POPT2.coeffs_AG_f2c_wisp(self, wl, adc)
        tantx = POPT2.ZX(self, x,y,cx)
        tanty = POPT2.ZY(self, x,y,cy)

        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)
        
        s = np.rad2deg(s)
        t = np.rad2deg(t)

        return s, t

    def celestial2focalplane_wosp(self, sep, zpa, adc, inr, el, m2pos3, wl):
        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans =  np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        x = tanx/0.014
        y = tany/0.014

        cx,cy = POPT2.coeffs_AG_c2f_wosp(self, wl, adc)
        telx = POPT2.ZX(self, x,y,cx) * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor_AG
        tely = POPT2.ZY(self, x,y,cy) * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor_AG

        adtelx,adtely = POPT2.additionaldistortion(self, telx,tely)
        telx = telx + adtelx
        tely = tely + adtely

        adtelx2,adtely2 = POPT2.additionaldistortion2(self, telx,tely,inr,el)
        telx = telx + adtelx2
        tely = tely + adtely2

        return telx, tely

    def focalplane2celestial_wosp(self, xt, yt, adc, inr, el, m2pos3, wl):
        adtelx2,adtely2 = POPT2.additionaldistortion2(self, xt, yt, inr, el)
        xt = xt - adtelx2
        yt = yt - adtely2
        adtelx,adtely = POPT2.additionaldistortion(self, xt, yt)
        xt = xt - adtelx
        yt = yt - adtely
        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor_AG
        y = y / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor_AG

        cx,cy = POPT2.coeffs_AG_f2c_wosp(self, wl, adc)
        tantx = POPT2.ZX(self, x,y,cx)
        tanty = POPT2.ZY(self, x,y,cy)

        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)
        
        s = np.rad2deg(s)
        t = np.rad2deg(t)

        return s, t

    def celestial2focalplane_cobra(self, sep, zpa, adc, inr, el, m2pos3, wl):
        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans =  np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        x = tanx/0.014
        y = tany/0.014

        cx,cy = POPT2.coeffs_COBRA_c2f(self, wl, adc)
        telx = POPT2.ZX(self, x,y,cx) * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor_cobra
        tely = POPT2.ZY(self, x,y,cy) * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor_cobra

        adtelx,adtely = POPT2.additionaldistortion(self, telx,tely)
        telx = telx + adtelx
        tely = tely + adtely

        # We are still not sure distortion found in 2023 April/May run is applied to Cobra.
        # adtelx2,adtely2 = POPT2.additionaldistortion2(self, telx,tely,inr,el)
        # telx = telx + adtelx2
        # tely = tely + adtely2

        return telx,tely

    def focalplane2celestial_cobra(self, xt, yt, adc, inr, el, m2pos3, wl):
        # We are still not sure distortion found in 2023 April/May run is applied to Cobra.
        # adtelx2,adtely2 = POPT2.additionaldistortion2(self, xt, yt, inr, el)
        # xt = xt - adtelx2
        # yt = yt - adtely2
        adtelx,adtely = POPT2.additionaldistortion(self, xt, yt)
        xt = xt - adtelx
        yt = yt - adtely
        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor_cobra
        y = y / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor_cobra

        cx,cy = POPT2.coeffs_COBRA_f2c(self, wl, adc)
        tantx = POPT2.ZX(self, x,y,cx)
        tanty = POPT2.ZY(self, x,y,cy)

        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)
        
        s = np.rad2deg(s)
        t = np.rad2deg(t)

        return s, t

class PFS():
    def fp2dp(self, xfp, yfp, inr_deg):
        inr  = np.deg2rad(inr_deg+inr_zero_offset)
        x    = xfp - inr_axis_on_fp_x
        y    = yfp - inr_axis_on_fp_y
        xdp  = +x*np.cos(inr)+y*np.sin(inr) + inr_axis_on_dp_x
        ydp  = +x*np.sin(inr)-y*np.cos(inr) + inr_axis_on_dp_y
        return xdp, ydp

    def dp2fp(self, xdp, ydp, inr_deg):
        inr  = np.deg2rad(inr_deg+inr_zero_offset)
        x    = xdp - inr_axis_on_dp_x
        y    = ydp - inr_axis_on_dp_y
        xfp  = +x*np.cos(inr)+y*np.sin(inr) + inr_axis_on_fp_x
        yfp  = +x*np.sin(inr)-y*np.cos(inr) + inr_axis_on_fp_y
        return xfp, yfp

    def fp2pfi(self, xfp, yfp, inr_deg):
        inr  = np.deg2rad(inr_deg+inr_zero_offset)
        x    = xfp - inr_axis_on_fp_x
        y    = yfp - inr_axis_on_fp_y
        xpfi = +x*np.sin(inr)-y*np.cos(inr) + inr_axis_on_pfi_x
        ypfi = +x*np.cos(inr)+y*np.sin(inr) + inr_axis_on_pfi_y
        ypfi = ypfi * pfi_parity
        return xpfi, ypfi

    def pfi2fp(self, xpfi, ypfi, inr_deg):
        inr  = np.deg2rad(inr_deg+inr_zero_offset)
        ypfi = ypfi * pfi_parity
        x    = xpfi - inr_axis_on_pfi_x
        y    = ypfi - inr_axis_on_pfi_y
        xfp  = +x*np.sin(inr)+y*np.cos(inr) + inr_axis_on_fp_x
        yfp  = -x*np.cos(inr)+y*np.sin(inr) + inr_axis_on_fp_y
        return xfp, yfp
    
    def dp2pfi(self, xdp, ydp):
        xpfi = ydp
        ypfi = xdp
        ypfi = ypfi * pfi_parity
        return xpfi, ypfi
    
    def pfi2dp(self, xpfi, ypfi):
        ypfi = ypfi * pfi_parity
        xdp  = ypfi
        ydp  = xpfi
        return xdp, ydp


class distCorr():
    def __init__(self):
        self.correction_factor = -1.0

    def xy2dxdy(self, xt, yt):
        #### 2024/03/16, after rot. center correction
        z0x  =    +6.13679631573732e-03
        z1x  =    -1.30385657868192e-02
        z2x  =    +2.38637664956091e-02
        z3x  =    -7.12248999631302e-03
        z4x  =    +1.43989657864945e-02
        z5x  =    +1.71599705259336e-02
        z6x  =    +6.46495265443865e-04
        z7x  =    -9.83219999113761e-05
        z8x  =    +2.92435019965087e-02
        z9x  =    -1.75203496499541e-04
        z12x =    +1.01007499985897e-03
        z0y  =    -3.29466999940409e-03
        z1y  =    +1.55727674977550e-02
        z2y  =    -1.59169510482724e-02
        z3y  =    +1.17914578909012e-02
        z4y  =    -7.62166473415474e-03
        z5y  =    +1.15573584183977e-02
        z6y  =    +9.02940500030079e-03
        z7y  =    +1.90820354988039e-02
        z8y  =    -2.68632299740984e-03
        z9y  =    +5.20598052522325e-03
        z12y =    -1.90307999915411e-03
        
        # xt,yt in mm
        x = xt / 270.0
        y = yt / 270.0
        ox = z0x + z1x*y + z2x*x + z3x*2*x*y + z4x*(2*(x**2+y**2)-1) + z5x*(x**2-y**2) + z6x*( 3*(x**2+y**2)-4*y**2)*y + z7x*( 3*(x**2+y**2)-2)*y + z8x*( 3*(x**2+y**2)-2)*x + z9x*(-3*(x**2+y**2)+4*x**2)*x + z12x*(6*(x**2+y**2)**2-6*(x**2+y**2)+1)
        oy = z0y + z1y*y + z2y*x + z3y*2*x*y + z4y*(2*(x**2+y**2)-1) + z5y*(x**2-y**2) + z6y*( 3*(x**2+y**2)-4*y**2)*y + z7y*( 3*(x**2+y**2)-2)*y + z8y*( 3*(x**2+y**2)-2)*x + z9y*(-3*(x**2+y**2)+4*x**2)*x + z12y*(6*(x**2+y**2)**2-6*(x**2+y**2)+1)

        return ox,oy

###
if __name__ == "__main__":
    print('# basic functions for Subaru telescope, POPT2 and PFS')
