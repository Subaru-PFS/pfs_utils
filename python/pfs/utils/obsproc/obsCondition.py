#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import toml
import matplotlib.pyplot as plt
from pfs.datamodel import Identity, PfsArm
from pfs.datamodel.pfsConfig import PfsConfig, TargetType
from opDB import OpDB
from qaDB import QaDB

__all__ = ["Condition"]


def read_conf(conf):
    config = toml.load(conf)
    return config


class Condition(object):
    """Observing condition

    Parameters
    ----------
    visits : `pfs_visit_id` or `list` of `pfs_visit_id`


    Examples
    ----------

    """

    def __init__(self, visits, conf='config.toml'):
        self.conf = read_conf(conf)       
        self.visits = visits
        self.visitList = []
        self.df = None
        self.opdb = OpDB(self.conf["db"]["opdb"])
        self.qadb = QaDB(self.conf["db"]["qadb"])

    def getAgcData(self):
        sqlWhere = ''
        if type(self.visits) == int:
            sqlWhere = f'agc_exposure.pfs_visit_id={self.visits}'
            self.visitList.append(self.visits)
        elif type(self.visits) == list:
            consts = []
            for v in self.visits:
                consts.append(f'agc_exposure.pfs_visit_id={v}')
                self.visitList.append(v)
            const = ' OR '.join(consts)
            sqlWhere = f'({const})'
        elif type(self.visits) == str:
            if '..' in self.visits:
                v1 = self.visits.split('..')[0]
                v2 = self.visits.split('..')[1]
                sqlWhere = f'agc_exposure.pfs_visit_id>={v1} AND agc_exposure.pfs_visit_id<={v2}'
                self.visitList += [v for v in range(int(v1), int(v2)+1)]
            elif '^' in self.visits:
                visits = self.visits.split('^')
                consts = []
                for v in visits:
                    consts.append(f'agc_exposure.pfs_visit_id={v}')
                    self.visitList.append(v)
                const = ' OR '.join(consts)
                sqlWhere = f'({const})'
        sqlCmd = f'SELECT agc_exposure.pfs_visit_id,agc_exposure.agc_exposure_id,agc_exposure.taken_at,agc_data.agc_camera_id,central_image_moment_11_pix,central_image_moment_20_pix,central_image_moment_02_pix,background,estimated_magnitude,agc_data.flags,agc_match.guide_star_id,pfs_design_agc.guide_star_magnitude FROM agc_exposure JOIN agc_data ON agc_exposure.agc_exposure_id=agc_data.agc_exposure_id JOIN agc_match ON agc_data.agc_exposure_id=agc_match.agc_exposure_id AND agc_data.agc_camera_id=agc_match.agc_camera_id AND agc_data.spot_id=agc_match.spot_id JOIN pfs_design_agc ON agc_match.pfs_design_id=pfs_design_agc.pfs_design_id AND agc_match.guide_star_id=pfs_design_agc.guide_star_id WHERE {sqlWhere} AND agc_data.flags<=1;'

        self.df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        self.cameraIds = np.unique(self.df['agc_camera_id'])
        return self.df, self.visitList

    def getSpsExposure(self):
        sqlWhere = ''
        if type(self.visits) == int:
            sqlWhere = f'sps_exposure.pfs_visit_id={self.visits}'
            self.visitList.append(self.visits)
        elif type(self.visits) == list:
            consts = []
            for v in self.visits:
                consts.append(f'sps_exposure.pfs_visit_id={v}')
                self.visitList.append(v)
            const = ' OR '.join(consts)
            sqlWhere = f'({const})'
        elif type(self.visits) == str:
            if '..' in self.visits:
                v1 = self.visits.split('..')[0]
                v2 = self.visits.split('..')[1]
                sqlWhere = f'sps_exposure.pfs_visit_id>={v1} AND sps_exposure.pfs_visit_id<={v2}'
                self.visitList += [v for v in range(int(v1), int(v2)+1)]
            elif '^' in self.visits:
                visits = self.visits.split('^')
                consts = []
                for v in visits:
                    consts.append(f'sps_exposure.pfs_visit_id={v}')
                    self.visitList.append(v)
                const = ' OR '.join(consts)
                sqlWhere = f'({const})'
        sqlCmd = f'SELECT * FROM sps_exposure WHERE {sqlWhere};'

        self.df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        self.spsCameraIds = np.unique(self.df['sps_camera_id'])
        return self.df, self.visitList

    def calcSeeing(self, plot=False, cc='cameraId'):
        """Calculate Seeing size based on AGC measurements

        Parameters
        ----------
            plot : plot the results? {True, False}       (default: False)
              cc : color coded by {'cameraId', 'visit'}  (default: 'cameraId')

        Returns
        ----------
            agc_exposure_id : AGC exposure ID
            taken_at_seq    : The time at which the exposure was taken (in HST)
            fwhm_median     : The median FWHM during the AGC exposure  (arcsec.)
            
        Examples
        ----------

        """
        self.df, _ = self.getAgcData()

        agc_exposure_id = self.df['agc_exposure_id']
        seq = np.unique(agc_exposure_id)
        taken_at = self.df['taken_at']
        pfs_visit_ids = self.df['pfs_visit_id']
        
        ''' reference: Magnier et al. 2020, ApJS, 251, 5 '''
        g1 = self.df['central_image_moment_20_pix'] + self.df['central_image_moment_02_pix']
        g2 = self.df['central_image_moment_20_pix'] - self.df['central_image_moment_02_pix']
        g3 = np.sqrt(g2**2 + 4*self.df['central_image_moment_11_pix']**2)
        sigma_a = self.conf['agc']['ag_pix_scale'] * np.sqrt((g1+g3)/2)
        sigma_b = self.conf['agc']['ag_pix_scale'] * np.sqrt((g1-g3)/2)
        sigma = 0.5 * (sigma_a + sigma_b)
        fwhm = sigma * 2.355

        taken_at_seq = []
        fwhm_mean = []          # calculate median per each AG exposure
        fwhm_median = []          # calculate median per each AG exposure
        fwhm_stddev = []          # calculate median per each AG exposure
        for s in seq:
            taken_at_seq.append(taken_at[agc_exposure_id == s].values[0])
            fwhm_mean.append(fwhm[agc_exposure_id == s].mean(skipna=True))
            fwhm_median.append(fwhm[agc_exposure_id == s].median(skipna=True))
            fwhm_stddev.append(fwhm[agc_exposure_id == s].std(skipna=True))
        visit_p_visit = []
        fwhm_mean_p_visit = []    # calculate mean per visit
        fwhm_median_p_visit = []  # calculate median per visit
        fwhm_stddev_p_visit = []  # calculate sigma per visit
        for v in np.unique(pfs_visit_ids):
            visit_p_visit.append(v)
            fwhm_mean_p_visit.append(fwhm[pfs_visit_ids == v].mean(skipna=True))
            fwhm_median_p_visit.append(fwhm[pfs_visit_ids == v].median(skipna=True))
            fwhm_stddev_p_visit.append(fwhm[pfs_visit_ids == v].std(skipna=True))
            
        ''' insert into qaDB '''
        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit}
            )
        self.qadb.populateQATable('pfs_visit', df)

        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'seeing_mean': fwhm_mean_p_visit,
                  'seeing_median': fwhm_median_p_visit,
                  'seeing_sigma': fwhm_stddev_p_visit
                  }
            )
        self.qadb.populateQATable('seeing', df)
        
        ''' plotting '''
        if plot is True:
            fig = plt.figure(figsize=(8, 5))
            axe = fig.add_subplot()
            axe.set_xlabel('taken_at (HST)')
            axe.set_ylabel('seeing FWHM (arcsec.)')
            axe.set_title(f'visits:{self.visits}')
            axe.set_ylim(0., 2.0)
            if cc == 'cameraId':
                for cid in self.cameraIds:
                    msk = self.df['agc_camera_id'] == cid
                    axe.scatter(taken_at[msk], fwhm[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'cameraId={cid}')
            elif cc == 'visit':
                for v in self.visitList:
                    msk = self.df['pfs_visit_id'] == v
                    axe.scatter(taken_at[msk], fwhm[msk], marker='o', s=10, alpha=0.5, rasterized=True)
            else:
                axe.scatter(taken_at[msk], fwhm[msk], marker='o', s=10, edgecolor='none', facecolor='C0', alpha=0.5, rasterized=True)
            # axe.scatter(taken_at_seq, fwhm_median, marker='o', s=50, edgecolor='none', facecolor='C1', alpha=0.8, rasterized=True)
            axe.plot(taken_at_seq, fwhm_median, ls='solid', lw=2, color='k', alpha=0.8)
            axe.plot([min(taken_at_seq), max(taken_at_seq)],
                     [0.8, 0.8], ls='dashed', lw=2, color='k', alpha=0.8)
            axe.legend(loc='upper left', ncol=2, fontsize=8)

        return agc_exposure_id, taken_at_seq, fwhm_mean, fwhm_median, fwhm_stddev, fwhm_mean_p_visit, fwhm_median_p_visit, fwhm_stddev_p_visit

    def calcTransparency(self, plot=False, cc='cameraId'):
        """Calculate Transparency based on AGC measurements

        Parameters
        ----------
            plot : plot the results? {True, False}       (default: False)
              cc : color coded by {'cameraId', 'visit'}  (default: 'cameraId')

        Returns
        ----------
            agc_exposure_id : AGC exposure ID
            taken_at_seq    : The time at which the exposure was taken (in HST)
            transp_median     : The median transparency during the AGC exposure
            
        Examples
        ----------

        """
        self.df, _ = self.getAgcData()

        agc_exposure_id = self.df['agc_exposure_id']
        seq = np.unique(agc_exposure_id)
        taken_at = self.df['taken_at']
        pfs_visit_ids = self.df['pfs_visit_id']

        mag1 = self.df['guide_star_magnitude']
        mag2 = self.df['estimated_magnitude']
        dmag = mag2 - mag1
        transp = 10**(-0.4*dmag) / self.conf['agc']['transparency_correction']

        taken_at_seq = []
        transp_mean = []     # calculate median per each AG exposure
        transp_median = []     # calculate median per each AG exposure
        transp_stddev = []     # calculate median per each AG exposure
        for s in seq:
            taken_at_seq.append(taken_at[agc_exposure_id == s].values[0])
            data = transp[agc_exposure_id == s]
            if len(data[data.notna()]) > 0:
                transp_mean.append(data.mean(skipna=True))
                transp_median.append(data.median(skipna=True))
                transp_stddev.append(data.std(skipna=True))
            else:
                transp_mean.append(np.nan)
                transp_median.append(np.nan)
                transp_stddev.append(np.nan)
                
        visit_p_visit = []
        transp_mean_p_visit = []    # calculate mean per visit
        transp_median_p_visit = []  # calculate median per visit
        transp_stddev_p_visit = []  # calculate sigma per visit
        for v in np.unique(pfs_visit_ids):
            visit_p_visit.append(v)
            data = transp[pfs_visit_ids == v]
            if len(data[data.notna()]) > 0:
                transp_mean_p_visit.append(data.mean(skipna=True))
                transp_median_p_visit.append(data.median(skipna=True))
                transp_stddev_p_visit.append(data.std(skipna=True))
            else:
                transp_mean_p_visit.append(np.nan)
                transp_median_p_visit.append(np.nan)
                transp_stddev_p_visit.append(np.nan)
                
        ''' insert into qaDB '''
        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'transparency_mean': transp_mean_p_visit,
                  'transparency_median': transp_median_p_visit,
                  'transparency_sigma': transp_stddev_p_visit
                  }
            )
        self.qadb.populateQATable('transparency', df)

        ''' plotting '''
        if plot is True:
            '''
            xmin = 12.0
            xmax = 22.0
            ymin = 0.0
            ymax = 2.0
            fig = plt.figure(figsize=(8,5))
            axe = fig.add_subplot()
            axe.set_xlabel('Gaia magnitude (ABmag)')
            axe.set_ylabel(r'transparency ($\times0.5$ tentatively)')
            axe.set_title(f'visits:{self.visits}')
            axe.set_xlim(xmin, xmax)
            axe.set_ylim(ymin, ymax)
            if cc=='cameraId':
                for cid in self.cameraIds:
                    msk = self.df['agc_camera_id']==cid
                    axe.scatter(mag1[msk], transp[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'cameraId={cid}')
            elif cc=='visit':
                for v in self.visitList:
                    msk = self.df['pfs_visit_id']==v
                    axe.scatter(mag1[msk], transp[msk], marker='o', s=10, alpha=0.5, rasterized=True)
            else:
                axe.scatter(mag1, transp, marker='o', s=10, ec='k', fc='C0', alpha=0.5, rasterized=True)
            axe.plot([xmin, xmax], [1.0, 1.0], ls='dashed', color='k')
            axe.legend(loc='upper left', ncol=2, fontsize=8)
            '''
            fig = plt.figure(figsize=(8, 5))
            axe = fig.add_subplot()
            axe.set_xlabel('taken_at (HST)')
            axe.set_ylabel(r'transparency ($\times0.5$ tentatively)')
            axe.set_title(f'visits:{self.visits}')
            axe.set_ylim(0., 2.0)
            if cc == 'cameraId':
                for cid in self.cameraIds:
                    msk = self.df['agc_camera_id'] == cid
                    axe.scatter(taken_at[msk], transp[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'cameraId={cid}')
            elif cc == 'visit':
                for v in self.visitList:
                    msk = self.df['pfs_visit_id'] == v
                    axe.scatter(taken_at[msk], transp[msk], marker='o', s=10, alpha=0.5, rasterized=True)
            else:
                axe.scatter(taken_at[msk], transp[msk], marker='o', s=10, edgecolor='none', facecolor='C0', alpha=0.5, rasterized=True)
            axe.plot(taken_at_seq, transp_median, ls='solid', lw=2, color='k', alpha=0.8)
            axe.plot([min(taken_at_seq), max(taken_at_seq)],
                     [1.0, 1.0], ls='dashed', lw=2, color='k', alpha=0.8)
            axe.legend(loc='upper left', ncol=2, fontsize=8)

        # return mag1, dmag, transp
        return agc_exposure_id, taken_at_seq, transp_mean, transp_median, transp_stddev, transp_mean_p_visit, transp_median_p_visit, transp_stddev_p_visit

    def calcSkyBackground(self, plot=False):
        """Calculate Transparency based on AGC measurements

        Parameters
        ----------
            plot : plot the results? {True, False}       (default: False)
            butler: (default: None)
            
        Returns
        ----------
            pfs_visit_id : pfs_visit_id
            sky_flux_mean : The mean flux of SKY fibers
            sky_flux_mean : The mean flux of SKY fibers
            sky_flux_mean : The mean flux of SKY fibers
            
        Examples
        ----------

        """
        ''' setup butler '''
        _drpDataConf = self.conf["drp"]["data"]
        baseDir = _drpDataConf["base"]
        pfsConfigDir = _drpDataConf["pfsConfigDir"]
        rerun = os.path.join(baseDir, 'rerun', _drpDataConf["rerun"])
        calibRoot = os.path.join(baseDir, _drpDataConf["calib"])
        try:
            import lsst.daf.persistence as dafPersist
            butler = dafPersist.Butler(rerun, calibRoot=calibRoot)
        except:
            butler = None

        _skyQaConf = self.conf["qa"]["sky"]["config"]
            
        ''' get visit information '''
        self.df, _ = self.getSpsExposure()
        pfs_visit_ids = self.df['pfs_visit_id']
        sps_camera_ids = self.df['sps_camera_id']
        exptimes = self.df['exptime']
        obstime = self.df['time_exp_end'][0]
        obstime = obstime.tz_localize('US/Hawaii')
        obstime = obstime.tz_convert('UTC')
        obsdate = obstime.date().strftime('%Y-%m-%d')
        
        visit_p_visit = []
        sky_mean_p_visit = []    # calculate mean per visit
        sky_median_p_visit = []  # calculate median per visit
        sky_stddev_p_visit = []  # calculate sigma per visit
        for v in np.unique(pfs_visit_ids):
            if butler is not None:
                dataId = dict(visit=v, spectrograph=_skyQaConf["ref_spectrograph_sky"], arm=_skyQaConf["ref_arm_sky"],
                )
                pfsArm = butler.get("pfsArm", dataId)
                pfsConfig = butler.get("pfsConfig", dataId)
            else:
                _dataDir = os.path.join(rerun, 'pfsArm', obsdate, 'v%06d' % (v))
                _identity = Identity(visit=v, 
                                     arm=_skyQaConf["ref_arm_sky"],
                                     spectrograph=_skyQaConf["ref_spectrograph_sky"]
                                     )
                pfsArm = PfsArm.read(_identity, dirName=_dataDir)
                _pfsDesignId = pfsArm.identity._pfsDesignId
                pfsConfig = PfsConfig.read(pfsDesignId=_pfsDesignId, visit=v, 
                                           dirName=os.path.join(pfsConfigDir, obsdate))
            ''' calculate the typical sky background level '''           
            print("calculating sky background level...")
            pfsArmSky = pfsArm.select(pfsConfig=pfsConfig, targetType=TargetType.SKY)
            data = []
            for wav, flx in zip(pfsArmSky.wavelength, pfsArmSky.flux):
                wc = _skyQaConf["ref_wav_sky"]
                dw = _skyQaConf["ref_dwav_sky"]
                msk = (wav > wc-dw) * (wav < wc+dw)
                data.append(np.nanmedian(flx[msk]))
            df = pd.DataFrame(data={'sky_level': data})
            sky = df['sky_level']
            visit_p_visit.append(v)
            sky_mean_p_visit.append(sky.mean(skipna=True))
            sky_median_p_visit.append(sky.median(skipna=True))
            sky_stddev_p_visit.append(sky.std(skipna=True))

        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'sky_background_mean': sky_mean_p_visit,
                  'sky_background_median': sky_median_p_visit,
                  'sky_background_sigma': sky_stddev_p_visit,
                  'wavelength_ref': [_skyQaConf["ref_wav_sky"] for _ in visit_p_visit]
                  }
            )
        self.qadb.populateQATable('sky', df)

