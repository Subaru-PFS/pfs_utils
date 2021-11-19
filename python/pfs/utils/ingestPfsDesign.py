#!/usr/bin/env python

import pandas as pd
from opdb import opdb

__all__ = ["ingestPfsDesign", "ingestPfsConfig"]

hostname = 'db-ics'
port = 5432
dbname = 'opdb'
username = 'pfs'


def ingestPfsDesign(pfsDesign, designed_at=None, to_be_observed_at=None):
    '''
        Description
        -----------
            Ingest pfsDesign to opDB

        Parameters
        ----------
            pfsDesign : `pfs.datamodel.pfsConfig.PfsDesign`
            designed_at : `datetime`
            to_be_observed_at : `datetime`

        Returns
        -------
            None

    '''
    db = opdb.OpDB(hostname=hostname,
                   port=port,
                   dbname=dbname,
                   username=username,
                   dialect='postgresql'
                   )

    ''' get information from pfsDesign '''
    numSciDesigned = len(pfsDesign.fiberId[pfsDesign.targetType == 1])
    numCalDesigned = len(pfsDesign.fiberId[pfsDesign.targetType == 3])
    numSkyDesigned = len(pfsDesign.fiberId[pfsDesign.targetType == 2])
    numGuideStars = len(pfsDesign.guideStars)

    ''' insert into `pfs_design` table '''
    df = pd.DataFrame({'pfs_design_id': [pfsDesign.pfsDesignId],
                       'design_name': [pfsDesign.designName],
                       'tile_id': [None],
                       'ra_center_designed': [pfsDesign.raBoresight],
                       'dec_center_designed': [pfsDesign.decBoresight],
                       'pa_designed': [pfsDesign.posAng],
                       'num_sci_designed': [numSciDesigned],
                       'num_cal_designed': [numCalDesigned],
                       'num_sky_designed': [numSkyDesigned],
                       'num_guide_stars': [numGuideStars],
                       'exptime_tot': [None],
                       'exptime_min': [None],
                       'ets_version': [None],
                       'ets_assigner': [None],
                       'designed_at': [designed_at],
                       'to_be_observed_at': [to_be_observed_at],
                       'is_obsolete': [False]
                       })
    #db.bulkInsert('pfs_design', df)
    db.insert('pfs_design', df)

    ''' insert into `pfs_design_fiber` table '''
    df = pd.DataFrame({'pfs_design_id': [pfsDesign.pfsDesignId for _ in pfsDesign.fiberId],
                       'fiber_id': pfsDesign.fiberId,
                       'target_cat_id': pfsDesign.catId,
                       'target_tract': pfsDesign.tract,
                       'target_patch': pfsDesign.patch,
                       'target_obj_id': pfsDesign.objId,
                       'target_ra': pfsDesign.ra,
                       'target_dec': pfsDesign.dec,
                       'target_type': pfsDesign.targetType,
                       'fiber_status': pfsDesign.fiberStatus,
                       'pfi_nominal_x_mm': pfsDesign.pfiNominal[:, 0],
                       'pfi_nominal_y_mm': pfsDesign.pfiNominal[:, 1],
                       'is_on_source': [True for _ in pfsDesign.fiberId]
                       })
    db.insert('pfs_design_fiber', df)

    ''' insert into `pfs_design_agc` table '''
    guideStars = pfsDesign.guideStars
    df = pd.DataFrame({'pfs_design_id': [pfsDesign.pfsDesignId for _ in guideStars.objId],
                       'guide_star_id': guideStars.objId,
                       'epoch': guideStars.epoch,
                       'guide_star_ra': guideStars.ra,
                       'guide_star_dec': guideStars.dec,
                       'guide_star_pm_ra': guideStars.pmRa,
                       'guide_star_pm_dec': guideStars.pmDec,
                       'guide_star_parallax': guideStars.parallax,
                       'guide_star_magnitude': guideStars.magnitude,
                       'passband': guideStars.passband,
                       'guide_star_color': guideStars.color,
                       'agc_camera_id': guideStars.agId,
                       'agc_target_x_pix': guideStars.agX,
                       'agc_target_y_pix': guideStars.agY,
                       })
    db.insert('pfs_design_agc', df)

    ''' close the DB connection '''
    db.close()


def ingestPfsConfig(pfsConfig, allocated_at=None):
    '''
        Description
        -----------
            Ingest pfsConfig to opDB

        Parameters
        ----------
            pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            allocated_at : `datetime`

        Returns
        -------
            None

    '''
    db = opdb.OpDB(hostname=hostname,
                   port=port,
                   dbname=dbname,
                   username=username,
                   dialect='postgresql'
                   )

    ''' get information from pfsConfig '''

    ''' insert into `pfs_config` table '''
    df = pd.DataFrame({'pfs_design_id': [pfsConfig.pfsDesignId],
                       'visit0': [pfsConfig.visit0],
                       'ra_center_config': [pfsConfig.raBoresight],
                       'dec_center_config': [pfsConfig.decBoresight],
                       'pa_config': [pfsConfig.posAng],
                       'converg_num_iter': [None],
                       'converg_elapsed_time': [None],
                       'alloc_rms_scatter': [None],
                       'allocated_at': [allocated_at],
                       'was_observed': [False]
                       })
    db.insert('pfs_config', df)

    ''' insert into `pfs_config_fiber` table '''
    df = pd.DataFrame({'pfs_design_id': [pfsConfig.pfsDesignId for _ in pfsConfig.fiberId],
                       'visit0': [pfsConfig.visit0 for _ in pfsConfig.fiberId],
                       'fiber_id': pfsConfig.fiberId,
                       'pfi_center_final_x_mm': pfsConfig.pfiCenter[:, 0],
                       'pfi_center_final_x_mm': pfsConfig.pfiCenter[:, 1],
                       'motor_map_summary': [None for _ in pfsConfig.fiberId],
                       'config_elapsed_time': [None for _ in pfsConfig.fiberId],
                       'is_on_source': [True for _ in pfsConfig.fiberId]
                       })
    db.insert('pfs_config_fiber', df)

    ''' insert into `pfs_config_agc` table '''
    guideStars = pfsConfig.guideStars
    df = pd.DataFrame({'pfs_design_id': [pfsConfig.pfsDesignId for _ in guideStars.objId],
                       'visit0': [pfsConfig.visit0 for _ in guideStars.objId],
                       'guide_star_id': guideStars.objId,
                       'agc_camera_id': guideStars.agId,
                       'agc_final_x_pix': guideStars.agX,
                       'agc_final_y_pix': guideStars.agY,
                       'comments': [None for _ in guideStars.objId],
                       })
    db.insert('pfs_config_agc', df)

    ''' close the DB connection '''
    db.close()
