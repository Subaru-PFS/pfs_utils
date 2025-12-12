#!/usr/bin/env python

import pandas as pd
from pfs.utils.database.opdb import OpDB

__all__ = ["ingestPfsDesign", "ingestPfsConfig"]


def ingestPfsDesign(pfsDesign, designed_at=None):
    """
    Ingest a `PfsDesign` into the operations database (OpDB).

    Parameters
    ----------
    pfsDesign : pfs.datamodel.pfsConfig.PfsDesign
        The design containing targets, guide stars, and metadata to insert.
    designed_at : datetime, optional
        Timestamp when the design was created (or is considered designed).
        If ``None``, the value stored in the database will be ``NULL`` and
        may be populated by downstream processes.

    Returns
    -------
    None

    Notes
    -----
    This function performs side effects only: it inserts rows into the
    following OpDB tables via `OpDB.insert`:

    - ``pfs_design``
    - ``pfs_design_fiber``
    - ``pfs_design_agc``
    """

    db = OpDB()

    # get information from pfsDesign
    numSciDesigned = len(pfsDesign.fiberId[pfsDesign.targetType == 1])
    numCalDesigned = len(pfsDesign.fiberId[pfsDesign.targetType == 3])
    numSkyDesigned = len(pfsDesign.fiberId[pfsDesign.targetType == 2])
    numGuideStars = len(pfsDesign.guideStars)

    # insert into `pfs_design` table
    df = pd.DataFrame(
        {
            "pfs_design_id": [pfsDesign.pfsDesignId],
            "design_name": [pfsDesign.designName],
            "variant": [pfsDesign.variant],
            "design_id0": [pfsDesign.designId0],
            "tile_id": [None],
            "ra_center_designed": [pfsDesign.raBoresight],
            "dec_center_designed": [pfsDesign.decBoresight],
            "pa_designed": [pfsDesign.posAng],
            "num_sci_designed": [numSciDesigned],
            "num_cal_designed": [numCalDesigned],
            "num_sky_designed": [numSkyDesigned],
            "num_guide_stars": [numGuideStars],
            "exptime_tot": [None],
            "exptime_min": [None],
            "ets_version": [None],
            "ets_assigner": [None],
            "designed_at": [designed_at],
            "to_be_observed_at": [pfsDesign.obstime],
            "pfs_utils_version": [pfsDesign.pfsUtilsVer],
            "is_obsolete": [False],
        }
    )

    db.insert_dataframe("pfs_design", df=df)

    # insert into `pfs_design_fiber` table
    df = pd.DataFrame(
        {
            "pfs_design_id": [pfsDesign.pfsDesignId for _ in pfsDesign.fiberId],
            "fiber_id": pfsDesign.fiberId,
            "target_cat_id": pfsDesign.catId,
            "target_tract": pfsDesign.tract,
            "target_patch": pfsDesign.patch,
            "target_obj_id": pfsDesign.objId,
            "target_ra": pfsDesign.ra,
            "target_dec": pfsDesign.dec,
            "target_type": pfsDesign.targetType,
            "fiber_status": pfsDesign.fiberStatus,
            "pfi_nominal_x_mm": pfsDesign.pfiNominal[:, 0],
            "pfi_nominal_y_mm": pfsDesign.pfiNominal[:, 1],
            "target_pm_ra": pfsDesign.pmRa,
            "target_pm_dec": pfsDesign.pmDec,
            "target_parallax": pfsDesign.parallax,
            "epoch": pfsDesign.epoch,
            "proposal_id": pfsDesign.proposalId,
            "ob_code": pfsDesign.obCode,
            "is_on_source": [True for _ in pfsDesign.fiberId],
        }
    )
    db.insert_dataframe("pfs_design_fiber", df=df)

    # insert into `pfs_design_agc` table
    guideStars = pfsDesign.guideStars

    df = pd.DataFrame(
        {
            "pfs_design_id": [pfsDesign.pfsDesignId for _ in guideStars.objId],
            "guide_star_id": guideStars.objId,
            "epoch": guideStars.epoch,
            "guide_star_ra": guideStars.ra,
            "guide_star_dec": guideStars.dec,
            "guide_star_pm_ra": guideStars.pmRa,
            "guide_star_pm_dec": guideStars.pmDec,
            "guide_star_parallax": guideStars.parallax,
            "guide_star_magnitude": guideStars.magnitude,
            "passband": guideStars.passband,
            "guide_star_color": guideStars.color,
            "agc_camera_id": guideStars.agId,
            "agc_target_x_pix": guideStars.agX,
            "agc_target_y_pix": guideStars.agY,
            "guide_star_flag": guideStars.flag,
        }
    )
    db.insert_dataframe("pfs_design_agc", df=df)


def ingestPfsConfig(
    pfsConfig, allocated_at=None, converg_num_iter=None, converg_elapsed_time=None, converg_tolerance=None
):
    """
    Ingest a `PfsConfig` into the operations database (OpDB).

    Parameters
    ----------
    pfsConfig : pfs.datamodel.pfsConfig.PfsConfig
        The configuration produced from a `PfsDesign`, including fiber
        allocations, final positions, and guide star results.
    allocated_at : datetime, optional
        Timestamp when the configuration was allocated. If ``None``, stored
        as ``NULL``.
    converg_num_iter : int, optional
        Number of iterations used by the convergence/assignment procedure.
    converg_elapsed_time : float, optional
        Elapsed time in seconds for the convergence/assignment procedure.
    converg_tolerance : float, optional
        Convergence tolerance used by the assignment procedure.

    Returns
    -------
    None

    Notes
    -----
    This function performs side effects only: it inserts rows into the
    following OpDB tables via `OpDB.insert`:

    - ``pfs_config``
    - ``pfs_config_fiber``
    - ``pfs_config_agc``
    """
    db = OpDB()

    # get information from pfsConfig

    # insert into `pfs_config` table
    df = pd.DataFrame(
        {
            "pfs_design_id": [pfsConfig.pfsDesignId],
            "visit0": [pfsConfig.visit],
            "ra_center_config": [pfsConfig.raBoresight],
            "dec_center_config": [pfsConfig.decBoresight],
            "pa_config": [pfsConfig.posAng],
            "converg_num_iter": [converg_num_iter],
            "converg_elapsed_time": [converg_elapsed_time],
            "converg_tolerance": [converg_tolerance],
            "alloc_rms_scatter": [None],
            "allocated_at": [allocated_at],
            "to_be_observed_at": [pfsConfig.obstime],
            "pfs_utils_version": [pfsConfig.pfsUtilsVer],
            "to_be_observed_at_design": [pfsConfig.obstimeDesign],
            "pfs_utils_version_design": [pfsConfig.pfsUtilsVerDesign],
            "was_observed": [False],
        }
    )
    db.insert_dataframe("pfs_config", df=df)

    # insert into `pfs_config_fiber` table
    df = pd.DataFrame(
        {
            "pfs_design_id": [pfsConfig.pfsDesignId for _ in pfsConfig.fiberId],
            "visit0": [pfsConfig.visit for _ in pfsConfig.fiberId],
            "fiber_id": pfsConfig.fiberId,
            "target_ra": pfsConfig.ra,
            "target_dec": pfsConfig.dec,
            "pfi_nominal_x_mm": pfsConfig.pfiNominal[:, 0],
            "pfi_nominal_y_mm": pfsConfig.pfiNominal[:, 1],
            "pfi_center_final_x_mm": pfsConfig.pfiCenter[:, 0],
            "pfi_center_final_y_mm": pfsConfig.pfiCenter[:, 1],
            "fiber_status": pfsConfig.fiberStatus,
            "motor_map_summary": [None for _ in pfsConfig.fiberId],
            "config_elapsed_time": [None for _ in pfsConfig.fiberId],
            "is_on_source": [True for _ in pfsConfig.fiberId],
        }
    )
    db.insert_dataframe("pfs_config_fiber", df=df)

    # insert into `pfs_config_agc` table
    guideStars = pfsConfig.guideStars
    df = pd.DataFrame(
        {
            "pfs_design_id": [pfsConfig.pfsDesignId for _ in guideStars.objId],
            "visit0": [pfsConfig.visit for _ in guideStars.objId],
            "guide_star_id": guideStars.objId,
            "agc_camera_id": guideStars.agId,
            "agc_final_x_pix": guideStars.agX,
            "agc_final_y_pix": guideStars.agY,
            "comments": [None for _ in guideStars.objId],
            "guide_star_ra": guideStars.ra,
            "guide_star_dec": guideStars.dec,
        }
    )
    db.insert_dataframe("pfs_config_agc", df=df)
