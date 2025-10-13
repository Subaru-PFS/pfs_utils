import datetime
import glob
import os
from datetime import timezone

import numpy as np
import pfs.utils.butler as pfsButler
import pfs.utils.coordinates.updateTargetPosition as updateTargetPosition
from future.backports.datetime import timezone
from pfs.datamodel.utils import convertToIso8601Utc
from pfs.utils.coordinates.CoordTransp import ag_pfimm_to_pixel
from pfs.utils.versions import getVersion

__all__ = ["getDateDir", "writePfsConfig", "tweakTargetPosition"]


def getDateDir(pfsConfig):
    """Definitely not the quickest but I have not better idea at this moment."""
    pfsConfigPath = ''
    year = datetime.date.today().year

    for year in reversed(range(year - 1, year + 2)):
        search = glob.glob(f'/data/raw/{year}-*-*/pfsConfig/{pfsConfig.filename}')
        if search:
            [pfsConfigPath] = search
            break

    if not pfsConfigPath:
        raise ValueError(f'could not find {pfsConfig.filename} in /data/raw/$DATE/pfsConfig')

    dirName, _ = os.path.split(pfsConfigPath)
    rootDir, _ = os.path.split(dirName)
    _, dateDir = os.path.split(rootDir)

    return dateDir


def writePfsConfig(pfsConfig):
    """Write pfsConfig file to /data/raw/$DATE/pfsConfig using pfsButler."""
    # Get path from pfsButler.
    filepath = pfsButler.Butler().getPath('pfsConfig', pfsConfigId=pfsConfig.pfsDesignId, visit=pfsConfig.visit)
    # Create date/pfsConfig directory if it does not exist.
    rootDir, fileName = os.path.split(filepath)
    if not os.path.exists(rootDir):
        dateDir, _ = os.path.split(rootDir)
        # we currently have weird permissions on /data so fix it manually for now.
        os.makedirs(rootDir, mode=0o2775)
        # ccdActor create the date directory as pfs-data, so I don't have the permission in that case.
        try:
            os.chmod(dateDir, 0o2775)
        except PermissionError:
            pass
    # Write pfsConfig file to disk and set correct permissions.
    pfsConfig.write(fileName=filepath)
    os.chmod(filepath, 0o444)


def tweakTargetPosition(pfsConfig, obstime='now'):
    """Update cobra-target and guide-star positions at observation time.

    Parameters
    ----------
    pfsConfig : pfs.datamodel.PfsConfig
    obstime : str or datetime or 'now', optional
        Observation time. If 'now', uses current UTC. Strings are converted via datetime.isoformat().

    Returns
    -------
    pfsConfig : pfs.datamodel.PfsConfig
        The same pfsConfig, updated in place with new RA/Dec, PFI nominal positions and guide-star positions.

    """

    def getUpdatedRaDecAndPosition(targets, pa, cent, obstime, mode):
        """Compute updated RA/Dec and focal-plane positions for pfsConfig targets and guideStars."""
        radec = np.vstack([targets.ra, targets.dec])
        # getting pm and par from design
        pm = np.vstack([targets.pmRa, targets.pmDec])
        par = targets.parallax

        return updateTargetPosition.update_target_position(radec, pa, cent, pm, par, obstime, mode=mode)

    cent = np.vstack([pfsConfig.raBoresight, pfsConfig.decBoresight])
    pa = pfsConfig.posAng

    obstime = datetime.datetime.now(timezone.utc) if obstime == 'now' else obstime

    if obstime.tzinfo is None or obstime.tzinfo.utcoffset(obstime) is None:
        raise ValueError("obstime must be timezone-aware (localized) or 'now'")

    # converting to ISO-8601
    obstime = convertToIso8601Utc(obstime.isoformat())

    # updating ra/dec/position for cobra targets.
    ra_now, dec_now, pfi_x_now, pfi_y_now = getUpdatedRaDecAndPosition(pfsConfig, pa, cent, obstime, mode='sky_pfi')
    # updating ra/dec/position for guideStars objects.
    guide_ra_now, guide_dec_now, guide_x_now, guide_y_now = getUpdatedRaDecAndPosition(pfsConfig.guideStars, pa, cent,
                                                                                       obstime, mode='sky_pfi_ag')
    # converting to ag pixels
    guide_xy_pix = np.array([ag_pfimm_to_pixel(agId, x, y)
                             for agId, x, y in zip(pfsConfig.guideStars.agId, guide_x_now, guide_y_now)])
    guide_x_pix = guide_xy_pix[:, 0].astype('float32')
    guide_y_pix = guide_xy_pix[:, 1].astype('float32')

    # Get pfs_utils version.
    pfsUtilsVer = getVersion('pfs_utils')
    # setting the new positions.
    pfsConfig.updateTargetPosition(ra=ra_now, dec=dec_now,
                                   pfiNominal=np.column_stack((pfi_x_now, pfi_y_now)),
                                   obstime=obstime, pfsUtilsVer=pfsUtilsVer,
                                   guide_ra=guide_ra_now, guide_dec=guide_dec_now,
                                   guide_x_pix=guide_x_pix, guide_y_pix=guide_y_pix)

    return pfsConfig
