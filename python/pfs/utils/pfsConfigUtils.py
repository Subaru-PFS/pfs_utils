import glob
import os

import pfs.utils.butler as pfsButler
from pfs.datamodel import PfsDesign, PfsConfig

__all__ = ["getDateDir", "writePfsConfig", "writePfsConfigFromDesign"]


def getDateDir(pfsConfig):
    """Definitely not the quickest but I have not better idea at this moment."""
    [pfsConfigPath] = glob.glob('/data/raw/*-*-*/pfsConfig/%s' % pfsConfig.filename)

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
        os.makedirs(rootDir, mode=0o775)
        os.chmod(dateDir, 0o775)
    # Write pfsConfig file to disk and set correct permissions.
    pfsConfig.write(fileName=filepath)
    os.chmod(filepath, 0o664)


def writePfsConfigFromDesign(visit, pfsDesignId, dirName):
    """Write fake pfsConfig given a visit and pfsDesignId."""
    # Reading pfsDesign file.
    pfsDesign = PfsDesign.read(pfsDesignId, dirName=dirName)
    # Creating a fake pfsConfig file from pfsDesign using pfsDesign.pfiNominal for pfiCenter.
    pfsConfig = PfsConfig.fromPfsDesign(pfsDesign, visit, pfsDesign.pfiNominal)
    # Write pfsConfig file to disk.
    writePfsConfig(pfsConfig)
    # return pfsConfig file.
    return pfsConfig
