import os

import pfs.utils.butler as pfsButler
from pfs.datamodel import PfsDesign, PfsConfig

__all__ = ["writePfsConfig", "writePfsConfigFromDesign"]


def writePfsConfig(pfsConfig):
    """Write pfsConfig file to /data/raw/$DATE/pfsConfig using pfsButler."""
    # Get path from pfsButler.
    path = pfsButler.Butler().getPath('pfsConfig', pfsConfigId=pfsConfig.pfsDesignId, visit=pfsConfig.visit)
    # Create date/pfsConfig directory if it does not exist.
    rootDir, fileName = os.path.split(path)
    if not os.path.exists(rootDir):
        dateDir, _ = os.path.split(rootDir)
        # we currently have weird permissions on /data so fix it manually for now.
        os.makedirs(rootDir, mode=0o775)
        os.chmod(dateDir, 0o775)
    # Write pfsConfig file to disk.
    pfsConfig.write(fileName=path)


def writePfsConfigFromDesign(visit, pfsDesignId, dirName):
    """Write fake pfsConfig given a visit and pfsDesignId."""
    # Reading pfsDesign file.
    pfsDesign = PfsDesign.read(pfsDesignId, dirName=dirName)
    # Creating a fake pfsConfig file from pfsDesign nominal
    pfsConfig = PfsConfig.fromPfsDesign(pfsDesign, visit, pfsDesign.pfiNominal)
    # Write pfsConfig file to disk.
    writePfsConfig(pfsConfig)
