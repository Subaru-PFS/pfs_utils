import os

import pfs.utils.butler as pfsButler

__all__ = ["writePfsConfig"]


def writePfsConfig(pfsConfig):
    """Write pfsConfig file to /data/raw/$DATE/pfsConfig using pfsButler."""
    # Get path from pfsButler.
    path = pfsButler.Butler().getPath('pfsConfig', pfsConfigId=pfsConfig.pfsDesignId, visit=pfsConfig.visit)
    # Create date/pfsConfig directory if it does not exist.
    rootDir, fileName = os.path.split(path)
    if not os.path.exists(rootDir):
        os.makedirs(rootDir)
    # Write pfsConfig file to disk.
    pfsConfig.write(fileName=path)
