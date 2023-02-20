import datetime
import glob
import os

import pfs.utils.butler as pfsButler

__all__ = ["getDateDir", "writePfsConfig"]


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
