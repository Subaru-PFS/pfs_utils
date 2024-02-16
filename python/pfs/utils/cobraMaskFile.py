import os

import numpy as np
import pandas as pd
from pfs.utils.butler import Butler as Nestor
from pfs.utils.fiberids import FiberIds

nestor = Nestor()

dots = nestor.get('black_dots')
calibModel = nestor.get('moduleXml', moduleName='ALL', version='')

gfm = pd.DataFrame(FiberIds().data)
sgfm = gfm.set_index('scienceFiberId').loc[np.arange(2394) + 1].reset_index().sort_values('cobraId')

# adding mtp groupId.
sgfm['mtpGroupId'] = np.array(list(map(int, [mtpA.split('-')[-2] for mtpA in sgfm.mtp_A])), dtype='int32')

# getting up-to-date cobras calibration.
xCob = np.array(calibModel.centers.real).astype('float32')
yCob = np.array(calibModel.centers.imag).astype('float32')
armLength = np.array(calibModel.L1 + calibModel.L2).astype('float32')
FIBER_BROKEN_MASK = (calibModel.status & calibModel.FIBER_BROKEN_MASK).astype('bool')
COBRA_OK_MASK = (calibModel.status & calibModel.COBRA_OK_MASK).astype('bool')

sgfm['x'] = xCob
sgfm['y'] = yCob
sgfm['FIBER_BROKEN_MASK'] = FIBER_BROKEN_MASK
sgfm['COBRA_OK_MASK'] = COBRA_OK_MASK
sgfm['armLength'] = np.array(calibModel.L1 + calibModel.L2).astype('float32')

# adding blackSpots position and radius.
np.testing.assert_equal(sgfm.cobraId.to_numpy(), dots.spotId.to_numpy())
sgfm['xDot'] = dots.x.to_numpy()
sgfm['yDot'] = dots.y.to_numpy()
sgfm['rDot'] = dots.r.to_numpy()

sgfm = sgfm[['scienceFiberId', 'cobraId', 'fiberId', 'spectrographId', 'mtpGroupId', 'FIBER_BROKEN_MASK',
             'COBRA_OK_MASK', 'x', 'y', 'xDot', 'yDot', 'rDot', 'armLength']]

__all__ = ("saveCobraMaskFile", "buildCobraMaskFile", "buildMODxMaskFile",)


def saveCobraMaskFile(maskFile, fileName, outputDir='.'):
    """Save cobra maskFile to output directory."""
    path = os.path.join(outputDir, f'{fileName}.csv')
    maskFile.to_csv(path)
    print(f"maskFile written to {path}")


def buildCobraMaskFile(subset, fileName, doSave=False, outputDir='.'):
    """Build and save cobra maskFiles"""
    brokenCobras = list(sgfm[~sgfm.COBRA_OK_MASK].fiberId)
    # Removing broken cobras from selected subset.
    movingFiberId = list(set(subset.fiberId.to_numpy()) - set(brokenCobras))
    # Making maskFile.
    cobraMask = makeCobraMoveMask(movingFiberId)

    if doSave:
        saveCobraMaskFile(cobraMask, fileName, outputDir=outputDir)

    return cobraMask


def buildMODxMaskFile(modNumber, doSave=False, outputDir='.'):
    """Build and save Modx maskFiles."""
    maskFiles = []
    # build evenly spaced subset.
    modDataset = makeModxDataset(modNumber)

    for groupId, subset in modDataset.groupby('subset'):
        fileName = f'MOD{modNumber}_group{groupId + 1}'
        # making maskFile but not saving right away.
        maskFile = buildCobraMaskFile(subset, fileName=fileName, doSave=False)

        # doing sanity check on the maskFile.
        moving = maskFile[maskFile.bitMask == 1].sort_values('fiberId')
        diff = np.diff(moving.fiberId)
        assert (len(diff[diff < modNumber]) == 0)

        # test have passed I can save safely.
        if doSave:
            saveCobraMaskFile(maskFile, fileName, outputDir=outputDir)

        maskFiles.append(maskFile)

    return maskFiles


def makeCobraMoveMask(fiberId):
    """Return cobra motion maskfile (0:ignore 1:doMove)."""
    subset = sgfm[['cobraId', 'fiberId', 'scienceFiberId', 'spectrographId']].sort_values('fiberId').copy()
    # finding corresponding indice.
    iFib = sgfm.set_index('fiberId').loc[fiberId].scienceFiberId.to_numpy() - 1
    # making the array.
    bitMask = np.zeros(len(subset), dtype='int32')
    bitMask[iFib] = 1
    # setting bitMask.
    subset['bitMask'] = bitMask
    # cobraId oriented, so sorting by cobraId
    return subset.sort_values('cobraId').reset_index(drop=True)


def selectScienceFiberId(scienceFiberId):
    """return sgfm subset given scienceFiberId."""
    return sgfm.set_index('scienceFiberId').loc[scienceFiberId].reset_index()


def makeModxDataset(nSpace):
    """Separate scienceFiberId in evenly spaced subset."""
    base = np.arange(0, 2394, nSpace)

    subsets = []

    for i in range(nSpace):
        sci = base + i + 1
        sci = sci[sci <= len(sgfm)]
        subset = selectScienceFiberId(sci)
        subset['subset'] = i
        subsets.append(subset)

    dfs = pd.concat(subsets).reset_index(drop=True)
    assert len(dfs) == len(sgfm)

    return dfs
