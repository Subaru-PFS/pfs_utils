import numpy as np
from pfs.datamodel import PfsDesign, FiberStatus, TargetType
from pfs.datamodel.utils import calculate_pfsDesignId

from pfs.utils.fiberids import FiberIds


def makePfsDesignFile(x, y,
                      raBoresight=100, decBoresight=100, posAng=0, arms='br',
                      tract=1, patch='1,1', ra=None, dec=None, catId=-1, objId=-1, targetType=TargetType.SCIENCE,
                      fiberStatus=FiberStatus.GOOD,
                      fiberFlux=np.NaN, psfFlux=np.NaN, totalFlux=np.NaN,
                      fiberFluxErr=np.NaN, psfFluxErr=np.NaN, totalFluxErr=np.NaN,
                      filterNames=None, guideStars=None):
    """ Make PfsDesign object from cobra x and y required positions.

        Parameters
        ----------
        x : array_like
            cobra x position (1..2394).
        y: array_like
            cobra y position (1..2394).

        Returns
        -------
        pfsDesign : pfs.datamodel.PfsDesign
           constructed pfsDesign for all fibers (science + engineering)
    """

    # Grand Fiber Map
    gfm = FiberIds()
    isEmpty = gfm.scienceFiberId == FiberIds.EMPTY
    isEng = gfm.scienceFiberId == FiberIds.ENGINEERING
    isCobra = ~isEmpty & ~isEng

    nFiber = len(gfm.scienceFiberId)
    nScienceFiber =  len(gfm.scienceFiberId[isCobra])
    fiberId = gfm.fiberId

    def setDefaultValues(sciVal, engVal, shape=nFiber, dtype=float):
        """assign provided sci values to cobra index and default value to engineering fibers. """
        array = np.empty(shape, dtype=dtype)

        if not isinstance(sciVal, (float, str)):
            sciVal = np.array(sciVal)
            if sciVal.ndim == 1 and array.ndim == 2:
                sciVal = sciVal[:, None]

        array[isCobra] = sciVal
        array[isEng] = engVal
        # we also need to fill empty fibers with some values
        array[isEmpty] = engVal

        return array

    tract = setDefaultValues(sciVal=tract, engVal=-1, dtype='int32')
    patch = setDefaultValues(sciVal=patch, engVal='0,0', dtype='<U32')

    if ra is None:
        ra = raBoresight + 1e-3 * x

    if dec is None:
        dec = decBoresight + 1e-3 * y

    ra = setDefaultValues(sciVal=ra, engVal=100)
    dec = setDefaultValues(sciVal=dec, engVal=100)

    catId = setDefaultValues(sciVal=catId, engVal=-1, dtype='int32')
    objId = setDefaultValues(sciVal=objId, engVal=-1, dtype='int64')

    targetType = setDefaultValues(sciVal=targetType, engVal=TargetType.ENGINEERING, dtype=int)
    fiberStatus = setDefaultValues(sciVal=fiberStatus, engVal=FiberStatus.GOOD)

    # I might be overaccommodating here but ...
    if filterNames is None:
        # making sure some input data are not discarded silently
        assert fiberFlux==np.NaN and psfFlux==np.NaN and totalFlux==np.NaN
        filterNameArray = np.array(nScienceFiber * [[]], dtype='str')  # list of filter names

    elif isinstance(filterNames, str):
        filterNameArray = np.array(nScienceFiber * [[filterNames]])

    else:
        filterNameArray = np.array(filterNames)
        filterNameArray = filterNameArray[:, None] if filterNameArray.ndim == 1 else filterNameArray

    nFilter = filterNameArray.shape[1]
    filterNameArray = setDefaultValues(sciVal=filterNameArray, engVal='', dtype=filterNameArray.dtype,
                                       shape=(nFiber, nFilter))

    fiberFlux = setDefaultValues(sciVal=fiberFlux, engVal=np.NaN, shape=(nFiber, nFilter))
    psfFlux = setDefaultValues(sciVal=psfFlux, engVal=np.NaN, shape=(nFiber, nFilter))
    totalFlux = setDefaultValues(sciVal=totalFlux, engVal=np.NaN, shape=(nFiber, nFilter))

    fiberFluxErr = setDefaultValues(sciVal=fiberFluxErr, engVal=np.NaN, shape=(nFiber, nFilter))
    psfFluxErr = setDefaultValues(sciVal=psfFluxErr, engVal=np.NaN, shape=(nFiber, nFilter))
    totalFluxErr = setDefaultValues(sciVal=totalFluxErr, engVal=np.NaN, shape=(nFiber, nFilter))

    x = setDefaultValues(sciVal=x, engVal=np.NaN)
    y = setDefaultValues(sciVal=y, engVal=np.NaN)
    pfiNominal = np.stack((x, y)).T

    pfsDesign = PfsDesign(0x0, raBoresight, decBoresight, posAng, arms, fiberId, tract, patch, ra, dec, catId, objId,
                          targetType, fiberStatus, fiberFlux, psfFlux, totalFlux, fiberFluxErr, psfFluxErr,
                          totalFluxErr, filterNameArray, pfiNominal, guideStars)
    # Drop empty fibers
    pfsDesign = pfsDesign[~isEmpty]
    pfsDesign.pfsDesignId = calculate_pfsDesignId(pfsDesign.fiberId, pfsDesign.ra, pfsDesign.dec)

    return pfsDesign
