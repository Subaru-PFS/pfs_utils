import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from pfs.datamodel import PfsDesign, FiberStatus, TargetType
from pfs.datamodel.utils import calculate_pfsDesignId
from pfs.utils.versions import getVersion

__all__ = ["makePfsDesign", "showPfsDesign", "fakeRaDecFromPfiNominal"]

subaru = None  # we'll look it up if we need it
utcoffset = -10  # UTC -> HST.  No daylight saving to worry about
fakeRa, fakeDec = 100, -89.

from pfs.utils.butler import Butler as Nestor
from pfs.utils.fiberids import FiberIds

def setFiberStatus(pfsDesign, calibModel=None, configRoot=None, fiberIdsPath=None):
    """
    Set the fiber status for the PFS design based on cobra calibration model.

    Parameters
    ----------
    pfsDesign : PfsDesign
        The PFS design object.
    calibModel : CalibModel, optional
        The cobra calibration model. If None, the latest version of the moduleXml is retrieved.
    configRoot : path-like, optional
        The root of the configuration directory tree.
        Defaults to None (will be set to $PFS_INSTDATA_DIR/data by the Nestor (Butler) class)
    fiberIdsPath : path-like, optional
        The path to the fiberids data.
        Defaults to None (will be set to $PFS_UTILS_DIR/data/fiberids).

    Returns
    -------
    PfsDesign
        The updated PFS design object with the fiber status set.
    """

    def loadCobraMaskFromXml(calibModel):
        """
        Load the cobra masks from the cobra XML file.

        Parameters
        ----------
        calibModel : CalibModel, optional
            The calibration model object. If None, the latest version of the moduleXml
            is retrieved using Nestor.

        Returns
        -------
        tuple
            A tuple containing the FIBER_BROKEN_MASK and COBRA_BROKEN_MASK.
        """
        calibModel = nestor.get('moduleXml', moduleName='ALL', version='') if calibModel is None else calibModel

        FIBER_BROKEN_MASK = (calibModel.status & calibModel.FIBER_BROKEN_MASK).astype('bool')
        COBRA_OK_MASK = (calibModel.status & calibModel.COBRA_OK_MASK).astype('bool')
        COBRA_BROKEN_MASK = np.logical_and(~COBRA_OK_MASK, ~FIBER_BROKEN_MASK)

        return FIBER_BROKEN_MASK, COBRA_BROKEN_MASK

    nestor = Nestor(configRoot=configRoot)

    # first setting BROKENFIBER and BROKENCOBRA fiberStatus.
    FIBER_BROKEN_MASK, COBRA_BROKEN_MASK = loadCobraMaskFromXml(calibModel=calibModel)

    engFiberMask = pfsDesign.targetType == TargetType.ENGINEERING
    fiberId = pfsDesign.fiberId[~engFiberMask]
    cobraId = FiberIds(path=fiberIdsPath).fiberIdToCobraId(fiberId)

    # resetting fiberStatus to GOOD.
    fiberStatus = pfsDesign.fiberStatus[~engFiberMask].copy()
    fiberStatus[:] = FiberStatus.GOOD

    fiberStatus[FIBER_BROKEN_MASK[cobraId - 1]] = FiberStatus.BROKENFIBER
    fiberStatus[COBRA_BROKEN_MASK[cobraId - 1]] = FiberStatus.BROKENCOBRA
    pfsDesign.fiberStatus[~engFiberMask] = fiberStatus

    # setting BAD_PSF fiberStatus.
    fiberBadPsf = nestor.get('fiberBadPsf')
    pfsDesign.fiberStatus[np.isin(pfsDesign.fiberId, fiberBadPsf['fiberId'])] = FiberStatus.BAD_PSF

    # then setting BLOCKED fiberStatus.
    fiberBlocked = nestor.get('fiberBlocked').set_index('fiberId')
    pfsDesign.fiberStatus[fiberBlocked.loc[pfsDesign.fiberId].status.to_numpy()] = FiberStatus.BLOCKED

    return pfsDesign


def fakeRaDecFromPfiNominal(pfiNominal):
    """
    This function generates fake right ascension (RA) and declination (Dec) values
    from the given cobra positions (x and y) in the PFI coordinate system.

    Parameters
    ----------
    pfiNominal : numpy.ndarray of float, shape: (nFiber, 2)
        Intended target cobra position (2-vector) of each fiber on the PFI, in millimeters.

    Returns
    -------
    ra : numpy.ndarray of float, shape: (nFiber),
        Fake right ascension values generated from the input cobra positions.
    dec : numpy.ndarray of float, shape: (nFiber),
        Fake declination values generated from the input cobra positions.
    """
    ra = fakeRa + pfiNominal[:, 0].astype('float64') / 250
    dec = fakeDec + pfiNominal[:, 1].astype('float64') / 250
    return ra, dec


def makePfsDesign(pfiNominal, ra, dec,
                  raBoresight=fakeRa, decBoresight=fakeDec, posAng=0, arms='br',
                  tract=1, patch='1,1', catId=-1, objId=-1, targetType=TargetType.SCIENCE,
                  fiberStatus=FiberStatus.GOOD,
                  epoch="J2000.0", pmRa=0.0, pmDec=0.0, parallax=1e-8,
                  proposalId="N/A", obCode="N/A",
                  fiberFlux=np.nan, psfFlux=np.nan, totalFlux=np.nan,
                  fiberFluxErr=np.nan, psfFluxErr=np.nan, totalFluxErr=np.nan,
                  filterNames=None, guideStars=None, designName=None, fiberidsPath=None, obstime=""):
    """ Make PfsDesign object from cobra x and y required positions.

    Parameters
    ----------
    pfiNominal : `numpy.ndarray` of `float` shape: (nFiber, 2)
        Intended target cobra position (2-vector) of each fiber on the PFI, millimeters.
    ra : `numpy.ndarray` of `float64`
        Right Ascension for each fiber, degrees.
    dec : `numpy.ndarray` of `float64`
        Declination for each fiber, degrees.
    raBoresight : `float`, degrees
        Right Ascension of telescope boresight.
    decBoresight : `float`, degrees
        Declination of telescope boresight.
    posAng : `float`, degrees
        The position angle from the
        North Celestial Pole to the PFI_Y axis,
        measured clockwise with respect to the
        positive PFI_Z axis
    arms : `str`
        arms to expose. Eg 'brn', 'bmn'.
    fiberId : `numpy.ndarary` of `int32`
        Fiber identifier for each fiber.
    tract : `numpy.ndarray` of `int32`
        Tract index for each fiber.
    patch : `numpy.ndarray` of `str`
        Patch indices for each fiber, typically two integers separated by a
        comma, e.g,. "5,6".
    catId : `numpy.ndarray` of `int32`
        Catalog identifier for each fiber.
    objId : `numpy.ndarray` of `int64`
        Object identifier for each fiber. Specifies the object within the
        catalog.
    targetType : `numpy.ndarray` of `int`
        Type of target for each fiber. Values must be convertible to
        `TargetType` (which limits the range of values).
    fiberStatus : `numpy.ndarray` of `int`
        Status of each fiber. Values must be convertible to `FiberStatus`
        (which limits the range of values).
    epoch : `numpy.chararray`
        reference epoch for each fiber.
    pmRa : `numpy.ndarray` of `float32`
        Proper motion in direction of Right Ascension
        for each fiber, mas/year.
    pmDec : `numpy.ndarray` of `float32`
        Proper motion in direction of Declination
        for each fiber, mas/year.
    parallax : `numpy.ndarray` of `float32`
        parallax for each fiber, mas.
    proposalId : `numpy.chararray`
        Proposal ID of each fiber (e.g, S23A-001QN).
    obCode : `numpy.chararray`
        Code for an Observing Block (OB) of each fiber.
    fiberFlux : `list` of `numpy.ndarray` of `float`
        Array of fiber fluxes for each fiber, in [nJy].
    psfFlux : `list` of `numpy.ndarray` of `float`
        Array of PSF fluxes for each target/fiber in [nJy]
    totalFlux : `list` of `numpy.ndarray` of `float`
        Array of total fluxes for each target/fiber in [nJy].
    fiberFluxErr : `list` of `numpy.ndarray` of `float`
        Array of fiber flux errors for each fiber in [nJy].
    psfFluxErr : `list` of `numpy.ndarray` of `float`
        Array of PSF flux errors for each target/fiber in [nJy].
    totalFluxErr : `list` of `numpy.ndarray` of `float`
        Array of total flux errors for each target/fiber in [nJy].
    filterNames : `list` of `list` of `str`
        List of filters used to measure the fiber fluxes for each filter.
    guideStars : `GuideStars`
        Guide star data. If `None`, an empty GuideStars instance will be created.
    designName : `str`
        Name for design file. If `None` use default
    fiberidsPath : `str` or None
        Path to the directory containing information of fiber IDs.
        This will be passed as the `path` keyword to `FiberIds()`.
        Default is set to `None`, i.e., `eups` will be used to search the path.
    obstime : `str`, optional
        Designed observation time ISO format (UTC-time).

    Returns
    -------
    pfsDesign : `pfs.datamodel.PfsDesign`
       constructed pfsDesign for all fibers (science + engineering)
    """

    # Grand Fiber Map
    gfm = FiberIds(path=fiberidsPath)
    isEmpty = gfm.scienceFiberId == FiberIds.EMPTY
    isEng = gfm.scienceFiberId == FiberIds.ENGINEERING
    isCobra = ~isEmpty & ~isEng

    nFiber = len(gfm.scienceFiberId)
    nScienceFiber = len(gfm.scienceFiberId[isCobra])
    fiberId = gfm.fiberId

    def setDefaultValues(sciVal, engVal, shape=nFiber, dtype=float):
        """assign provided sci values to cobra index and default value to engineering fibers. "
        Parameters
        ----------
        sciVal : array_like of dtype / dtype
             value(s) for science fibers.
        engVal: array_like of dtype / dtype
            value(s) for engineering fibers.
        """
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

    if len(pfiNominal) == 0:
        raise RuntimeError("You must specify at least one position (n.b. [(NaN, NaN)] is acceptable)")

    pfiNominal = setDefaultValues(sciVal=pfiNominal, engVal=np.nan, shape=(nFiber, 2))

    tract = setDefaultValues(sciVal=tract, engVal=-1, dtype='int32')
    patch = setDefaultValues(sciVal=patch, engVal='0,0', dtype='U32')

    ra = setDefaultValues(sciVal=ra, engVal=np.nan)
    dec = setDefaultValues(sciVal=dec, engVal=np.nan)

    catId = setDefaultValues(sciVal=catId, engVal=-1, dtype='int32')
    objId = setDefaultValues(sciVal=objId, engVal=-1, dtype='int64')

    targetType = setDefaultValues(sciVal=targetType, engVal=TargetType.ENGINEERING, dtype=int)
    fiberStatus = setDefaultValues(sciVal=fiberStatus, engVal=FiberStatus.GOOD)

    epoch = setDefaultValues(sciVal=epoch, engVal="J2000.0", dtype="U32")
    pmRa = setDefaultValues(sciVal=pmRa, engVal=0.0, dtype="float32")
    pmDec = setDefaultValues(sciVal=pmDec, engVal=0.0, dtype="float32")
    parallax = setDefaultValues(sciVal=parallax, engVal=1.0e-8, dtype="float32")

    proposalId = setDefaultValues(sciVal=proposalId, engVal="N/A", dtype="U32")
    obCode = setDefaultValues(sciVal=obCode, engVal="N/A", dtype=object)  # string with arbitrary lengths

    # I might be overaccommodating here but ...
    if filterNames is None:
        # making sure some input data are not discarded silently
        assert np.isnan(fiberFlux) and np.isnan(psfFlux) and np.isnan(totalFlux)
        filterNameArray = np.array(nScienceFiber * [[]], dtype='str')  # list of filter names

    elif isinstance(filterNames, str):
        filterNameArray = np.array(nScienceFiber * [[filterNames]])

    else:
        filterNameArray = np.array(filterNames)
        filterNameArray = filterNameArray[:, None] if filterNameArray.ndim == 1 else filterNameArray

    nFilter = filterNameArray.shape[1]
    filterNameArray = setDefaultValues(sciVal=filterNameArray, engVal=None, dtype=object, shape=(nFiber, nFilter))
    fiberFlux = setDefaultValues(sciVal=fiberFlux, engVal=None, dtype=object, shape=(nFiber, nFilter))
    psfFlux = setDefaultValues(sciVal=psfFlux, engVal=None, dtype=object, shape=(nFiber, nFilter))
    totalFlux = setDefaultValues(sciVal=totalFlux, engVal=None, dtype=object, shape=(nFiber, nFilter))
    fiberFluxErr = setDefaultValues(sciVal=fiberFluxErr, engVal=None, dtype=object, shape=(nFiber, nFilter))
    psfFluxErr = setDefaultValues(sciVal=psfFluxErr, engVal=None, dtype=object, shape=(nFiber, nFilter))
    totalFluxErr = setDefaultValues(sciVal=totalFluxErr, engVal=None, dtype=object, shape=(nFiber, nFilter))

    filterList = [list(filter(None, filters)) for filters in filterNameArray]
    fiberFlux = [np.array(list(filter(None, values)), dtype=float) for values in fiberFlux]
    psfFlux = [np.array(list(filter(None, values)), dtype=float) for values in psfFlux]
    totalFlux = [np.array(list(filter(None, values)), dtype=float) for values in totalFlux]
    fiberFluxErr = [np.array(list(filter(None, values)), dtype=float) for values in fiberFluxErr]
    psfFluxErr = [np.array(list(filter(None, values)), dtype=float) for values in psfFluxErr]
    totalFluxErr = [np.array(list(filter(None, values)), dtype=float) for values in totalFluxErr]

    pfsUtilsVer = getVersion('pfs_utils')

    pfsDesign = PfsDesign(0x0, raBoresight, decBoresight, posAng, arms, fiberId, tract, patch, ra, dec, catId, objId,
                          targetType, fiberStatus,
                          epoch, pmRa, pmDec, parallax,
                          proposalId, obCode,
                          fiberFlux, psfFlux, totalFlux, fiberFluxErr, psfFluxErr,
                          totalFluxErr, filterList, pfiNominal, guideStars, obstime=obstime, pfsUtilsVer=pfsUtilsVer)

    if designName is not None:
        pfsDesign.designName = designName

    # Drop empty fibers
    pfsDesign = pfsDesign[~isEmpty]
    pfsDesign.isSubset = False

    pfsDesign.pfsDesignId = calculate_pfsDesignId(pfsDesign.fiberId, pfsDesign.ra, pfsDesign.dec)

    return pfsDesign


def showPfsDesign(pfsDesigns, date=None, timerange=8, showTime=None, showDesignId=True):
    """Show the altitude at Subaru of the available pfsDesigns as a function of time.

    pfsDesigns: array of `pfs.datamodel.PfsDesign`
       The available designs
    date: `str` or ``None``
       The desired date; if None use today (in HST)
       If None, also show the current time unless showTime is False
    timerange: `int`
       Total number of hours to show, centred at midnight
    showTime: `bool` or `None`
       If False don't show the current time even if known
    showDesignId: `bool`
       Include the pfsDesignId in the labels
    """
    if date is None:
        now = datetime.datetime.now(pytz.timezone('US/Hawaii'))
        date = now.date()

        now += datetime.timedelta(utcoffset / 24)  # all times are actually kept in UTC
    else:
        now = None

    midnight = Time(f'{date} 00:00:00')  # UTC, damn astropy
    time = midnight + np.linspace(24 - timerange / 2, 24 + timerange / 2, 100) * u.hour

    global subaru
    if subaru is None:
        subaru = EarthLocation.of_site('Subaru')

    telstate = AltAz(obstime=time - utcoffset * u.hour, location=subaru)  # AltAz needs UTC

    xformatter = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(xformatter)

    for pfsDesign in pfsDesigns:
        boresight = SkyCoord(ra=pfsDesign.raBoresight * u.degree, dec=pfsDesign.decBoresight * u.degree,
                             frame='icrs')

        label = showDesignId
        plt.plot(time.datetime, boresight.transform_to(telstate).alt,
                 label=f"0x{pfsDesign.pfsDesignId:x} {pfsDesign.designName}" if showDesignId else
                 pfsDesign.designName)

    plt.gcf().autofmt_xdate()
    plt.legend(prop={"family": "monospace"} if showDesignId else {})

    if now is not None and showTime is not False:
        plt.axvline(now, ls='-', color='black', alpha=0.5)

    plt.ylim(max([-1, plt.ylim()[0]]), None)

    plt.xlabel("Local Time")
    plt.ylabel("Altitude (deg)")
    plt.title(f"{date}")
