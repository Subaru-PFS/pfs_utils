import datetime
import numpy as np
import pytz

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from pfs.datamodel import PfsDesign, FiberStatus, TargetType
from pfs.datamodel.utils import calculate_pfsDesignId

from pfs.utils.fiberids import FiberIds

__all__ = ["makePfsDesign", "showPfsDesign"]

subaru = None                           # we'll look it up if we need it
utcoffset = -10  # UTC -> HST.  No daylight saving to worry about


def makePfsDesign(pfiNominal, ra, dec,
                  raBoresight=100, decBoresight=100, posAng=0, arms='br',
                  tract=1, patch='1,1', catId=-1, objId=-1, targetType=TargetType.SCIENCE,
                  fiberStatus=FiberStatus.GOOD,
                  epoch="J2000.0", pmRa=0.0, pmDec=0.0, parallax=1e-8,
                  fiberFlux=np.NaN, psfFlux=np.NaN, totalFlux=np.NaN,
                  fiberFluxErr=np.NaN, psfFluxErr=np.NaN, totalFluxErr=np.NaN,
                  filterNames=None, guideStars=None, designName=None):
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

    Returns
    -------
    pfsDesign : `pfs.datamodel.PfsDesign`
       constructed pfsDesign for all fibers (science + engineering)
    """

    # Grand Fiber Map
    gfm = FiberIds()
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

    pfiNominal = setDefaultValues(sciVal=pfiNominal, engVal=np.NaN, shape=(nFiber, 2))

    tract = setDefaultValues(sciVal=tract, engVal=-1, dtype='int32')
    patch = setDefaultValues(sciVal=patch, engVal='0,0', dtype='U32')

    ra = setDefaultValues(sciVal=ra, engVal=np.NaN)
    dec = setDefaultValues(sciVal=dec, engVal=np.NaN)

    catId = setDefaultValues(sciVal=catId, engVal=-1, dtype='int32')
    objId = setDefaultValues(sciVal=objId, engVal=-1, dtype='int64')

    targetType = setDefaultValues(sciVal=targetType, engVal=TargetType.ENGINEERING, dtype=int)
    fiberStatus = setDefaultValues(sciVal=fiberStatus, engVal=FiberStatus.GOOD)

    epoch = setDefaultValues(sciVal=epoch, engVal="J2000.0", dtype="U32")
    pmRa = setDefaultValues(sciVal=pmRa, engVal=0.0, dtype="float32")
    pmDec = setDefaultValues(sciVal=pmDec, engVal=0.0, dtype="float32")
    parallax = setDefaultValues(sciVal=parallax, engVal=1.0e-8, dtype="float32")

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

    pfsDesign = PfsDesign(0x0, raBoresight, decBoresight, posAng, arms, fiberId, tract, patch, ra, dec, catId, objId,
                          targetType, fiberStatus,
                          epoch, pmRa, pmDec, parallax,
                          fiberFlux, psfFlux, totalFlux, fiberFluxErr, psfFluxErr,
                          totalFluxErr, filterList, pfiNominal, guideStars)

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

        now += datetime.timedelta(utcoffset/24)  # all times are actually kept in UTC
    else:
        now = None

    midnight = Time(f'{date} 00:00:00')  # UTC, damn astropy
    time = midnight + np.linspace(24 - timerange/2, 24 + timerange/2, 100)*u.hour

    global subaru
    if subaru is None:
        subaru = EarthLocation.of_site('Subaru')

    telstate = AltAz(obstime=time - utcoffset*u.hour, location=subaru)  # AltAz needs UTC

    xformatter = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(xformatter)

    for pfsDesign in pfsDesigns:
        boresight = SkyCoord(ra=pfsDesign.raBoresight*u.degree, dec=pfsDesign.decBoresight*u.degree,
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
