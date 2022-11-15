import numpy as np
from pfs.utils.coordinates.CoordTransp import CoordinateTransform
from pfs.datamodel.utils import calculate_pfsDesignId
from pfs.datamodel.pfsConfig import PfsDesign

__all__ = ["makeVariantDesign", ]


def makeVariantDesign(pfsDesign0, variant=0, sigma=1, doHex=False):
    """Return a copy of pfsDesign0, modified suitably
    variant : int; which variant is required (0: no change)
    sigma   : float; standard deviation of random offsets in arcsec
    doHex   : generate a hexagonal dither (not implemented)
    """

    assert not doHex

    # calculate new pfsDesignId
    pfsDesignId = calculate_pfsDesignId(pfsDesign0.fiberId, pfsDesign0.ra, pfsDesign0.dec, variant=variant)

    np.random.seed((pfsDesign0.pfsDesignId + variant) & 0xffffffff)  # numpy doesn't like more than 32 bits

    if variant == 0:
        dra, ddec = np.zeros((2, len(pfsDesign0)))
    else:
        dra, ddec = np.random.normal(0, sigma, size=(2, len(pfsDesign0)))  # arcsec

    # add random dithers to ra and dec
    ra = pfsDesign0.ra   + dra/(3600*np.cos(np.deg2rad(pfsDesign0.dec)))
    dec = pfsDesign0.dec + ddec/3600

    # And now add the _same_ random dithers to pfiNominal
    boresight = [[pfsDesign0.raBoresight], [pfsDesign0.decBoresight]]

    altitude = 70  # we're making a differential offset so the actual value isn't critical
    pa = pfsDesign0.posAng
    utc = "2022-02-05 00:00:00"  # again, the actual value isn't critical

    x0, y0 = CoordinateTransform(np.stack(([pfsDesign0.ra], [pfsDesign0.dec])),  # original (ra, dec) mapped to mm
                                 mode="sky_pfi", za=90.0 - altitude,
                                 pa=pa, cent=boresight, time=utc)[0:2]
    x, y = CoordinateTransform(np.stack(([ra], [dec])),                          # dithered (ra, dec) mapped to mm
                               mode="sky_pfi", za=90.0 - altitude,
                               pa=pa, cent=boresight, time=utc)[0:2]

    pfiNominal = pfsDesign0.pfiNominal.copy()

    pfiNominal.T[0] += x - x0
    pfiNominal.T[1] += y - y0

    if False:
        pfiNominal = np.stack([x0, y0]).T   # check that (x0, y0) aren't crtazy
    #
    # Create the variant PfsDesign
    #
    kwargs = {}
    if variant != 0:
        kwargs.update(variant=variant,
                      designId0=pfsDesign0.pfsDesignId)

    pfsDesign = PfsDesign(pfsDesignId, pfsDesign0.raBoresight, pfsDesign0.decBoresight,
                          pfsDesign0.posAng,
                          pfsDesign0.arms,
                          pfsDesign0.fiberId, pfsDesign0.tract, pfsDesign0.patch, ra, dec,
                          pfsDesign0.catId, pfsDesign0.objId,
                          pfsDesign0.targetType, pfsDesign0.fiberStatus,
                          pfsDesign0.fiberFlux,
                          pfsDesign0.psfFlux,
                          pfsDesign0.totalFlux,
                          pfsDesign0.fiberFluxErr,
                          pfsDesign0.psfFluxErr,
                          pfsDesign0.totalFluxErr,
                          pfsDesign0.filterNames, pfiNominal,
                          pfsDesign0.guideStars,
                          pfsDesign0.designName,
                          **kwargs)

    return pfsDesign
