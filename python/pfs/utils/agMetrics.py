"""Image sizes of AG spots from the second moments in opdb ``agc_data``.

ics_agccActor measures each spot's second moments under a Gaussian weight that it iterates
until the weight matches the spot (adaptive moments), and stores them in
``central_image_moment_20_pix``, ``_02_pix`` and ``_11_pix``. The stored moments are the
spot's variance.

They were not always. A bug in ics_agccActor made it store the weighted moments of its last
pass instead of the moments of the converged weight: a Gaussian seen through a matched
Gaussian weight has half its variance, so every moment was half what it should be and a size
taken from it sqrt(2) too small. The exposures written with the bug are those whose
``agc_exposure.version_actor`` is the literal ``"git"``; agcc has recorded its real version
since the fix. `VARIANCE_PER_AGC_MOMENT` (2) is the correction for those exposures, and
``variancePerMoment`` is how the functions here are told which kind of moment they are given:
2 for moments as stored by the bug, 1 for the variance. A reader should correct the moments
where it reads them (ics_agActor and drp_qa double the ``"git"`` exposures in their queries)
and leave the default, 1.

Not every ``"git"`` exposure has the bug. agcc wrote that literal from May 2022, before it
measured adaptive moments: its earliest exposures hold the detection's unweighted moments,
about the variance, on every spot (none is flagged ``BAD_SHAPE`` before exposure 174776,
2023-04-20). The version does not tell them apart, so readers double them too and their sizes
read sqrt(2) too large, as they did before the fix.

The bug touched only spots whose iteration converged. A spot flagged
`SourceDetectionFlags.BAD_SHAPE` keeps the unweighted moments of the detection instead, about
the variance in any exposure: it needs no correction. Those moments are another measurement,
though: leave such spots out before calling, as ics_agActor's detection filter does. Before
ics_agccActor 2.1.0 (2026-06-19) the iteration used the wrong cross moment, and spots
elongated off the pixel axes were stored smaller.

The spots are not Gaussian. Moffat profiles fitted to 1,200 isolated spots in 24 AG frames from
two observing runs (17 nights, four fields, FWHM of 4 to 9 pixels) have a median beta of 3.3 and
3.9 and a FWHM of 2.88 and 2.94 times the radius of the moments as the bug stored them, the same
for both detector halves; the frames' medians run from 2.7 to 3.0. The factor used here is 2.9,
which is 2.05 sigma where a Gaussian's FWHM is 2.355 sigma. It is a measurement, good to about 3%
from frame to frame, not a property of a profile.
"""

from dataclasses import dataclass

import numpy as np
from pfs.utils.datamodel.ag import SourceDetectionFlags

__all__ = [
    "AG_ARCSEC_PER_PIX",
    "AG_FWHM_PER_SIGMA",
    "AG_PIXEL_SIZE_UM",
    "AG_PLATE_SCALE_UM_PER_ARCSEC",
    "VARIANCE_PER_AGC_MOMENT",
    "AgImageSize",
    "momentsToFwhm",
    "momentsToSigma",
    "storedRadiusToFwhm",
    "summarizeFwhm",
]

#: Size of an AG detector pixel, microns.
AG_PIXEL_SIZE_UM = 13.0

#: Plate scale at the AG cameras, microns per arcsec: the geometric mean of the radial (98.1)
#: and tangential (91.6) scales of ``Subaru_POPT2_PFS`` at the cameras' radius, 241 mm.
AG_PLATE_SCALE_UM_PER_ARCSEC = 94.7

#: An AG pixel on the sky, arcsec (0.137).
AG_ARCSEC_PER_PIX = AG_PIXEL_SIZE_UM / AG_PLATE_SCALE_UM_PER_ARCSEC

#: The FWHM of an AG spot over its sigma, measured (see the module docstring); 2.355 for a Gaussian.
AG_FWHM_PER_SIGMA = 2.05

#: The correction for an ics_agccActor bug: a spot's variance over its second moment in
#: ``agc_data``, for the exposures whose ``agc_exposure.version_actor`` is ``"git"``, which
#: hold half the variance (see the module docstring). Moments stored since are the variance.
VARIANCE_PER_AGC_MOMENT = 2.0

_RADII = ("trace", "determinant")


def momentsToSigma(mxx, myy, mxy=None, *, radius="trace", variancePerMoment=1.0):
    """Return the sigma of AG spots from their ``agc_data`` second moments.

    Parameters
    ----------
    mxx, myy : array-like of `float`
        ``central_image_moment_20_pix`` and ``central_image_moment_02_pix`` (pix^2).
    mxy : array-like of `float`, optional
        ``central_image_moment_11_pix`` (pix^2). Needed for the determinant radius.
    radius : `str`
        ``"trace"`` for sqrt((mxx + myy) / 2), which grows with any elongation of the spot;
        ``"determinant"`` for (mxx myy - mxy^2)^(1/4), the radius of the circle with the
        spot's area.
    variancePerMoment : `float` or array-like of `float`
        The spots' variance over the moments given: 1 (the default) for moments that are
        the variance, `VARIANCE_PER_AGC_MOMENT` for moments as ics_agccActor stored them
        while a bug halved them (see the module docstring). An array gives each spot its
        own.

    Returns
    -------
    sigma : `numpy.ndarray` of `float`, or `numpy.float64` for scalars
        The spots' sigma (pix), with ``variancePerMoment`` applied. NaN where the moments are
        not those of a spot (a negative ``mxx`` or ``myy``, or for the determinant radius
        ``mxy^2 > mxx myy``); zero moments give zero. Raises ------ ValueError If ``radius`` is
        not one of the two, or is ``"determinant"`` without ``mxy``.
    """
    if radius not in _RADII:
        raise ValueError(f"radius must be one of {_RADII}, not {radius!r}")
    mxx = np.asarray(mxx, dtype=float)
    myy = np.asarray(myy, dtype=float)
    bad = (mxx < 0) | (myy < 0)
    if radius == "trace":
        moment = 0.5 * (mxx + myy)
    else:
        if mxy is None:
            raise ValueError("The determinant radius needs mxy")
        det = mxx * myy - np.asarray(mxy, dtype=float) ** 2
        bad = bad | (det < 0)
        moment = np.sqrt(np.where(bad, np.nan, det))
    return np.sqrt(np.asarray(variancePerMoment, dtype=float) * np.where(bad, np.nan, moment))


def momentsToFwhm(mxx, myy, mxy=None, *, radius="trace", variancePerMoment=1.0):
    """Return the FWHM (arcsec) of AG spots from their ``agc_data`` second moments.

    `AG_FWHM_PER_SIGMA` times `momentsToSigma`'s sigma, at `AG_ARCSEC_PER_PIX`. The parameters
    are `momentsToSigma`'s.
    """
    sigma = momentsToSigma(mxx, myy, mxy, radius=radius, variancePerMoment=variancePerMoment)
    return AG_FWHM_PER_SIGMA * AG_ARCSEC_PER_PIX * sigma


def storedRadiusToFwhm(radius, *, variancePerMoment=1.0):
    """Return the FWHM (arcsec) of AG spots from the radius of their stored second moments.

    For a radius already taken from the moments, such as a median of sqrt((mxx + myy) / 2)
    computed in the database; `momentsToFwhm` on the moments gives the same.

    Parameters
    ----------
    radius : array-like of `float`
        The radius (pix) of the moments: the spots' sigma, when the moments are the variance.
    variancePerMoment : `float` or array-like of `float`
        As for `momentsToSigma`: the variance over the moments the radius was taken from.

    Returns
    -------
    fwhm : `numpy.ndarray` of `float`, or `numpy.float64` for scalars
        The spots' FWHM (arcsec); NaN for a negative radius.
    """
    radius = np.asarray(radius, dtype=float)
    sigma = np.sqrt(np.asarray(variancePerMoment, dtype=float)) * np.where(radius < 0, np.nan, radius)
    return AG_FWHM_PER_SIGMA * AG_ARCSEC_PER_PIX * sigma


@dataclass(frozen=True)
class AgImageSize:
    """The image size of the spots of one AG exposure.

    The two halves of each AG detector focus differently, so each half has its own size.

    Attributes
    ----------
    fwhm : `float`
        Median FWHM (arcsec) of all the spots; NaN if there are none.
    fwhmLeft, fwhmRight : `float`
        Median FWHM (arcsec) of the spots on the left halves of the detectors and on the
        right halves (`SourceDetectionFlags.RIGHT`); NaN for a half without spots.
    nLeft, nRight : `int`
        Number of spots with a FWHM on each half.
    """

    fwhm: float
    fwhmLeft: float
    fwhmRight: float
    nLeft: int
    nRight: int


def summarizeFwhm(fwhm, flags):
    """Return the median FWHM of an AG exposure's spots, pooled and for each detector half.

    Parameters
    ----------
    fwhm : array-like of `float`
        The spots' FWHM (arcsec), from `momentsToFwhm`; NaN and infinities are left out.
    flags : array-like of `int`
        The spots' ``agc_data.flags``, as integers. Only `SourceDetectionFlags.RIGHT` is read:
        spots with a bad detection flag are counted like the others, so leave them out first.

    Returns
    -------
    size : `AgImageSize`
        The medians and the number of spots behind each.
    """
    fwhm = np.asarray(fwhm, dtype=float)
    right = (np.asarray(flags, dtype=int) & int(SourceDetectionFlags.RIGHT)) != 0
    good = np.isfinite(fwhm)

    def median(select):
        return float(np.median(fwhm[select])) if select.any() else float("nan")

    return AgImageSize(
        fwhm=median(good),
        fwhmLeft=median(good & ~right),
        fwhmRight=median(good & right),
        nLeft=int((good & ~right).sum()),
        nRight=int((good & right).sum()),
    )
