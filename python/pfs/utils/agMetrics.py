"""Image sizes of AG spots from the second moments in opdb ``agc_data``.

ics_agccActor measures each spot's second moments under a Gaussian weight that it iterates
until the weight matches the spot (adaptive moments), and stores the weighted moments as
measured (``central_image_moment_20_pix``, ``_02_pix``, ``_11_pix``). A Gaussian seen through
a matched Gaussian weight has half its variance, so the sigma here is that of twice the stored
moment; a size taken straight from the stored moments is too small by sqrt(2).

That holds for spots whose iteration converged. A spot flagged
`SourceDetectionFlags.BAD_SHAPE` keeps the unweighted moments of the detection instead, about
the full variance, and comes out sqrt(2) too large here: leave such spots out before calling,
as ics_agActor's detection filter does. Before ics_agccActor 2.1.0 (2026-06-19) the iteration
used the wrong cross moment, and spots elongated off the pixel axes were stored smaller.

The spots are not Gaussian. Moffat profiles fitted to 1,200 isolated spots in 24 AG frames from
two observing runs (17 nights, four fields, FWHM of 4 to 9 pixels) have a median beta of 3.3
and 3.9 and a FWHM of 2.88 and 2.94 times the radius of the stored moments, the same for both
detector halves; the frames' medians run from 2.7 to 3.0. The factor used here is 2.9, which is
2.05 sigma where a Gaussian's FWHM is 2.355 sigma. It is a measurement, good to about 3% from
frame to frame, not a property of a profile.
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

#: A spot's variance over its second moment in ``agc_data`` (see the module docstring).
VARIANCE_PER_AGC_MOMENT = 2.0

_RADII = ("trace", "determinant")


def momentsToSigma(mxx, myy, mxy=None, *, radius="trace"):
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

    Returns
    -------
    sigma : `numpy.ndarray` of `float`, or `numpy.float64` for scalars
        The spots' sigma (pix), with the factor between the stored moments and the variance
        applied. NaN where the moments are not those of a spot (a negative ``mxx`` or
        ``myy``, or for the determinant radius ``mxy^2 > mxx myy``); zero moments give zero.

    Raises
    ------
    ValueError
        If ``radius`` is not one of the two, or is ``"determinant"`` without ``mxy``.
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
    return np.sqrt(VARIANCE_PER_AGC_MOMENT * np.where(bad, np.nan, moment))


def momentsToFwhm(mxx, myy, mxy=None, *, radius="trace"):
    """Return the FWHM (arcsec) of AG spots from their ``agc_data`` second moments.

    `AG_FWHM_PER_SIGMA` times `momentsToSigma`'s sigma, at `AG_ARCSEC_PER_PIX`. The parameters
    are `momentsToSigma`'s.
    """
    return AG_FWHM_PER_SIGMA * AG_ARCSEC_PER_PIX * momentsToSigma(mxx, myy, mxy, radius=radius)


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
