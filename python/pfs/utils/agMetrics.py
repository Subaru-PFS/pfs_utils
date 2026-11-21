"""Image sizes and sky transparency from the AG spots in opdb ``agc_data``.

Image sizes
-----------

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

Seeing
------
`calibratedSeeing` turns an image size into the atmospheric seeing, the FWHM at zenith. An AG
image size is broader than the seeing: it grows with airmass and carries the instrument's own
PSF. So it is corrected to zenith by ``airmass ** AG_SEEING_AIRMASS_EXPONENT``, and the PSF
`AG_SEEING_PSF_FWHM` is taken out in quadrature. Both are from seven clear nights against the
CFHT/MKWC DIMM, the Maunakea seeing standard: the corrected median tracks DIMM one to one, where
the raw image size reads about 16% high and floors near 0.5 arcsec in the best seeing. There is
about 15% scatter per exposure, so it is a nightly conditions proxy, not a seeing monitor. The
PSF was fit on exposures before ics_agccActor 2.4.0; confirm it on 2.4.0 before trusting the
seeing below about 0.5 arcsec.

Transparency
------------
The transparency a guide star shows is its measured flux over the flux its catalogue
magnitude predicts: 10^(-0.4 dmag), where dmag is ics_agccActor's instrumental magnitude
(``agc_data.estimated_magnitude``) less the catalogue magnitude
(``pfs_design_agc.guide_star_magnitude``) and a colour term in the star's Gaia BP-RP
(``pfs_design_agc.guide_star_color``). It is 1 for a clear sky and falls with cloud. The
colour term and the selection of stars are those of the observatory's tool
(pfs_obsproc_qa_tools, ``obsCondition.calcTransparency``), whose qadb values these reproduce.

``estimated_magnitude`` has meant several things:

- From ics_agccActor 2.4.0 (``agc_exposure.version_actor``) it comes from the flux in a fixed
  aperture and is linear in the star's light. Its zero point is nominal: it equals the
  earlier magnitude for a star of magnitude 16.
- From exposure 484418 (2024-05-02) until then it comes from the isophotal flux, which holds a
  smaller share of a fainter star's light: the transparency of the same sky reads 0.72 for
  stars of magnitude 13 to 14 and 0.83 at 17 to 18. These are the exposures whose qadb
  values are reproduced.
- Earlier exposures hold it with other coefficients, or with an offset of 13.92 mag (March
  2024), or not at all (before April 2023): they give no transparency here.

The transparency is not yet calibrated to read 1 under a clear sky: in 2026 the clearest
exposures of a night read 0.75 to 0.92, the cameras differ by up to 20%, and the level changes
by up to 10% from one observing run to the next. ``zeroPoint`` is where a calibration goes
(the run's zero point plus the camera's). Until there is one, compare an exposure with others
of the same field, not with 1.
"""

from dataclasses import dataclass

import numpy as np
from pfs.utils.datamodel.ag import SourceDetectionFlags

__all__ = [
    "AG_ARCSEC_PER_PIX",
    "AG_COLOR_TERM_COEFFS",
    "AG_COLOR_TERM_OFFSET_MAG",
    "AG_FOCUS_MM_PER_PIX2",
    "AG_FOCUS_OFFSET_MM",
    "AG_FWHM_PER_SIGMA",
    "AG_PIXEL_SIZE_UM",
    "AG_PLATE_SCALE_UM_PER_ARCSEC",
    "VARIANCE_PER_AGC_MOMENT",
    "AgImageSize",
    "AgTransparency",
    "agColorTerm",
    "magnitudesToTransparency",
    "momentsToFwhm",
    "momentsToSigma",
    "storedRadiusToFwhm",
    "summarizeFwhm",
    "summarizeTransparency",
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

#: The exponent of the airmass scaling of seeing: FWHM grows as airmass to this power off zenith
#: (the Kolmogorov value). `calibratedSeeing` uses it to correct an AG image size to zenith.
AG_SEEING_AIRMASS_EXPONENT = 0.6

#: The AG instrument PSF (arcsec FWHM) added to the atmospheric seeing in quadrature, so the AG
#: image size floors here in excellent seeing. Calibrated against CFHT/MKWC DIMM over seven clear
#: nights, on the trace-radius FWHM `momentsToFwhm` gives by default. Provisional: fit on
#: ics_agccActor exposures before agcc 2.4.0; confirm on 2.4.0 before relying on it below ~0.5 arcsec.
AG_SEEING_PSF_FWHM = 0.31

#: The correction for an ics_agccActor bug: a spot's variance over its second moment in
#: ``agc_data``, for the exposures whose ``agc_exposure.version_actor`` is ``"git"``, which
#: hold half the variance (see the module docstring). Moments stored since are the variance.
VARIANCE_PER_AGC_MOMENT = 2.0

#: The slope of ics_agActor's focus calibration, mm per pix^2. A camera's focus error (mm) is
#: this times the difference between the two halves of its detector in the spots' second
#: moments, less `AG_FOCUS_OFFSET_MM`: the median of mxx + myy over the left half minus that
#: over the right half (`SourceDetectionFlags.RIGHT`, where the glass was removed). The moments
#: are the variance; on the moments a bug in ics_agccActor halved (`VARIANCE_PER_AGC_MOMENT`),
#: on which it was calibrated, the slope was twice this.
AG_FOCUS_MM_PER_PIX2 = 0.0043

#: The offset of that calibration, mm: subtracted from the slope term.
AG_FOCUS_OFFSET_MM = 0.026

#: The colour term of the AG instrumental magnitude: a polynomial in Gaia BP-RP, highest power
#: first (mag), from pfs_obsproc_qa_tools. `agColorTerm` adds `AG_COLOR_TERM_OFFSET_MAG` to it.
AG_COLOR_TERM_COEFFS = (-0.00578838, 0.02017098, -0.00800825, 0.06147291, 0.12580689, -0.03511767)

#: The constant pfs_obsproc_qa_tools adds to that polynomial (mag).
AG_COLOR_TERM_OFFSET_MAG = -0.20

#: The BP-RP range the colour term is used over; a star outside it gets no transparency.
_COLOR_RANGE = (-1.0, 3.0)

#: A star's transparency counts only between these: beyond them it is a mismatch, not sky.
_TRANSPARENCY_RANGE = (0.0, 2.0)

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


def calibratedSeeing(fwhm, altitude, *, psf=AG_SEEING_PSF_FWHM, airmassExponent=AG_SEEING_AIRMASS_EXPONENT):
    """Return the atmospheric seeing (arcsec FWHM at zenith) from an AG image size.

    The AG image size (`momentsToFwhm`, `summarizeFwhm`) is the spots' FWHM at the field's
    airmass, broadened by the instrument. This corrects it to zenith by ``airmass ** exponent``,
    then removes the instrument PSF in quadrature: ``sqrt((fwhm / airmass ** exponent) ** 2 -
    psf ** 2)``, with ``airmass = 1 / sin(altitude)``. Over seven clear nights it reproduces the
    CFHT/MKWC DIMM seeing in the median (per-exposure scatter about 15%); it is a nightly
    conditions proxy, not a per-exposure seeing monitor.

    Parameters
    ----------
    fwhm : array-like of `float`
        The AG image size (arcsec), such as `AgImageSize.fwhm`, at the field's airmass.
    altitude : array-like of `float`
        The telescope altitude (degrees), ``agc_exposure.altitude``.
    psf : `float`
        The instrument PSF removed in quadrature (arcsec FWHM); `AG_SEEING_PSF_FWHM`.
    airmassExponent : `float`
        The exponent of the airmass scaling; `AG_SEEING_AIRMASS_EXPONENT`.

    Returns
    -------
    seeing : `numpy.ndarray` of `float`, or `numpy.float64` for scalars
        The atmospheric seeing (arcsec FWHM at zenith). Zero where the image size is at or below
        the PSF (seeing too good for the AG to measure); NaN where an input is NaN or the altitude
        is not above the horizon.
    """
    fwhm = np.asarray(fwhm, dtype=float)
    sinAlt = np.sin(np.radians(np.asarray(altitude, dtype=float)))
    airmass = 1.0 / np.where(sinAlt > 0, sinAlt, np.nan)  # NaN below the horizon, no divide by zero
    zenith = fwhm / airmass ** airmassExponent
    return np.sqrt(np.clip(zenith**2 - np.asarray(psf, dtype=float) ** 2, 0.0, None))[()]


def agColorTerm(color):
    """Return the colour term of the AG instrumental magnitude (mag).

    Parameters
    ----------
    color : array-like of `float`
        The stars' Gaia BP-RP (``pfs_design_agc.guide_star_color``).

    Returns
    -------
    term : `numpy.ndarray` of `float`, or `numpy.float64` for scalars
        What to subtract from the instrumental minus the catalogue magnitude: the polynomial
        `AG_COLOR_TERM_COEFFS` plus `AG_COLOR_TERM_OFFSET_MAG`. NaN outside -1 < BP-RP < 3.
    """
    color = np.asarray(color, dtype=float)
    inRange = (color > _COLOR_RANGE[0]) & (color < _COLOR_RANGE[1])
    return np.where(inRange, np.polyval(AG_COLOR_TERM_COEFFS, color) + AG_COLOR_TERM_OFFSET_MAG, np.nan)[()]


def magnitudesToTransparency(estimatedMagnitude, catalogMagnitude, color, *, zeroPoint=0.0):
    """Return the sky transparency each AG guide star shows.

    Parameters
    ----------
    estimatedMagnitude : array-like of `float`
        ``agc_data.estimated_magnitude``, ics_agccActor's instrumental magnitude (see the
        module docstring for the exposures it is valid for).
    catalogMagnitude : array-like of `float`
        ``pfs_design_agc.guide_star_magnitude``.
    color : array-like of `float`
        ``pfs_design_agc.guide_star_color``, Gaia BP-RP.
    zeroPoint : `float` or array-like of `float`
        What a star reads, less its catalogue magnitude and colour term, under a clear sky
        (mag): a calibration's zero point, which may differ per star (by camera). Zero, the
        default, is no calibration (see the module docstring).

    Returns
    -------
    transparency : `numpy.ndarray` of `float`, or `numpy.float64` for scalars
        Measured over expected flux, 10^(-0.4 dmag): less under cloud; a star measured
        fainter than its catalogue magnitude gives less. NaN where an input is missing or
        the colour is outside the colour term's range.
    """
    dmag = np.asarray(estimatedMagnitude, dtype=float) - np.asarray(catalogMagnitude, dtype=float)
    dmag = dmag - agColorTerm(color) - np.asarray(zeroPoint, dtype=float)
    return np.power(10.0, -0.4 * dmag)[()]


@dataclass(frozen=True)
class AgTransparency:
    """The sky transparency of one AG exposure.

    Attributes
    ----------
    transparency : `float`
        Median transparency of the stars used; NaN if there are none.
    nStars : `int`
        Number of stars used.
    """

    transparency: float
    nStars: int


def summarizeTransparency(transparency, estimatedMagnitude, *, brightFraction=0.75):
    """Return the median transparency of an AG exposure's guide stars.

    Parameters
    ----------
    transparency : array-like of `float`
        The stars' transparency, from `magnitudesToTransparency`, for the detections matched
        to guide stars. NaN and values outside 0 to 2 are left out.
    estimatedMagnitude : array-like of `float`
        The stars' ``agc_data.estimated_magnitude``, to choose the brighter ones.
    brightFraction : `float`
        Use the stars up to this quantile of ``estimatedMagnitude``: the faintest quarter is
        left out by default.

    Returns
    -------
    summary : `AgTransparency`
        The median and the number of stars behind it.

    Notes
    -----
    pfs_obsproc_qa_tools takes the quantile over all the stars of a visit rather than of one
    exposure, so its per-exposure values differ a little from these where the stars change
    during a visit.
    """
    transparency = np.asarray(transparency, dtype=float)
    magnitude = np.asarray(estimatedMagnitude, dtype=float)
    finite = np.isfinite(magnitude)
    if not finite.any():
        return AgTransparency(transparency=float("nan"), nStars=0)
    with np.errstate(invalid="ignore"):
        use = magnitude <= np.quantile(magnitude[finite], brightFraction)
        use = use & (transparency > _TRANSPARENCY_RANGE[0]) & (transparency < _TRANSPARENCY_RANGE[1])
    if not use.any():
        return AgTransparency(transparency=float("nan"), nStars=0)
    return AgTransparency(transparency=float(np.median(transparency[use])), nStars=int(use.sum()))
