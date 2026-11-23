"""The image sizes of AG spots from the second moments ics_agccActor stores."""

import unittest

import numpy as np
from pfs.utils import agMetrics
from pfs.utils.coordinates import Subaru_POPT2_PFS
from pfs.utils.datamodel.ag import SourceDetectionFlags

AG_RADIUS_MM = 241.0
"""Radius of the AG cameras on the focal plane."""


def adaptiveMoments(image, x, y):
    """Second moments of an image by ics_agccActor's method, as (stored, halved).

    Its ``windowedFWHM`` weights the image by a Gaussian; each pass measures the weighted
    moments, takes the weight out of them and uses the result as the next weight, until the
    weight stops changing. ``stored`` is (mxx, myy, mxy) of that weight, which agcc stores:
    the spot's variance. ``halved`` is the weighted moments of the last pass, which a bug made
    it store instead: half the variance.
    """
    weight = np.diag([6.0, 6.0])
    for _ in range(200):
        (wxx, wxy), (_, wyy) = weight
        w = np.exp(-0.5 * (x * x * wyy - 2 * x * y * wxy + y * y * wxx) / np.linalg.det(weight))
        norm = (image * w).sum()
        mxx, myy, mxy = ((image * w * a * b).sum() / norm for a, b in ((x, x), (y, y), (x, y)))
        measured = np.array([[mxx, mxy], [mxy, myy]])
        newWeight = np.linalg.inv(np.linalg.inv(measured) - np.linalg.inv(weight))
        if np.allclose(newWeight, weight, rtol=1e-9, atol=1e-9):
            break
        weight = newWeight
    else:
        raise AssertionError("The weight did not converge")
    return (weight[0, 0], weight[1, 1], weight[0, 1]), (mxx, myy, mxy)


def storedMoments(image, x, y):
    """The moments ics_agccActor stores for an image: its variance. See `adaptiveMoments`."""
    return adaptiveMoments(image, x, y)[0]


def halvedMoments(image, x, y):
    """The moments ics_agccActor stored for an image while a bug halved them. See `adaptiveMoments`."""
    return adaptiveMoments(image, x, y)[1]


def grid(half=40, oversample=5):
    """The x and y (pix) of a grid finer than the spots."""
    step = 1.0 / oversample
    y, x = np.mgrid[-half : half + step : step, -half : half + step : step]
    return x, y


def gaussianSpot(covariance):
    """A Gaussian spot of the given covariance (pix^2); returns the image and its x, y."""
    x, y = grid()
    (cxx, cxy), (_, cyy) = covariance
    det = cxx * cyy - cxy**2
    return np.exp(-0.5 * (x * x * cyy - 2 * x * y * cxy + y * y * cxx) / det), x, y


def moffatFwhmPerStoredRadius(beta, fwhm=5.0):
    """The FWHM of a round Moffat spot over the radius of the halved moments once stored for it."""
    x, y = grid()
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    mxx, myy, _ = halvedMoments((1 + (x * x + y * y) / alpha**2) ** (-beta), x, y)
    return fwhm / np.sqrt(0.5 * (mxx + myy))


class MomentsTestCase(unittest.TestCase):
    def testGaussianSigma(self):
        """A Gaussian spot's stored moments give back its sigma."""
        halvedFactor = agMetrics.VARIANCE_PER_AGC_MOMENT
        for sigma in (1.5, 2.0, 3.5):
            spot = gaussianSpot(np.diag([sigma**2, sigma**2]))
            mxx, myy, mxy = storedMoments(*spot)
            halved = halvedMoments(*spot)
            fwhm = agMetrics.AG_FWHM_PER_SIGMA * sigma * agMetrics.AG_ARCSEC_PER_PIX
            for radius in ("trace", "determinant"):
                self.assertAlmostEqual(
                    agMetrics.momentsToSigma(mxx, myy, mxy, radius=radius), sigma, places=6
                )
                self.assertAlmostEqual(agMetrics.momentsToFwhm(mxx, myy, mxy, radius=radius), fwhm, places=6)
                # The moments a bug in agcc halved take the correction.
                self.assertAlmostEqual(
                    agMetrics.momentsToSigma(*halved, radius=radius, variancePerMoment=halvedFactor),
                    sigma,
                    places=6,
                )
                self.assertAlmostEqual(
                    agMetrics.momentsToFwhm(*halved, radius=radius, variancePerMoment=halvedFactor),
                    fwhm,
                    places=6,
                )
            # Negative control: a halved moment is not the variance; read as one, the sigma
            # comes out sqrt(2) too small.
            naive = agMetrics.momentsToSigma(*halved)
            self.assertFalse(np.allclose(naive, sigma, rtol=0.25))
            self.assertAlmostEqual(sigma / naive, np.sqrt(2), places=6)

    def testFwhmFactorAgainstMoffats(self):
        """The measured FWHM factor lies among those of Moffat spots like the AG cameras'.

        The factor comes from fits to AG images, not from a profile: the spots' beta runs from
        2.5 to 5, and Moffats over that range bracket it.
        """
        factor = agMetrics.AG_FWHM_PER_SIGMA * np.sqrt(agMetrics.VARIANCE_PER_AGC_MOMENT)
        self.assertAlmostEqual(factor, 2.9, places=2)
        self.assertLess(moffatFwhmPerStoredRadius(2.5), factor)
        self.assertGreater(moffatFwhmPerStoredRadius(4.0), factor)
        self.assertAlmostEqual(factor / moffatFwhmPerStoredRadius(3.3), 1.0, delta=0.04)
        # Negative control: a Gaussian's factor, 2.355 sqrt(2), is above that of any of them.
        gaussian = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(2)
        for beta in (2.5, 3.3, 4.0, 5.0):
            self.assertGreater(gaussian / moffatFwhmPerStoredRadius(beta), 1.05)
        self.assertGreater(gaussian / factor, 1.10)

    def testRadii(self):
        """The trace radius grows with elongation; the determinant radius keeps the area."""
        covariance = np.array([[6.0, 2.0], [2.0, 3.0]])
        mxx, myy, mxy = storedMoments(*gaussianSpot(covariance))
        trace = agMetrics.momentsToSigma(mxx, myy, mxy)
        determinant = agMetrics.momentsToSigma(mxx, myy, mxy, radius="determinant")
        self.assertAlmostEqual(trace, np.sqrt(0.5 * np.trace(covariance)), places=6)
        self.assertAlmostEqual(determinant, np.linalg.det(covariance) ** 0.25, places=6)
        self.assertGreater(trace, determinant)
        # The FWHM follows the radius asked for.
        perSigma = agMetrics.AG_FWHM_PER_SIGMA * agMetrics.AG_ARCSEC_PER_PIX
        self.assertAlmostEqual(agMetrics.momentsToFwhm(mxx, myy, mxy), perSigma * trace, places=6)
        self.assertAlmostEqual(
            agMetrics.momentsToFwhm(mxx, myy, mxy, radius="determinant"), perSigma * determinant, places=6
        )
        # Negative control: the determinant radius needs the cross moment; without it the
        # elongated spot reads larger.
        self.assertGreater(agMetrics.momentsToSigma(mxx, myy, 0.0, radius="determinant"), determinant + 0.05)

    def testStoredRadius(self):
        """A radius of the stored moments gives the FWHM its moments give."""
        mxx = np.array([2.0, 4.5, 0.0])
        myy = np.array([3.0, 1.5, 0.0])
        radius = np.sqrt(0.5 * (mxx + myy))
        np.testing.assert_allclose(agMetrics.storedRadiusToFwhm(radius), agMetrics.momentsToFwhm(mxx, myy))
        self.assertAlmostEqual(agMetrics.storedRadiusToFwhm(2.0), 2.05 * 2.0 * 13 / 94.7, places=6)
        self.assertTrue(np.isnan(agMetrics.storedRadiusToFwhm(-1.0)))
        # A radius of the moments a bug in agcc halved takes the correction: 2.9 times it, per spot if asked.
        halvedFactor = agMetrics.VARIANCE_PER_AGC_MOMENT
        np.testing.assert_allclose(
            agMetrics.storedRadiusToFwhm(radius, variancePerMoment=halvedFactor),
            agMetrics.momentsToFwhm(mxx, myy, variancePerMoment=halvedFactor),
        )
        self.assertAlmostEqual(
            agMetrics.storedRadiusToFwhm(2.0, variancePerMoment=halvedFactor), 2.9 * 2.0 * 13 / 94.7, places=3
        )
        np.testing.assert_allclose(
            agMetrics.storedRadiusToFwhm([2.0, 2.0], variancePerMoment=[halvedFactor, 1]),
            [
                agMetrics.storedRadiusToFwhm(2.0, variancePerMoment=halvedFactor),
                agMetrics.storedRadiusToFwhm(2.0),
            ],
        )
        # Negative controls: the correction on a radius of the variance gives a FWHM sqrt(2) too
        # large; a Gaussian of that sigma has a FWHM 15% larger than the spots'.
        corrected = agMetrics.storedRadiusToFwhm(2.0, variancePerMoment=halvedFactor)
        self.assertAlmostEqual(corrected / agMetrics.storedRadiusToFwhm(2.0), np.sqrt(2), places=6)
        gaussian = 2 * np.sqrt(2 * np.log(2)) * 2.0 * agMetrics.AG_ARCSEC_PER_PIX
        self.assertGreater(gaussian / agMetrics.storedRadiusToFwhm(2.0), 1.10)

    def testArraysAndBadMoments(self):
        """Arrays in, arrays out; moments that are not a spot's give NaN."""
        mxx = np.array([2.0, -1.0, 2.0, 2.0])
        myy = np.array([2.0, 2.0, -0.5, 2.0])
        mxy = np.array([0.0, 0.0, 0.0, 3.0])
        sigma = np.sqrt(2.0)
        np.testing.assert_allclose(agMetrics.momentsToSigma(mxx, myy), [sigma, np.nan, np.nan, sigma])
        np.testing.assert_allclose(
            agMetrics.momentsToSigma(mxx, myy, mxy, radius="determinant"), [sigma, np.nan, np.nan, np.nan]
        )
        self.assertEqual(agMetrics.momentsToFwhm(mxx, myy).shape, (4,))
        self.assertEqual(np.ndim(agMetrics.momentsToFwhm(2.0, 2.0)), 0)
        # One factor per spot, through to the FWHM.
        perSpot = [2, 1]
        np.testing.assert_allclose(
            agMetrics.momentsToSigma([1.0, 2.0], [1.0, 2.0], variancePerMoment=perSpot), [sigma, sigma]
        )
        np.testing.assert_allclose(
            agMetrics.momentsToFwhm([1.0, 2.0], [1.0, 2.0], variancePerMoment=perSpot),
            np.full(2, agMetrics.momentsToFwhm(2.0, 2.0)),
        )
        # Scalars in, NumPy scalars out.
        for value in (agMetrics.momentsToSigma(2.0, 2.0), agMetrics.storedRadiusToFwhm(2.0)):
            self.assertIsInstance(value, np.float64)
        # Zero moments are a spot of no size, not a bad one.
        self.assertEqual(agMetrics.momentsToSigma(0.0, 0.0), 0.0)
        self.assertEqual(agMetrics.momentsToSigma(0.0, 0.0, 0.0, radius="determinant"), 0.0)
        with self.assertRaises(ValueError):
            agMetrics.momentsToSigma(mxx, myy, radius="determinant")
        with self.assertRaises(ValueError):
            agMetrics.momentsToSigma(mxx, myy, mxy, radius="isophotal")


class FocusCalibrationTestCase(unittest.TestCase):
    def testCalibration(self):
        """The focus calibration is for moments that are the variance."""
        self.assertEqual(agMetrics.AG_FOCUS_MM_PER_PIX2, 0.0043)
        self.assertEqual(agMetrics.AG_FOCUS_OFFSET_MM, 0.026)
        # It was calibrated as 0.0086 on halved moments: the same spots give the same focus error.
        halved = np.array([4.5, 7.0])  # the difference between the halves of a detector, pix^2
        variance = agMetrics.VARIANCE_PER_AGC_MOMENT * halved
        np.testing.assert_allclose(agMetrics.AG_FOCUS_MM_PER_PIX2 * variance, 0.0086 * halved)
        # Negative control: the old slope on the variance doubles it.
        self.assertFalse(np.allclose(0.0086 * variance, 0.0086 * halved))


class TransparencyTestCase(unittest.TestCase):
    def testColorTerm(self):
        """The colour term is the 2.4.0 fit: a cubic in BP-RP, plus the clear-sky constant."""
        for color in (0.6, 1.0, 2.2):
            fit = 0.06010 * color**3 - 0.19370 * color**2 + 0.40108 * color - 0.050
            self.assertAlmostEqual(agMetrics.agColorTerm(color), fit, places=12)
        # The fitted values at BP-RP 1 and 2 (the colour polynomial about 1, plus the clear sky).
        self.assertAlmostEqual(agMetrics.agColorTerm(1.0), 0.2174, places=3)
        self.assertAlmostEqual(agMetrics.agColorTerm(2.0) - agMetrics.agColorTerm(1.0), 0.240, places=2)
        # Negative control: pfs_obsproc_qa_tools' colour term, fitted to the isophotal magnitude, is
        # 0.14 mag steeper from BP-RP 1 to 2.
        toolCoeffs = (-0.00578838, 0.02017098, -0.00800825, 0.06147291, 0.12580689, -0.03511767)
        toolSlope = np.diff(np.polyval(toolCoeffs, [1.0, 2.0]))[0]
        self.assertGreater(toolSlope - (agMetrics.agColorTerm(2.0) - agMetrics.agColorTerm(1.0)), 0.1)
        # Outside the range it was fitted over there is no term; the limits themselves are outside.
        for color in (0.0, 0.5, 2.8, 3.5, np.nan):
            self.assertTrue(np.isnan(agMetrics.agColorTerm(color)))
        np.testing.assert_allclose(
            agMetrics.agColorTerm([1.0, 5.0]), [agMetrics.agColorTerm(1.0), np.nan], equal_nan=True
        )
        # Negative control: the coefficients are highest power first; reversed they give another value.
        reversed_ = np.polyval(agMetrics.AG_COLOR_TERM_COEFFS[::-1], 1.5) + agMetrics.AG_COLOR_TERM_OFFSET_MAG
        self.assertFalse(np.isclose(agMetrics.agColorTerm(1.5), reversed_, atol=0.01))

    def testCameraZeroPoint(self):
        """Each spot gets its camera's zero point, by agc_camera_id and detector half."""
        right = int(SourceDetectionFlags.RIGHT)
        # AG3 (id 2): its half without RIGHT is 0.19 mag faint; its RIGHT half is not.
        self.assertAlmostEqual(agMetrics.agCameraZeroPoint(2, 0), 0.190)
        self.assertAlmostEqual(agMetrics.agCameraZeroPoint(2, right), -0.009)
        np.testing.assert_allclose(
            agMetrics.agCameraZeroPoint([0, 0, 5], [0, right, right]), [-0.073, -0.026, -0.069]
        )
        self.assertIsInstance(agMetrics.agCameraZeroPoint(2, 0), np.float64)
        # Negative control: the camera id is 0-based; read as AG number (1-based) AG3 would be AG4's.
        self.assertFalse(np.isclose(agMetrics.agCameraZeroPoint(3, 0), 0.190, atol=0.05))
        # Other flags do not change the half; the zero points are about the mean of the cameras.
        self.assertEqual(agMetrics.agCameraZeroPoint(2, 0x10), agMetrics.agCameraZeroPoint(2, 0))
        working = [z for i, z in enumerate(agMetrics.AG_CAMERA_ZERO_POINTS_MAG) if i != 1]
        self.assertAlmostEqual(float(np.mean(working)), 0.0, places=2)
        # A star on AG3 without RIGHT that reads 0.19 mag faint shows a clear sky with the zero point.
        color, catalog = 1.2, 16.0
        measured = catalog + agMetrics.agColorTerm(color) + 0.190
        zeroPoint = agMetrics.agCameraZeroPoint(2, 0)
        self.assertAlmostEqual(
            agMetrics.magnitudesToTransparency(measured, catalog, color, zeroPoint=zeroPoint), 1.0, places=12
        )
        self.assertLess(agMetrics.magnitudesToTransparency(measured, catalog, color), 0.85)
        with self.assertRaises(IndexError):
            agMetrics.agCameraZeroPoint(6, 0)

    def testTransparency(self):
        """A star measured at its catalogue magnitude, after the colour term, shows a clear sky."""
        color = 0.8
        catalog = 15.0
        clear = catalog + agMetrics.agColorTerm(color)
        self.assertAlmostEqual(agMetrics.magnitudesToTransparency(clear, catalog, color), 1.0, places=12)
        # Half the light is 0.7526 mag fainter.
        halfLight = clear + 2.5 * np.log10(2)
        self.assertAlmostEqual(agMetrics.magnitudesToTransparency(halfLight, catalog, color), 0.5, places=12)
        # Negative control: fainter is less transparent, not more.
        fainter = agMetrics.magnitudesToTransparency(halfLight, catalog, color)
        self.assertFalse(np.isclose(fainter, 2.0, rtol=0.5))
        self.assertLess(fainter, 1.0)
        # Negative control: without the colour term the same star does not read 1.
        self.assertFalse(np.isclose(10 ** (-0.4 * (clear - catalog)), 1.0, rtol=0.05))
        # Arrays in, arrays out; a missing input or a colour out of range gives NaN.
        values = agMetrics.magnitudesToTransparency(
            [clear, np.nan, clear, clear], [catalog, catalog, np.nan, catalog], [color, color, color, 4.0]
        )
        np.testing.assert_allclose(values, [1.0, np.nan, np.nan, np.nan], equal_nan=True)
        self.assertIsInstance(agMetrics.magnitudesToTransparency(clear, catalog, color), np.float64)

    def testZeroPoint(self):
        """A calibration's zero point is what a clear sky reads: a star at it shows a transparency of 1."""
        color, catalog, zeroPoint = 1.2, 16.0, 0.15
        clear = catalog + agMetrics.agColorTerm(color) + zeroPoint
        self.assertAlmostEqual(
            agMetrics.magnitudesToTransparency(clear, catalog, color, zeroPoint=zeroPoint), 1.0, places=12
        )
        # One zero point per star, as for stars on two cameras.
        perStar = agMetrics.magnitudesToTransparency(
            [clear, clear + 0.1], catalog, color, zeroPoint=[zeroPoint, zeroPoint + 0.1]
        )
        np.testing.assert_allclose(perStar, [1.0, 1.0])
        # Negative controls: without it the same star reads low; with the other sign, lower still.
        self.assertAlmostEqual(agMetrics.magnitudesToTransparency(clear, catalog, color), 10 ** (-0.4 * 0.15))
        self.assertLess(agMetrics.magnitudesToTransparency(clear, catalog, color, zeroPoint=-zeroPoint), 0.8)

    def testSummary(self):
        """The median is over the brighter stars with a transparency between 0 and 2."""
        magnitude = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        transparency = np.array([0.90, 0.80, 0.70, 5.0, np.nan, 0.60, 0.10, 0.10])
        summary = agMetrics.summarizeTransparency(transparency, magnitude)
        # The 0.75 quantile of the magnitudes is 17.25: the two faintest stars are out, as are 5.0 and NaN.
        self.assertEqual(summary.nStars, 4)
        self.assertAlmostEqual(summary.transparency, 0.75)
        # Negative control: with every star, the faint ones pull the median down.
        everyStar = agMetrics.summarizeTransparency(transparency, magnitude, brightFraction=1.0)
        self.assertEqual(everyStar.nStars, 6)
        self.assertAlmostEqual(everyStar.transparency, 0.65)
        # Negative control: the stars are chosen by magnitude, not by transparency.
        shuffled = agMetrics.summarizeTransparency(transparency, magnitude[::-1])
        self.assertNotAlmostEqual(shuffled.transparency, summary.transparency)

    def testSummaryWithMagnitudesThatAreNotFinite(self):
        """A star without a finite magnitude is left out; it does not cost the others their quantile."""
        summary = agMetrics.summarizeTransparency([0.9, 0.8, 0.7, 0.6], [16.0, 16.1, np.inf, np.nan])
        # The brighter three quarters of the two stars with a magnitude is one star.
        self.assertEqual(summary.nStars, 1)
        self.assertAlmostEqual(summary.transparency, 0.9)
        # Negative control: all four with a magnitude, and three of them count.
        allFinite = agMetrics.summarizeTransparency([0.9, 0.8, 0.7, 0.6], [16.0, 16.1, 16.2, 16.3])
        self.assertEqual(allFinite.nStars, 3)

    def testSummaryWithoutStars(self):
        for transparency, magnitude in (([], []), ([np.nan, 3.0], [12.0, 13.0]), ([0.9], [np.nan])):
            summary = agMetrics.summarizeTransparency(transparency, magnitude)
            self.assertTrue(np.isnan(summary.transparency))
            self.assertEqual(summary.nStars, 0)


class PlateScaleTestCase(unittest.TestCase):
    def modelScales(self, flag):
        """Radial and tangential plate scale (microns/arcsec) of the optical model at the AG radius."""
        model = Subaru_POPT2_PFS.POPT2()

        def focalPlane(sep, zpa):
            x, y = model.celestial2focalplane(
                np.array([sep]), np.array([zpa]), 0.0, 0.0, 60.0, 6.0, 0.62, np.array([flag])
            )
            return np.array([x[0], y[0]])

        seps = np.linspace(0.6, 0.79, 200)
        radii = np.array([np.hypot(*focalPlane(sep, 0.0)) for sep in seps])
        sep = np.interp(AG_RADIUS_MM, radii, seps)
        arcsec = 1.0 / 3600
        radial = np.hypot(*(focalPlane(sep + arcsec / 2, 0.0) - focalPlane(sep - arcsec / 2, 0.0)))
        turn = np.rad2deg(np.deg2rad(arcsec) / np.sin(np.deg2rad(sep)))
        tangential = np.hypot(*(focalPlane(sep, turn / 2) - focalPlane(sep, -turn / 2)))
        return 1e3 * radial, 1e3 * tangential

    def testPlateScaleIsTheModels(self):
        """The plate scale is the optical model's at the AG cameras, on both detector halves."""
        for flag in (0, 1):
            radial, tangential = self.modelScales(flag)
            self.assertAlmostEqual(
                np.sqrt(radial * tangential) / agMetrics.AG_PLATE_SCALE_UM_PER_ARCSEC, 1.0, delta=0.005
            )
            # Negative control: the scale of a 15 m focal length (72.7 microns/arcsec), which
            # ics_agActor's spot size used, is not the model's.
            self.assertGreater(abs(np.sqrt(radial * tangential) / (15000 / 206.265) - 1.0), 0.25)
        self.assertAlmostEqual(agMetrics.AG_ARCSEC_PER_PIX, 0.137, places=3)


class SummarizeTestCase(unittest.TestCase):
    def testHalves(self):
        """Each detector half has its own median; NaN are left out."""
        right = int(SourceDetectionFlags.RIGHT)
        fwhm = np.array([0.6, 0.8, 1.0, 1.2, np.nan, 1.6])
        flags = np.array([0, 0, right, right, right, right])
        size = agMetrics.summarizeFwhm(fwhm, flags)
        self.assertAlmostEqual(size.fwhmLeft, 0.7)
        self.assertAlmostEqual(size.fwhmRight, 1.2)
        self.assertAlmostEqual(size.fwhm, 1.0)
        self.assertEqual((size.nLeft, size.nRight), (2, 3))
        # Negative control: only RIGHT moves a spot to the right half.
        other = int(SourceDetectionFlags.EDGE | SourceDetectionFlags.BAD_SHAPE)
        same = agMetrics.summarizeFwhm(fwhm, flags | other)
        self.assertEqual(same, size)
        swapped = agMetrics.summarizeFwhm(fwhm, flags ^ right)
        self.assertFalse(np.allclose([swapped.fwhmLeft, swapped.fwhmRight], [size.fwhmLeft, size.fwhmRight]))

    def testEmpty(self):
        """No spots, or none on a half, gives NaN and a count of zero."""
        size = agMetrics.summarizeFwhm([], [])
        self.assertTrue(np.isnan([size.fwhm, size.fwhmLeft, size.fwhmRight]).all())
        self.assertEqual((size.nLeft, size.nRight), (0, 0))
        size = agMetrics.summarizeFwhm([0.7, 0.9], [0, 0])
        self.assertAlmostEqual(size.fwhm, 0.8)
        self.assertTrue(np.isnan(size.fwhmRight))
        self.assertEqual((size.nLeft, size.nRight), (2, 0))
        # An infinite FWHM is left out like a NaN.
        size = agMetrics.summarizeFwhm([0.7, np.inf, np.nan], [0, 0, 0])
        self.assertAlmostEqual(size.fwhm, 0.7)
        self.assertEqual((size.nLeft, size.nRight), (1, 0))


class SeeingCalibrationTestCase(unittest.TestCase):
    """`calibratedSeeing`: airmass to zenith, then the instrument PSF out in quadrature."""

    def testRoundTrip(self):
        """An image size built from a known seeing and airmass comes back as that seeing."""
        psf, exponent = agMetrics.AG_SEEING_PSF_FWHM, agMetrics.AG_SEEING_AIRMASS_EXPONENT
        seeing, altitude = 0.7, 40.0
        airmass = 1.0 / np.sin(np.radians(altitude))
        fwhm = np.sqrt(seeing**2 + psf**2) * airmass**exponent
        self.assertAlmostEqual(float(agMetrics.calibratedSeeing(fwhm, altitude)), seeing)
        # Negative control: skipping the quadrature PSF, or the airmass correction, overestimates.
        zenithOnly = fwhm / airmass**exponent
        self.assertGreater(zenithOnly, seeing + 0.03)  # PSF not removed -> too large
        self.assertGreater(fwhm, zenithOnly + 0.03)  # airmass not removed -> larger still

    def testFloorAndArray(self):
        """At or below the PSF the seeing is NaN (unmeasurable, dropped downstream); inputs broadcast."""
        psf = agMetrics.AG_SEEING_PSF_FWHM
        self.assertTrue(np.isnan(float(agMetrics.calibratedSeeing(psf, 90.0))))  # at zenith, image size = PSF
        self.assertTrue(np.isnan(float(agMetrics.calibratedSeeing(psf * 0.5, 90.0))))  # below the PSF
        out = agMetrics.calibratedSeeing([psf, 1.0, np.nan], [90.0, 90.0, 90.0])
        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(out[1], np.sqrt(1.0 - psf**2))
        self.assertTrue(np.isnan(out[2]))

    def testBelowHorizon(self):
        """A non-positive altitude gives NaN, not a division error."""
        self.assertTrue(np.isnan(float(agMetrics.calibratedSeeing(0.8, 0.0))))
        self.assertTrue(np.isnan(float(agMetrics.calibratedSeeing(0.8, -5.0))))

    def testOverrides(self):
        """The psf and airmass exponent are overridable, the knobs a later refit uses."""
        # Exponent 0 removes the airmass correction; psf 0 removes the quadrature term.
        self.assertAlmostEqual(float(agMetrics.calibratedSeeing(0.9, 30.0, psf=0.0, airmassExponent=0.0)), 0.9)
        base = float(agMetrics.calibratedSeeing(0.9, 90.0, psf=0.2))
        self.assertAlmostEqual(base, np.sqrt(0.9**2 - 0.2**2))
        # Negative control: a larger PSF takes more out, so the seeing is smaller.
        self.assertLess(float(agMetrics.calibratedSeeing(0.9, 90.0, psf=0.4)), base)


if __name__ == "__main__":
    unittest.main()
