"""The image sizes of AG spots from the second moments ics_agccActor stores."""

import unittest

import numpy as np
from pfs.utils import agMetrics
from pfs.utils.coordinates import Subaru_POPT2_PFS
from pfs.utils.datamodel.ag import SourceDetectionFlags

AG_RADIUS_MM = 241.0
"""Radius of the AG cameras on the focal plane."""


def storedMoments(image, x, y, covariance):
    """Second moments of an image as ics_agccActor stores them.

    Its ``windowedFWHM`` weights the image by a Gaussian and iterates the weight until it has
    the spot's covariance, then returns the weighted moments of that last pass. Here the
    weight is given the spot's covariance directly, which is where the iteration ends.
    """
    (cxx, cxy), (_, cyy) = covariance
    det = cxx * cyy - cxy**2
    weight = np.exp(-0.5 * (x * x * cyy - 2 * x * y * cxy + y * y * cxx) / det)
    norm = (image * weight).sum()
    return (
        (image * weight * x * x).sum() / norm,
        (image * weight * y * y).sum() / norm,
        (image * weight * x * y).sum() / norm,
    )


def gaussianSpot(covariance, half=30, oversample=5):
    """A Gaussian spot on a pixel grid finer than its size; returns the image and its x, y."""
    step = 1.0 / oversample
    y, x = np.mgrid[-half : half + step : step, -half : half + step : step]
    (cxx, cxy), (_, cyy) = covariance
    det = cxx * cyy - cxy**2
    return np.exp(-0.5 * (x * x * cyy - 2 * x * y * cxy + y * y * cxx) / det), x, y


class MomentsTestCase(unittest.TestCase):
    def testGaussianFwhm(self):
        """A Gaussian spot's stored moments give back its FWHM."""
        for sigma in (1.5, 2.0, 3.5):
            covariance = np.diag([sigma**2, sigma**2])
            mxx, myy, mxy = storedMoments(*gaussianSpot(covariance), covariance)
            expected = agMetrics.GAUSSIAN_FWHM_PER_SIGMA * sigma * agMetrics.AG_ARCSEC_PER_PIX
            for radius in ("trace", "determinant"):
                self.assertAlmostEqual(agMetrics.momentsToSigma(mxx, myy, mxy, radius=radius), sigma, places=6)
                self.assertAlmostEqual(agMetrics.momentsToFwhm(mxx, myy, mxy, radius=radius), expected, places=6)
            # Negative control: the stored moment is not the variance; read as one, the FWHM
            # comes out sqrt(2) too small.
            naive = agMetrics.GAUSSIAN_FWHM_PER_SIGMA * np.sqrt(0.5 * (mxx + myy)) * agMetrics.AG_ARCSEC_PER_PIX
            self.assertFalse(np.allclose(naive, expected, rtol=0.25))
            self.assertAlmostEqual(expected / naive, np.sqrt(2), places=6)

    def testRadii(self):
        """The trace radius grows with elongation; the determinant radius keeps the area."""
        covariance = np.array([[6.0, 2.0], [2.0, 3.0]])
        mxx, myy, mxy = storedMoments(*gaussianSpot(covariance), covariance)
        trace = agMetrics.momentsToSigma(mxx, myy, mxy)
        determinant = agMetrics.momentsToSigma(mxx, myy, mxy, radius="determinant")
        self.assertAlmostEqual(trace, np.sqrt(0.5 * np.trace(covariance)), places=6)
        self.assertAlmostEqual(determinant, np.linalg.det(covariance) ** 0.25, places=6)
        self.assertGreater(trace, determinant)
        # Negative control: the determinant radius needs the cross moment; without it the
        # elongated spot reads larger.
        self.assertGreater(agMetrics.momentsToSigma(mxx, myy, 0.0, radius="determinant"), determinant + 0.05)

    def testArraysAndBadMoments(self):
        """Arrays in, arrays out; moments that are not a spot's give NaN."""
        mxx = np.array([2.0, -1.0, 2.0, 2.0])
        myy = np.array([2.0, 2.0, -0.5, 2.0])
        mxy = np.array([0.0, 0.0, 0.0, 3.0])
        np.testing.assert_allclose(agMetrics.momentsToSigma(mxx, myy), [2.0, np.nan, np.nan, 2.0])
        np.testing.assert_allclose(
            agMetrics.momentsToSigma(mxx, myy, mxy, radius="determinant"), [2.0, np.nan, np.nan, np.nan]
        )
        self.assertEqual(agMetrics.momentsToFwhm(mxx, myy).shape, (4,))
        self.assertEqual(np.ndim(agMetrics.momentsToFwhm(2.0, 2.0)), 0)
        with self.assertRaises(ValueError):
            agMetrics.momentsToSigma(mxx, myy, radius="determinant")
        with self.assertRaises(ValueError):
            agMetrics.momentsToSigma(mxx, myy, mxy, radius="isophotal")


class PlateScaleTestCase(unittest.TestCase):
    def modelScales(self, flag):
        """Radial and tangential plate scale (microns/arcsec) of the optical model at the AG radius."""
        model = Subaru_POPT2_PFS.POPT2()

        def focalPlane(sep, zpa):
            x, y = model.celestial2focalplane(np.array([sep]), np.array([zpa]), 0.0, 0.0, 60.0, 6.0, 0.62, np.array([flag]))
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
            self.assertAlmostEqual(np.sqrt(radial * tangential) / agMetrics.AG_PLATE_SCALE_UM_PER_ARCSEC, 1.0, delta=0.005)
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


if __name__ == "__main__":
    unittest.main()
