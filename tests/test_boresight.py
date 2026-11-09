import unittest

import numpy as np
import pandas as pd

import pfs.utils.coordinates.boresight as boresight

CENTRE = np.array([4438.49, 2839.29])
"""True boresight of the synthetic frames, pixels."""

LOOP = np.arange(-165, 166, 30)
"""Rotator angles of a twelve-position boresight loop, degrees."""


def makeFrames(
    insRots,
    *,
    seed=1,
    nPoints=600,
    radius=2700,
    insRotError=0.2,
    frameShift=0.07,
    spotNoise=0.1,
    missing=0.05,
    firstFrameId=15395100
):
    """Synthetic frames of fixed points turned about CENTRE.

    Each frame turns the points by -insrot plus an error of up to insRotError degrees (the
    rotator readback), is displaced as a whole by frameShift pixels rms per axis, and carries
    spotNoise pixels of centroid noise; a fraction missing of the spots is dropped.
    """
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, nPoints))
    phi = rng.uniform(0, 2 * np.pi, nPoints)
    offset = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    kind = np.where(np.arange(nPoints) < 50, 'fiducial', 'cobra')
    ids = np.where(kind == 'fiducial', np.arange(nPoints) + 1, np.arange(nPoints) - 49)

    frames, points = [], []
    for k, insRot in enumerate(insRots):
        frameId = firstFrameId + k
        theta = np.deg2rad(-insRot + rng.uniform(-insRotError, insRotError))
        xy = (
            CENTRE
            + offset @ boresight.rotation(theta).T
            + rng.normal(0, frameShift, 2)
            + rng.normal(0, spotNoise, (nPoints, 2))
        )
        keep = rng.uniform(0, 1, nPoints) >= missing
        frames.append(dict(mcs_frame_id=frameId, insrot=insRot, altitude=89.9))
        points.append(
            pd.DataFrame(
                dict(mcs_frame_id=frameId, kind=kind[keep], id=ids[keep], x=xy[keep, 0], y=xy[keep, 1])
            )
        )

    return pd.DataFrame(frames), pd.concat(points, ignore_index=True)


def displace(frames, points, rows, shift):
    """Displace the frames at rows by shift pixels in x."""
    points.loc[points.mcs_frame_id.isin(frames.mcs_frame_id[rows]), 'x'] += shift


class BoresightTestCase(unittest.TestCase):

    def assertCentre(self, result, tolerance):
        self.assertLess(abs(result.x - CENTRE[0]), tolerance)
        self.assertLess(abs(result.y - CENTRE[1]), tolerance)

    def testLoop(self):
        """Twelve positions recover the centre, with the uncertainty of the measured displacement."""
        result = boresight.measureBoresight(*makeFrames(LOOP))
        self.assertCentre(result, 0.1)
        self.assertEqual(result.nPositions, 12)
        self.assertTrue(result.frames.used.all())
        self.assertTrue(result.sigmaFMeasured)
        self.assertAlmostEqual(result.xErr, result.sigmaF / np.sqrt(12), places=3)
        self.assertTrue(0.01 < result.xErr < 0.04)
        self.assertEqual(result.warnings, [])
        self.assertIsNone(result.standingOut)
        self.assertAlmostEqual(result.altitude, 89.9)

    def testMediocreNight(self):
        """Frames displaced by 0.2 pix rms are all kept, and the uncertainty follows them."""
        result = boresight.measureBoresight(*makeFrames(LOOP, frameShift=0.2, seed=3))
        self.assertTrue(result.frames.used.all())
        self.assertGreater(result.sigmaF, 0.1)
        self.assertGreater(result.xErr, 0.03)
        self.assertCentre(result, 0.25)

    def testRotationFitted(self):
        """The rotations are fitted, not taken from insrot: its errors do not reach the centre."""
        frames, points = makeFrames(LOOP, insRotError=0.3, frameShift=0, spotNoise=0.01)
        result = boresight.measureBoresight(frames, points)
        self.assertCentre(result, 0.01)
        self.assertGreater(result.frames.rotationOffset.abs().max(), 0.05)

    def testWrongInsRotWarned(self):
        """A frame whose insrot is far off still gives the centre, keeps its position, and is named."""
        frames, points = makeFrames(LOOP)
        frames.loc[4, 'insrot'] += 60
        result = boresight.measureBoresight(frames, points)
        self.assertCentre(result, 0.1)
        self.assertEqual(result.nPositions, 12)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn(str(frames.mcs_frame_id[4]), result.warnings[0])

    def testFrameOrder(self):
        """The order the frames come in does not matter."""
        frames, points = makeFrames(LOOP)
        result = boresight.measureBoresight(frames, points)
        shuffled = boresight.measureBoresight(frames.iloc[::-1], points.sample(frac=1, random_state=2))
        self.assertAlmostEqual(result.x, shuffled.x, places=5)
        self.assertAlmostEqual(result.y, shuffled.y, places=5)

    def testTwoFramesPerPosition(self):
        """Frames repeated at each position count as one position."""
        result = boresight.measureBoresight(*makeFrames(np.repeat(LOOP[::3], 2)))
        self.assertEqual(result.nPositions, 4)
        self.assertCentre(result, 0.15)
        self.assertTrue(result.sigmaFMeasured)

    def testTwoFrames(self):
        """Two frames give the centre with the error FRAME_NOISE leads to for their separation."""
        for separation in (30, 90, 180):
            with self.subTest(separation=separation):
                result = boresight.measureBoresight(*makeFrames([0, separation], insRotError=0))
                expected = boresight.FRAME_NOISE / (np.sqrt(2) * np.sin(np.deg2rad(separation) / 2))
                self.assertAlmostEqual(result.xErr, expected, places=3)
                self.assertAlmostEqual(result.yErr, expected, places=3)
                self.assertFalse(result.sigmaFMeasured)
                self.assertTrue(any('could not be checked' in w for w in result.warnings))
                self.assertCentre(result, 5 * expected)

    def testDisplacedFrameLeftOut(self):
        """A frame displaced as a whole is left out, with twelve positions as with four."""
        for insRots, shift in ((LOOP, 2.0), (LOOP[::3], 2.5)):
            with self.subTest(positions=len(insRots)):
                frames, points = makeFrames(insRots)
                displace(frames, points, [1], shift)
                result = boresight.measureBoresight(frames, points)
                self.assertFalse(result.frames.used[1])
                self.assertIn('displaced', result.frames.reason[1])
                self.assertTrue(result.frames.used.drop(1).all())
                self.assertCentre(result, 0.15)

    def testDisplacedFrameNotBlamedAtThreePositions(self):
        """With three positions no frame can be told displaced: none is blamed, the error grows."""
        frames, points = makeFrames([-120, 0, 120])
        displace(frames, points, [1], 2.0)
        result = boresight.measureBoresight(frames, points)
        self.assertTrue(result.frames.used.all())
        self.assertTrue(any('could not be checked' in w for w in result.warnings))
        self.assertGreater(max(result.xErr, result.yErr), 0.25)

    def testStillDisplacedAfterTwo(self):
        """A third displaced frame, once two were left out, is named in the result."""
        frames, points = makeFrames(LOOP)
        displace(frames, points, [1, 5, 9], 3.0)
        result = boresight.measureBoresight(frames, points)
        self.assertEqual(result.frames.used.sum(), 10)
        self.assertIn(result.standingOut, frames.mcs_frame_id[[1, 5, 9]].tolist())
        self.assertTrue(any('still displaced' in w for w in result.warnings))

    def testScatteredFrame(self):
        """A frame whose spots scatter has them clipped and does not move the centre."""
        frames, points = makeFrames(LOOP)
        bad = points.mcs_frame_id == frames.mcs_frame_id[5]
        points.loc[bad, ['x', 'y']] += np.random.default_rng(3).normal(0, 3, (bad.sum(), 2))
        result = boresight.measureBoresight(frames, points)
        self.assertGreater(result.frames.nLeftOut[5], bad.sum() / 2)
        self.assertCentre(result, 0.1)

    def testOutlyingSpotsClipped(self):
        """Mismatched spots are clipped and do not move the centre."""
        frames, points = makeFrames(LOOP)
        rng = np.random.default_rng(4)
        wrong = rng.uniform(0, 1, len(points)) < 0.03
        points.loc[wrong, ['x', 'y']] += rng.choice([-1, 1], (wrong.sum(), 2)) * 15
        result = boresight.measureBoresight(frames, points)
        self.assertGreaterEqual(result.nLeftOut, wrong.sum() * 0.9)
        self.assertTrue(result.frames.used.all())
        self.assertCentre(result, 0.1)

    def testFrameWithoutSpots(self):
        """A frame with no identified spot is left out; with fewer than two left, refused."""
        frames, points = makeFrames(LOOP[:4])
        result = boresight.measureBoresight(frames, points[points.mcs_frame_id != frames.mcs_frame_id[1]])
        self.assertFalse(result.frames.used[1])
        self.assertEqual(result.frames.reason[1], 'no identified spot')
        self.assertCentre(result, 0.5)

        with self.assertRaisesRegex(ValueError, 'identified spots'):
            boresight.measureBoresight(frames, points[points.mcs_frame_id == frames.mcs_frame_id[0]])
        with self.assertRaisesRegex(ValueError, 'identified spots'):
            boresight.measureBoresight(frames, points.iloc[:0])

    def testFrameSharingNoPoint(self):
        """A frame whose points are seen in no other frame is refused, naming the frames."""
        frames, points = makeFrames(LOOP[:4])
        points.loc[points.mcs_frame_id == frames.mcs_frame_id[2], 'id'] += 100000
        with self.assertRaisesRegex(ValueError, 'share no point'):
            boresight.measureBoresight(frames, points)

    def testRefused(self):
        """A single rotator position, an unknown angle, several altitudes or no rigid rotation."""
        frames, points = makeFrames([0, 30, 60])
        for column, values, match in (
            ('insrot', [15, 15, 19], 'single rotator position'),
            ('insrot', [-179.5, 179.5, 175], 'single rotator position'),
            ('insrot', [0, 30, np.nan], 'unknown insrot'),
            ('altitude', [60, 60, np.nan], 'unknown insrot'),
            ('altitude', [60, 60, 61], 'altitude'),
        ):
            with self.subTest(column=column, values=values):
                changed = frames.copy()
                changed[column] = values
                with self.assertRaisesRegex(ValueError, match):
                    boresight.measureBoresight(changed, points)

        frames, points = makeFrames(LOOP[::2])
        points.loc[points.mcs_frame_id.isin(frames.mcs_frame_id[::2]), 'x'] += 30
        with self.assertRaisesRegex(ValueError, 'rigid rotation'):
            boresight.measureBoresight(frames, points)

    def testRotatorPositions(self):
        """Angles within SAME_POSITION share a position, across the wrap too."""
        positions = boresight.rotatorPositions([10, 18, 40, -179.8, 175, 45])
        self.assertEqual(positions.tolist(), [0, 0, 1, 2, 2, 1])


if __name__ == '__main__':
    unittest.main()
