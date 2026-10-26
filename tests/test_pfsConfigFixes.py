import unittest
from types import SimpleNamespace

import numpy as np

from pfs.datamodel.pfsConfig import FiberStatus
from pfs.utils.pfsConfigFixes import fixPfsConfig


def design(visit, nFibers=2604):
    return SimpleNamespace(
        visit=visit,
        fiberId=np.arange(1, nFibers + 1),
        pfiCenter=np.full((nFibers, 2), -1.0),
        fiberStatus=np.full(nFibers, FiberStatus.GOOD, dtype=int),
    )


class FixNight20260903TestCase(unittest.TestCase):
    """The MCS frame after the blind move on 2026-09-03/04 is replaced by
    iteration 7."""

    def testSwappedFibreRestored(self):
        # visit0 148876: fiberId 1147 took a neighbour's spot on that frame and
        # was NOTCONVERGED; iteration 7 has it at its own position, GOOD.
        d = design(148876)
        fixPfsConfig(d)
        i = 1147 - 1
        self.assertNotEqual(d.pfiCenter[i, 0], -1.0)
        self.assertEqual(d.fiberStatus[i], FiberStatus.GOOD)

    def testNotConvergedAtIterationSevenStaysSo(self):
        # cobra 820 (fiberId 2007) was also swapped, but iteration 7 judges it
        # off target: its position is restored and the verdict is NOTCONVERGED.
        d = design(148876)
        fixPfsConfig(d)
        i = 2007 - 1
        self.assertNotEqual(d.pfiCenter[i, 0], -1.0)
        self.assertEqual(d.fiberStatus[i], FiberStatus.NOTCONVERGED)

    def testSpsVisitsReadTheSameFix(self):
        a, b = design(148876), design(148878)
        fixPfsConfig(a)
        fixPfsConfig(b)
        np.testing.assert_array_equal(a.pfiCenter, b.pfiCenter)
        np.testing.assert_array_equal(a.fiberStatus, b.fiberStatus)

    def testEveryAffectedConfigIsCovered(self):
        # the ten configurations of that night whose cobra_target carries
        # the extra frame
        for visit in (148840, 148864, 148867, 148870, 148876,
                      148891, 148894, 148897, 148900, 148903):
            d = design(visit)
            fixPfsConfig(d)
            self.assertTrue((d.pfiCenter != -1.0).any(), visit)

    def testSubsetCarriesOnlyItsOwnFibres(self):
        # a per-spectrograph subset passes through the fix too
        d = design(148876)
        d.fiberId = d.fiberId[:600]
        d.pfiCenter = d.pfiCenter[:600]
        d.fiberStatus = d.fiberStatus[:600]
        fixPfsConfig(d)
        self.assertEqual(len(d.fiberId), 600)

    def testUnaffectedVisitIsUntouched(self):
        d = design(148849)
        fixPfsConfig(d)
        self.assertTrue((d.pfiCenter == -1.0).all())
        self.assertTrue(np.isin(d.fiberStatus, [FiberStatus.GOOD, FiberStatus.BROKENCOBRA]).all())


class FixBrokenCobrasTestCase(unittest.TestCase):
    """Cobras with a broken motor are BROKENCOBRA from visit 109273 on."""

    def status(self, d, fiberId):
        return d.fiberStatus[fiberId - 1]

    def testClearedCalibrationIsFlagged(self):
        # visit 125745 (2025-05): the calibration in use had cleared fiberId 84
        d = design(125745)
        fixPfsConfig(d)
        self.assertEqual(self.status(d, 84), FiberStatus.BROKENCOBRA)
        self.assertEqual(self.status(d, 100), FiberStatus.GOOD)

    def testNotConvergedIsFlagged(self):
        # run 24: fiberId 2184 and 745 were NOTCONVERGED or GOOD
        d = design(130000)
        d.fiberStatus[2184 - 1] = FiberStatus.NOTCONVERGED
        fixPfsConfig(d)
        self.assertEqual(self.status(d, 2184), FiberStatus.BROKENCOBRA)
        self.assertEqual(self.status(d, 745), FiberStatus.BROKENCOBRA)

    def testOtherStatusesTakePrecedence(self):
        d = design(125745)
        d.fiberStatus[2344 - 1] = FiberStatus.BLOCKED
        d.fiberStatus[84 - 1] = FiberStatus.BROKENFIBER
        fixPfsConfig(d)
        self.assertEqual(self.status(d, 2344), FiberStatus.BLOCKED)
        self.assertEqual(self.status(d, 84), FiberStatus.BROKENFIBER)

    def testAppliedAfterRun29(self):
        # fixRun29 blocks fiberId 2344; fiberId 2184 still becomes BROKENCOBRA
        d = design(144600)
        fixPfsConfig(d)
        self.assertEqual(self.status(d, 2344), FiberStatus.BLOCKED)
        self.assertEqual(self.status(d, 2184), FiberStatus.BROKENCOBRA)

    def testOutsideTheRangeIsUntouched(self):
        for visit in (109272, 152449):
            d = design(visit)
            fixPfsConfig(d)
            self.assertTrue((d.fiberStatus == FiberStatus.GOOD).all(), visit)

    def testSubsetCarriesOnlyItsOwnFibres(self):
        d = design(125745)
        d.fiberId = d.fiberId[651:1302]
        d.pfiCenter = d.pfiCenter[651:1302]
        d.fiberStatus = d.fiberStatus[651:1302]
        fixPfsConfig(d)
        flagged = d.fiberId[d.fiberStatus == FiberStatus.BROKENCOBRA]
        np.testing.assert_array_equal(flagged, [745, 950, 963, 1003, 1214, 1217, 1231, 1251])


if __name__ == "__main__":
    unittest.main()
