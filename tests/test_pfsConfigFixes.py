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
        self.assertTrue((d.fiberStatus == FiberStatus.GOOD).all())


if __name__ == "__main__":
    unittest.main()
