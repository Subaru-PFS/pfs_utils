import os
import shutil
import tempfile
import unittest

import numpy as np
from types import SimpleNamespace

from pfs.datamodel import FiberStatus, TargetType
from pfs.utils.fiberids import FiberIds
from pfs.utils.pfsDesignUtils import setFiberStatus


class FakeCalibModel:
    """Minimal stand-in for the cobra calibration product (ics_cobraCharmer PFIDesign).

    setFiberStatus only reads ``status``, ``COBRA_OK_MASK`` and ``FIBER_BROKEN_MASK``,
    so faking those keeps the test free of any pfs_instdata / ics_cobraCharmer dependency.
    """

    COBRA_OK_MASK = np.uint16(0x0001)
    COBRA_BROKEN_PHI_MASK = np.uint16(0x0008)
    FIBER_A_BROKEN = np.uint16(0x0010)
    FIBER_BROKEN_MASK = np.uint16(0x0070)

    def __init__(self, nCobras, brokenFiberIdx, brokenCobraIdx):
        # everything nominal, then clearing the OK bit for the cobras we declare bad.
        self.status = np.full(nCobras, self.COBRA_OK_MASK, dtype='uint16')
        self.status[brokenFiberIdx] = self.FIBER_A_BROKEN
        self.status[brokenCobraIdx] = self.COBRA_BROKEN_PHI_MASK


class SetFiberStatusTestCase(unittest.TestCase):
    """Check pfsDesignUtils.setFiberStatus assigns mutually exclusive statuses,
    in decreasing order of precedence :

    BROKENFIBER = FIBER_BROKEN_MASK
    BLOCKED = BLOCKED & ~FIBER_BROKEN_MASK
    BROKENCOBRA = ~COBRA_OK_MASK & ~(FIBER_BROKEN_MASK | BLOCKED)
    BAD_PSF = BAD_PSF & ~(FIBER_BROKEN_MASK | BLOCKED | BROKENCOBRA)

    The test is self-contained : the calibModel is faked and the ``blocked.csv``
    and ``badPsf.yaml`` tables are written to a temporary ``configRoot``, so it
    needs neither pfs_instdata nor whatever those tables currently release.  The
    only real input is the grand fiber map, which ships with this product.

    setFiberStatus is duck-typed (it only reads ``fiberId``/``targetType`` and
    mutates ``fiberStatus``), so a lightweight stand-in replaces PfsDesign.
    """

    # fibers we will always flag as blocked in the fixture (incl. INSTRM-2965 samples)
    BASE_BLOCKED = sorted({94, 95, 96, 121, 122, 2337, 2364})

    @classmethod
    def setUpClass(cls):
        gfm = FiberIds()
        science = np.asarray(gfm.scienceFiberId)
        isEmpty = science == FiberIds.EMPTY
        isEng = science == FiberIds.ENGINEERING

        # A real pfsDesign covers the non-empty fibers (science cobras + eng).
        keep = ~isEmpty
        cls.fiberId = np.asarray(gfm.fiberId)[keep].astype("int32")
        cls.targetType = np.where(isEng[keep], int(TargetType.ENGINEERING),
                                  int(TargetType.SCIENCE)).astype(int)
        cls.engMask = cls.targetType == int(TargetType.ENGINEERING)

        # A handful of bad cobras, spread over the modules.
        cobraId = gfm.fiberIdToCobraId(cls.fiberId[~cls.engMask])
        nCobras = int(cobraId.max())
        cls.calibModel = FakeCalibModel(nCobras, brokenFiberIdx=[0, 137, 1042],
                                        brokenCobraIdx=[1, 666, 1789, 2000])

        # The raw cobra masks, projected onto the design fibers, engineering fibers carry none.
        fiberBroken = (cls.calibModel.status & cls.calibModel.FIBER_BROKEN_MASK).astype(bool)
        cobraNotOk = ~(cls.calibModel.status & cls.calibModel.COBRA_OK_MASK).astype(bool)
        cls.brokenFiber = np.zeros(len(cls.fiberId), dtype=bool)
        cls.cobraNotOk = np.zeros(len(cls.fiberId), dtype=bool)
        cls.brokenFiber[~cls.engMask] = fiberBroken[cobraId - 1]
        cls.cobraNotOk[~cls.engMask] = cobraNotOk[cobraId - 1]

        # Blocking a broken fiber and a broken cobra as well, to pin the precedence down.
        cls.blockedBrokenFiber = int(cls.fiberId[cls.brokenFiber][0])
        cls.blockedBrokenCobra = int(cls.fiberId[cls.cobraNotOk & ~cls.brokenFiber][0])
        cls.blocked = sorted(set(cls.BASE_BLOCKED) | {cls.blockedBrokenFiber, cls.blockedBrokenCobra})
        cls.blockedMask = np.isin(cls.fiberId, cls.blocked)

        # Same for badPsf : one otherwise good fiber and one broken cobra.
        isGood = ~(cls.brokenFiber | cls.cobraNotOk | cls.blockedMask | cls.engMask)
        cls.badPsfGood = int(cls.fiberId[isGood][0])
        cls.badPsfBrokenCobra = int(cls.fiberId[cls.cobraNotOk & ~cls.brokenFiber & ~cls.blockedMask][0])
        cls.badPsf = sorted({cls.badPsfGood, cls.badPsfBrokenCobra})

        # Build a temp config root with the two fiber tables setFiberStatus reads.
        cls.configRoot = tempfile.mkdtemp(prefix="pfsDesignUtilsTest_")
        fibersDir = os.path.join(cls.configRoot, "fibers")
        os.makedirs(fibersDir)
        blockedSet = set(cls.blocked)
        with open(os.path.join(fibersDir, "blocked.csv"), "w") as fh:
            fh.write("fiberId,b,r,n,status\n")
            for fiberId in cls.fiberId:
                v = "True" if int(fiberId) in blockedSet else "False"
                fh.write(f"{int(fiberId)},{v},{v},{v},{v}\n")
        with open(os.path.join(fibersDir, "badPsf.yaml"), "w") as fh:
            fh.write("fiberId: [%s]\n" % ", ".join(str(fiberId) for fiberId in cls.badPsf))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.configRoot, ignore_errors=True)

    def setUp(self):
        self.design = SimpleNamespace(
            fiberId=self.fiberId.copy(),
            targetType=self.targetType.copy(),
            fiberStatus=np.full(len(self.fiberId), int(FiberStatus.GOOD), dtype=int),
        )
        setFiberStatus(self.design, calibModel=self.calibModel, configRoot=self.configRoot)

    def statusOf(self, fiberId):
        """Return the fiberStatus assigned to that single fiberId."""
        return int(self.design.fiberStatus[self.design.fiberId == fiberId][0])

    def testBrokenFiberMatchesCalibModel(self):
        """BROKENFIBER is exactly FIBER_BROKEN_MASK, nothing overrides it."""
        self.assertTrue(np.array_equal(self.design.fiberStatus == int(FiberStatus.BROKENFIBER),
                                       self.brokenFiber))

    def testBlockedExcludesBrokenFiber(self):
        """BLOCKED is the blocked table minus the broken fibers."""
        self.assertTrue(np.array_equal(self.design.fiberStatus == int(FiberStatus.BLOCKED),
                                       self.blockedMask & ~self.brokenFiber))

    def testBrokenCobraExcludesFiberAndBlocked(self):
        """BROKENCOBRA is ~COBRA_OK_MASK minus the broken fibers and the blocked ones."""
        self.assertTrue(np.array_equal(self.design.fiberStatus == int(FiberStatus.BROKENCOBRA),
                                       self.cobraNotOk & ~(self.brokenFiber | self.blockedMask)))

    def testBrokenFiberWinsOverBlocked(self):
        """A blocked fiber which is also broken stays BROKENFIBER."""
        self.assertEqual(self.statusOf(self.blockedBrokenFiber), int(FiberStatus.BROKENFIBER))

    def testBlockedWinsOverBrokenCobra(self):
        """A blocked fiber whose cobra is not OK stays BLOCKED."""
        self.assertEqual(self.statusOf(self.blockedBrokenCobra), int(FiberStatus.BLOCKED))

    def testBadPsfLosesToBrokenCobra(self):
        """BAD_PSF only applies to fibers which are not flagged otherwise."""
        self.assertEqual(self.statusOf(self.badPsfGood), int(FiberStatus.BAD_PSF))
        self.assertEqual(self.statusOf(self.badPsfBrokenCobra), int(FiberStatus.BROKENCOBRA))

    def testUnblockedFibersNotBlocked(self):
        """No fiber outside the table is marked BLOCKED."""
        outside = ~np.isin(self.design.fiberId, self.blocked)
        self.assertFalse(np.any(self.design.fiberStatus[outside] == int(FiberStatus.BLOCKED)))


if __name__ == "__main__":
    unittest.main()
