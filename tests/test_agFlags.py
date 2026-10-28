"""The AG flag conventions readers of the opdb rely on (INSTRM-3038)."""

import unittest

from pfs.utils.datamodel.ag import GuideOffsetFlag, SourceDetectionFlags, SourceMatchingFlags


class AgFlagsTestCase(unittest.TestCase):
    def testGuideOffsetValues(self):
        """``agc_guide_offset.mask`` values written since 2025-09-11 keep their meaning."""
        self.assertEqual(int(GuideOffsetFlag.OK), 0)
        self.assertEqual(int(GuideOffsetFlag.INVALID_OFFSET), 1)
        self.assertEqual(int(GuideOffsetFlag.TOO_FEW_STARS), 2)

    def testTooFewStarsIsInvalid(self):
        """ics_agActor writes TOO_FEW_STARS with INVALID_OFFSET, so ``mask & INVALID_OFFSET`` finds it."""
        tooFew = GuideOffsetFlag.INVALID_OFFSET | GuideOffsetFlag.TOO_FEW_STARS
        self.assertEqual(int(tooFew), 3)
        self.assertTrue(tooFew & GuideOffsetFlag.INVALID_OFFSET)
        # Negative control: the bits are distinct, so TOO_FEW_STARS alone would read as used.
        self.assertFalse(GuideOffsetFlag.TOO_FEW_STARS & GuideOffsetFlag.INVALID_OFFSET)
        self.assertEqual(str(tooFew), "INVALID_OFFSET|TOO_FEW_STARS")
        self.assertEqual(str(GuideOffsetFlag(0)), "OK")

    def testGoodMatchIsOne(self):
        """``agc_match.flags == 1`` has meant a valid match since the column was first written."""
        self.assertEqual(int(SourceMatchingFlags.GOOD_MATCH), 1)
        # Negative control: no other reason shares the bit, so a bad match never reads as 1.
        for flag in (
            SourceMatchingFlags.NO_MATCH,
            SourceMatchingFlags.BAD_RESIDUAL,
            SourceMatchingFlags.UNUSED_MULTI_MATCH,
        ):
            self.assertNotEqual(int(flag), 1)
            self.assertEqual(flag & SourceMatchingFlags.GOOD_MATCH, 0)
        self.assertNotEqual(int(SourceMatchingFlags.BAD_RESIDUAL | SourceMatchingFlags.UNUSED_MULTI_MATCH), 1)

    def testRightIsOne(self):
        """``agc_data.flags & 1`` is the glass-free half of the detector."""
        self.assertEqual(int(SourceDetectionFlags.RIGHT), 1)
        self.assertEqual(SourceDetectionFlags.BAD_DETECTION & SourceDetectionFlags.RIGHT, 0)

    def testStr(self):
        self.assertEqual(str(SourceMatchingFlags.GOOD_MATCH), "GOOD_MATCH")
        self.assertEqual(
            str(SourceMatchingFlags.BAD_RESIDUAL | SourceMatchingFlags.UNUSED_MULTI_MATCH),
            "BAD_RESIDUAL|UNUSED_MULTI_MATCH",
        )
        self.assertEqual(str(SourceMatchingFlags.NONE), "NONE")


if __name__ == "__main__":
    unittest.main()
