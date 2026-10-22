"""The AG flag conventions readers of the opdb rely on (INSTRM-3038)."""

import unittest

from pfs.utils.datamodel.ag import SourceDetectionFlags, SourceMatchingFlags


class AgFlagsTestCase(unittest.TestCase):
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
