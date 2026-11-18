"""The MCS spot flag conventions readers of the opdb and of PFSC files rely on (INSTRM-3109)."""

import unittest

from pfs.utils.datamodel.mcs import MCSSpotFlags


class MCSSpotFlagsTestCase(unittest.TestCase):
    def testValues(self):
        """Each flag is one bit, numbered as the FLAG<n> cards of the SPOTS HDU name them."""
        for bit, flag in enumerate(
            (MCSSpotFlags.NO_SAT_CHECK, MCSSpotFlags.BAD_PHOTOMETRY, MCSSpotFlags.BAD_CENTROID,
             MCSSpotFlags.SATURATED)
        ):
            self.assertEqual(int(flag), 1 << bit)
        self.assertEqual(int(MCSSpotFlags.NONE), 0)

    def testUnflaggedIsZero(self):
        """``mcs_data.flags == 0`` is a checked spot clear of saturation, the value written before the flags."""
        self.assertEqual(MCSSpotFlags(0), MCSSpotFlags.NONE)
        self.assertEqual(str(MCSSpotFlags(0)), "NONE")
        # Negative control: an unchecked spot never reads as clear.
        self.assertNotEqual(int(MCSSpotFlags.NO_SAT_CHECK), 0)

    def testNested(self):
        """A heavily saturated spot carries the three saturation bits, and decodes to their names."""
        saturated = MCSSpotFlags.BAD_PHOTOMETRY | MCSSpotFlags.BAD_CENTROID | MCSSpotFlags.SATURATED
        self.assertEqual(int(saturated), 14)
        self.assertEqual(str(saturated), "BAD_PHOTOMETRY|BAD_CENTROID|SATURATED")
        self.assertFalse(saturated & MCSSpotFlags.NO_SAT_CHECK)


if __name__ == "__main__":
    unittest.main()
