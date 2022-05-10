import unittest
import pfs.utils.coordinates.transform as transformUtils

import numpy as np
import pandas as pd


class TransformTestCase(unittest.TestCase):


    def testMcsPfiTransform(self):
        """
        """
        np.random.seed(123)

        npoints = 10
        mcs_x_pix = np.random.randint(1, 8690, npoints)
        mcs_y_pix = np.random.randint(1, 5776, npoints)

        pt = transformUtils.fromCameraName('rmod', altitude=45, insrot=45.)

        pfi_x_mm, pfi_y_mm = pt.mcsToPfi(mcs_x_pix, mcs_y_pix)
        fiducialId = range(npoints)
        mcs_data = pd.DataFrame(list(zip(mcs_x_pix, mcs_y_pix)),
                                columns=['mcs_center_x_pix', 'mcs_center_y_pix'])
        fiducials = pd.DataFrame(list(zip(pfi_x_mm, pfi_y_mm, fiducialId)),
                                columns=['x_mm', 'y_mm', 'fiducialId'])
        pt.updateTransform(mcs_data, fiducials)

        pfi_x_mm_actual, pfi_y_mm_actual = pt.mcsToPfi(mcs_x_pix, mcs_y_pix)
        np.testing.assert_array_almost_equal(pfi_x_mm_actual, pfi_x_mm, 8)
        np.testing.assert_array_almost_equal(pfi_y_mm_actual, pfi_y_mm, 8)

if __name__ == '__main__':
    unittest.main()
