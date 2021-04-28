import unittest
import numpy as np
import re
from pfs.utils.fiberids import FiberIds
from pfs.utils import constants


class FiberIdsTestCase(unittest.TestCase):

    # Derive the total number of fibers in the PFS
    TOTAL_NUMBER_FIBERS = (constants.N_SPECTROGRAPHS *
                           constants.FIBERS_PER_SPECTROGRAPH)

    MTP_REGEX = re.compile(r'(U\d|EN|D\d)-(\d)-(\d)-(\d+)-(x|\d+)')
    
    def testCreation(self):
        """Checks that FiberId object is correctly
        instantiated from grand fiber map
        and basic sanity checks are applied
        to accessor methods.
        """

        fbi = FiberIds()

        # Check number of columns in grand fiber map are as expected
        self.assertEqual(len(fbi.data.dtype), 21)

        # And the number of rows
        # FIXME: currently this is 6 fewer than expected.
        # This will be fixed in INSTRM-1260
        # self.assertEqual(fbi.data.shape[0], TOTAL_NUMBER_FIBERS)
        self.assertEqual(fbi.data.shape[0], self.TOTAL_NUMBER_FIBERS - 6)

        # Check properties are accessible
        for name in fbi.data.dtype.names:
            getattr(fbi, name)

        # Check cobraIds
        cobraId = fbi.cobraId[fbi.cobraId != FiberIds.MISSING_VALUE_UINT16
]
        self.assertEqual(min(cobraId), 1)
        self.assertEqual(max(cobraId), constants.N_SCIENCE_FIBERS)

        # Check fieldId
        fieldId = fbi.fieldId[fbi.fieldId != FiberIds.MISSING_VALUE_UINT16
]
        self.assertEqual(min(fieldId), 1)
        self.assertEqual(max(fieldId), constants.N_FIELDS_PFI)

        # Check cobraInFieldId
        cobraInFieldId = fbi.cobraInFieldId[fbi.cobraInFieldId !=
                                            FiberIds.MISSING_VALUE_UINT16
]
        self.assertEqual(min(cobraInFieldId), 1)
        self.assertEqual(max(cobraInFieldId),
                         constants.N_MODULES_PER_FIELD *
                         constants.N_COBRAS_PER_MODULE)

        # Check cobraModuleId
        cobraModuleId = fbi.cobraModuleId[fbi.cobraModuleId !=
                                          FiberIds.MISSING_VALUE_UINT16
]
        self.assertEqual(min(cobraModuleId), 1)
        self.assertEqual(max(cobraModuleId),
                         constants.N_FIELDS_PFI *
                         constants.N_MODULES_PER_FIELD)

        # Check moduleInFieldId
        moduleInFieldId = fbi.moduleInFieldId[fbi.moduleInFieldId !=
                                              FiberIds.MISSING_VALUE_UINT16
]
        self.assertEqual(min(moduleInFieldId), 1)
        self.assertEqual(max(moduleInFieldId), constants.N_MODULES_PER_FIELD)

        # Check cobraInModuleId
        cobraInModuleId = fbi.cobraInModuleId[fbi.cobraInModuleId !=
                                              FiberIds.MISSING_VALUE_UINT16
]
        self.assertEqual(min(cobraInModuleId), 1)
        self.assertEqual(max(cobraInModuleId), constants.N_COBRAS_PER_MODULE)

        # Check cobraInFieldId in more detail
        #  Using 57*(moduleInFieldId-1)+cobraInModuleId
        expectedCobraInFieldId = (moduleInFieldId-1)*57 + cobraInModuleId
        np.testing.assert_array_equal(cobraInFieldId, expectedCobraInFieldId)

        # Check cobraInBoardId
        cobraInBoardId = fbi.cobraInBoardId[fbi.cobraInBoardId !=
                                            FiberIds.MISSING_VALUE_UINT16
]
        self.assertEqual(min(cobraInBoardId), 1)
        self.assertEqual(max(cobraInBoardId), 29)

        # Check boardId
        boardId = fbi.boardId[fbi.boardId != FiberIds.MISSING_VALUE_UINT16
]
        self.assertEqual(min(boardId), 1)
        self.assertEqual(max(boardId), 84)

        # Check cobraModuleIdInMTP
        cobraModuleIdInMtpA = []
        cobraModuleIdInMtpB = []
        expectedMtp = ['A', 'B']
        for cm in fbi.cobraModuleIdInMTP:

            if cm == '-':
                continue

            mtp = cm[-1]
            self.assertTrue(mtp in expectedMtp,
                            f"MTP '{mtp}' of cobraModuleInMtp '{cm}'"
                            f" is not one of the expected ones of"
                            f" {expectedMtp}")
            mtpId = int(cm[:-1])
            if mtp == 'A':
                cobraModuleIdInMtpA.append(mtpId)
            else:
                cobraModuleIdInMtpB.append(mtpId)

        self.assertEqual(min(cobraModuleIdInMtpA), 1)
        self.assertEqual(max(cobraModuleIdInMtpA), 42)
        self.assertEqual(min(cobraModuleIdInMtpB), 1)
        self.assertEqual(max(cobraModuleIdInMtpB), 42)

        # Check x, y
        # Assume PFI region is a hexagon with vertices
        # [(0, 224), (194, 112),
        # (194, -112), (0, -224),
        # (-194, -112), (-194, 112)]
        x = fbi.x[~np.isnan(fbi.x)]
        y = fbi.y[~np.isnan(fbi.y)]

        # Map coordinates to positive quadrant to simplify check
        inRegion = np.logical_and(np.absolute(x) < 194.0,
                                  np.absolute(y) <= -(112.0/194.0)*x + 224.0)
        badPoints = [(x_i, y_i)
                     for x_i, y_i in zip(x[~inRegion], 
                                         y[~inRegion])]
        self.assertTrue(np.all(inRegion),
                        msg="Points with coordinates"
                            f"{badPoints}"
                            f" are outside PFI hexagon."
                        )

        # Check rad (rad^2 = x^2 + y^2)
        rad = fbi.rad[~np.isnan(fbi.rad)]
        self.assertEqual(len(rad), len(x))
        self.assertEqual(len(rad), len(y))
        # Note: tolerance for this test is very low,
        # but the precision of x and y are very low
        np.testing.assert_allclose(np.square(rad),
                                   np.square(x) + np.square(y),
                                   rtol=1.0, atol=0)

        # Check spectrographId
        self.assertEqual(min(fbi.spectrographId), 1)
        self.assertEqual(max(fbi.spectrographId),
                         constants.N_SPECTROGRAPHS)

        # Check fiberId
        self.assertEqual(min(fbi.fiberId), 1)
        self.assertEqual(max(fbi.fiberId), self.TOTAL_NUMBER_FIBERS)

        # Check SuNSS fiber IDs
        sunssImagingFibers = []
        sunssDiffuseFibers = []
        for sfi in fbi.sunssFiberId:

            if sfi == 'emp':
                continue

            instrument = sfi[0]

            expectedSunssInstruments = ['i', 'd']
            self.assertTrue(instrument in expectedSunssInstruments,
                            f"SuNSS instrument '{instrument}'"
                            f" is not one of the expected ones of"
                            f" {expectedSunssInstruments}")

            fiberId = int(sfi[1:])
            if instrument == 'i':
                sunssImagingFibers.append(fiberId)
            else:
                sunssDiffuseFibers.append(fiberId)

        self.assertEqual(min(sunssImagingFibers), 1)
        self.assertEqual(max(sunssImagingFibers),
                         constants.N_SUNSS_FIBERS_INST)
        self.assertEqual(min(sunssDiffuseFibers), 1)
        self.assertEqual(max(sunssDiffuseFibers),
                         constants.N_SUNSS_FIBERS_INST)

        # Check MTP values
        for field, mtps in zip(['A', 'PC', 'BA', 'BC'],
                               [fbi.mtp_A, fbi.mtp_PC,
                                fbi.mtp_BA, fbi.mtp_BC]):
            for mtp in mtps:
                if mtp == '-':
                    continue
                self.assertTrue(self.MTP_REGEX.match(mtp),
                                msg=f"MTP value '{mtp}'"
                                    f" in column 'mtp_{field}'"
                                    " does not match expected pattern.")
