import unittest
import numpy as np
import re
from pfs.utils.fiberids import FiberIds
from pfs.utils import constants
from pfs.utils.fibpacking import N_PER_FERRULE


class FiberIdsTestCase(unittest.TestCase):

    MISSING_VALUE = FiberIds.MISSING_VALUE

    # Derive the total number of slit holes in the PFS
    TOTAL_NUMBER_SLIT_HOLES = (constants.N_SPECTROGRAPHS *
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
        self.assertEqual(fbi.data.shape[0], self.TOTAL_NUMBER_SLIT_HOLES - 6)

        # Check properties are accessible
        for name in fbi.data.dtype.names:
            getattr(fbi, name)

        # Check cobraIds
        cobraId = fbi.cobraId[fbi.cobraId != self.MISSING_VALUE]
        self.assertTrue(len(np.unique(cobraId)) == len(cobraId))
        self.assertEqual(min(cobraId), 1)
        self.assertEqual(max(cobraId), 2394)

        # Check fieldId
        fieldId = fbi.fieldId[fbi.fieldId != self.MISSING_VALUE]
        self.assertEqual(min(fieldId), 1)
        self.assertEqual(max(fieldId), constants.N_FIELDS_PFI)

        # Check cobraInFieldId
        print(f'fbi.cobraInFieldId {fbi.cobraInFieldId}')
        cobraInFieldId = fbi.cobraInFieldId[fbi.cobraInFieldId
                                            != self.MISSING_VALUE]
        self.assertEqual(min(cobraInFieldId), 1)
        self.assertEqual(max(cobraInFieldId),
                         constants.N_MODULES_PER_FIELD *
                         constants.N_COBRAS_PER_MODULE)

        # Check cobraModuleId
        cobraModuleId = fbi.cobraModuleId[fbi.cobraModuleId
                                          != self.MISSING_VALUE]
        self.assertEqual(min(cobraModuleId), 1)
        self.assertEqual(max(cobraModuleId),
                         constants.N_FIELDS_PFI *
                         constants.N_MODULES_PER_FIELD)

        # Check moduleInFieldId
        moduleInFieldId = fbi.moduleInFieldId[fbi.moduleInFieldId
                                              != self.MISSING_VALUE]
        self.assertEqual(min(moduleInFieldId), 1)
        self.assertEqual(max(moduleInFieldId), constants.N_MODULES_PER_FIELD)

        # Check cobraInModuleId
        cobraInModuleId = fbi.cobraInModuleId[fbi.cobraInModuleId
                                              != self.MISSING_VALUE]
        self.assertEqual(min(cobraInModuleId), 1)
        self.assertEqual(max(cobraInModuleId), constants.N_COBRAS_PER_MODULE)

        # Check that (cobraModuleId, cobraInModuleId) is unique
        for module in range(constants.N_FIELDS_PFI
                            * constants.N_MODULES_PER_FIELD):
            for cobraInModule in range(constants.N_COBRAS_PER_MODULE):
                cobraIdsInCombo = cobraId[np.logical_and(
                                          cobraModuleId == module,
                                          cobraInModuleId == cobraInModule)]
                self.assertEqual(len(np.unique(cobraIdsInCombo)),
                                 len(cobraIdsInCombo),
                                 msg=f'For module={module} '
                                     f'and cobraInModule={cobraInModule}: '
                                     f'cobraIds are not unique. '
                                     f'cobraIds = {cobraIdsInCombo}')

        # Check cobraInFieldId in more detail
        expectedCobraInFieldId = ((moduleInFieldId-1) *
                                  constants.N_COBRAS_PER_MODULE +
                                  cobraInModuleId)
        np.testing.assert_array_equal(cobraInFieldId, expectedCobraInFieldId)

        # Check cobraInBoardId
        cobraInBoardId = fbi.cobraInBoardId[fbi.cobraInBoardId
                                            != self.MISSING_VALUE]
        self.assertEqual(min(cobraInBoardId), 1)
        self.assertTrue(max(cobraInBoardId)
                        in [23, 29])

        # Check boardId
        boardId = fbi.boardId[fbi.boardId
                              != self.MISSING_VALUE]
        self.assertEqual(min(boardId), 1)
        self.assertEqual(max(boardId),
                         constants.N_COBRAS_PER_MTP *
                         constants.N_MTP_FERRULES)

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
        self.assertEqual(max(cobraModuleIdInMtpA), constants.N_COBRAS_PER_MTP)
        self.assertEqual(min(cobraModuleIdInMtpB), 1)
        self.assertEqual(max(cobraModuleIdInMtpB), constants.N_COBRAS_PER_MTP)

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
        self.assertEqual(max(fbi.fiberId), self. TOTAL_NUMBER_SLIT_HOLES)

        # Check SuNSS fiber IDs
        sunssImagingFibers = []
        sunssDiffuseFibers = []
        for sfi in fbi.sunssFiberId[fbi.sunssFiberId != 'emp']:

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
                         N_PER_FERRULE)
        self.assertEqual(min(sunssDiffuseFibers), 1)
        self.assertEqual(max(sunssDiffuseFibers),
                         N_PER_FERRULE)

        # Check MTP values
        for field, mtps in zip(['A', 'PC', 'BA', 'BC'],
                               [fbi.mtp_A, fbi.mtp_C,
                                fbi.mtp_BA, fbi.mtp_BC]):
            for mtp in mtps:
                if mtp == '-':
                    continue
                self.assertTrue(self.MTP_REGEX.match(mtp),
                                msg=f"MTP value '{mtp}'"
                                    f" in column 'mtp_{field}'"
                                    " does not match expected pattern.")
