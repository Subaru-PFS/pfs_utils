import glob
import numpy as np
import os

import eups

class FiberIds(object):
    """ Track all fiber ids and positions.

    This comes from JEG's reference file. We simply load the entire thing into a
    numpy structured array and provide accessors and mappers.

    See the cobra.pdf file in the data directory for details.

    A summary of the fields.
    For the PFI:
     - cobraId: 1..2394
     - fieldId: 1..3
         Three fields, rotated by 120deg. The outer row of field 2 is numbered slightly differently.
     - cobraInFieldId: 1..798
     - moduleInFieldId: 1..14
     - cobraInModuleId: 1..57
     - moduleId: 1A..42B
         A and B refer to the two boards which respectively control the even- and odd- cobras.
     - x,y mm from center
     - r mm from center

     - spectrograph: 1..4
     - slitHoleId: 1..650
        Note that there are more slit holes than cobras.
     - scienceFiberId: 1..2394

     - USCONEC ID: This encodes the USCONEC and USCONEC hole ids.
     - fiberId: a unique identifier for each fiber (both science and engineering; 1..2604).

    Yes, ids are 1-indexed. Water under the bridge.

    Todo
    ----

    Add proper module ID (no board)
    Add methods as needed
    cobraId vs. scienceFiberId? Why two, really?
    fiducials? Separate class, probably.
    Identify engineering/blank holes?
    """

    # When reading in the grandfibermap,
    # missing values for unsigned 16-bit integer types
    # are assigned the following
    MISSING_VALUE_UINT16 = 65535

    def __init__(self, path=None):
        if path is None:
            path = os.path.join(eups.productDir('PFS_UTILS'),
                                'data', 'fiberids')
        self.filepath = 'unset'
        self.load(path=path)

    def __str__(self):
        return f'FiberIds(path={self.filepath})'

    def load(self, path):
        """ Load datafile.

        Args
        ----
        path : str
           The directory continaing the data file(s).

        Besides simply loading the data, creates direct accessors for all the fields.
        """

        flist = glob.glob(os.path.join(path, 'grandfibermap*.txt'))
        if len(flist) != 1:
            raise RuntimeError("no unique fibermap file: %s" % (flist))
        self.filepath = flist[0]

        dtype = [('cobraId', 'u2'),
                 ('fieldId', 'u2'),
                 ('cobraInFieldId', 'u2'),
                 ('cobraModuleId', 'u2'),
                 ('moduleInFieldId', 'u2'),
                 ('cobraInModuleId', 'u2'),
                 ('cobraInBoardId', 'u2'),
                 ('boardId', 'u2'),
                 ('cobraModuleIdInMTP', 'U3'),
                 ('x', 'f4'),
                 ('y', 'f4'),
                 ('rad', 'f4'),
                 ('spectrographId', 'u2'),
                 ('fiberHoleId', 'u2'),
                 ('scienceFiberId', 'u2'),
                 ('fiberId', 'u2'),
                 ('sunssFiberId', 'U4'),
                 ('mtp_A', 'U15'),
                 ('mtp_PC', 'U15'),
                 ('mtp_BA', 'U15'),
                 ('mtp_BC', 'U15'),
                 ]

        self.data = np.genfromtxt(flist[0], dtype=dtype,
                                  comments='\\',
                                  missing_values='-',
                                  filling_values=[self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  self.MISSING_VALUE_UINT16,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan])
        for name in [d[0] for d in dtype]:
            def _fetchOne(self, name=name):
                return self.data.__getitem__(name)

            setattr(self.__class__, name,
                    property(_fetchOne))

    def cobrasForModule(self, moduleId):
        """ Return the indices of the cobras in a given module.

        Args
        ----
        moduleId : 1..42

        Returns
        -------
        idx : ndarray of ints
          The 0-based indices of the cobras.
        """

        if moduleId < 1 or moduleId > 42:
            raise ValueError(f'moduleId ({moduleId}) must be 1..42')
        mod_w = np.where(np.logical_or(self.cobraModuleIdInMTP == f'{moduleId}A',
                                       self.cobraModuleIdInMTP == f'{moduleId}B'))
        return mod_w

    def cobrasForSpectrograph(self, spectrographId, holeIds=None):
        """ Return the indices of all the cobras for a given spectrograph.

        Args
        ----
        spectrographId : 1..4
        holeIds : list of ints, optional
           The holeIds along the slit.

        Returns
        -------
        idx : ndarray of ints
          The 0-based indices of the cobras.
        """

        if spectrographId < 1 or spectrographId > 4:
            raise ValueError(f'spectrographId ({spectrographId}) must be 1..4')

        sCobras = self.data[self.data['spectrographId'] == spectrographId]

        if holeIds is not None:
            fh_w = np.argsort(sCobras['fiberHoleId'])
            res_w = np.searchsorted(sCobras['fiberHoleId'], holeIds, sorter=fh_w)
            sCobras = sCobras[fh_w][res_w]
            if not np.all(np.equal(sCobras['fiberHoleId'], holeIds)):
                raise ValueError("All holeIds not uniquely found.")

        return sCobras['cobraId']-1

    def moduleNumsForCobra(self, cobraId):
        row = np.where(self.cobraId == cobraId)[0][0]
        moduleNum = self.cobraModuleId[row]
        cobraInModule = self.cobraInModuleId[row]

        return moduleNum, cobraInModule

    def cobraIdForModulePlusCobra(self, module, cobraInModule):
        """ Return the global cobraId for given (module, cobraInModule) ids. """
        result = self.cobraId[np.logical_and(self.cobraModuleId == module,
                              self.cobraInModuleId == cobraInModule)]
        if len(result) != 1:
            raise ValueError(f"No unique cobraId found for "
                             f"module={module} and "
                             f"cobraInModule={cobraInModule}."
                             f"Got following results: {result}")
        return result[0]

    def xyForCobras(self, cobras):
        """ Return the (x,y) positions of the given cobras.

        Args
        ----
        cobras : 0-indexed ints

        Returns
        -------
        x,y : ndarray of positions on the PFI.
          In mm.
        """

        return self.data[['x','y']][cobras]
