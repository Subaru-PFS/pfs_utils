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
                 ('moduleInFieldId', 'u2'),
                 ('cobraInModuleId', 'u2'),
                 ('boardId', 'U3'),
                 ('x', 'f4'),
                 ('y', 'f4'),
                 ('rad', 'f4'),
                 ('spectrographId', 'u2'),
                 ('fiberHoleId', 'u2'),
                 ('scienceFiberId', 'u2'),
                 ('USCONECId', 'U15'),
                 ('fiberId', 'u2'),
                 ]

        self.data = np.genfromtxt(flist[0], dtype=dtype,
                                  comments='\\')
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
        mod_w = np.where(np.logical_or(self.boardId == f'{moduleId}A',
                                       self.boardId == f'{moduleId}B'))
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
        board = self.boardId[row]
        moduleNum = int(board[:-1])
        cobraInModule = self.cobraInModuleId[row]

        return moduleNum, cobraInModule

    def cobraIdForModulePlusCobra(self, module, cobraInModule):
        """ Return the global cobraId for given (module, cobraInModule) ids. """

        if cobraInModule <= 29:
            boardName = f"{module}A"
        else:
            boardName = f"{module}B"

        w = (self.boardId == boardName) & (self.cobraInModuleId == cobraInModule)
        return self.cobraId[w][0]

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
