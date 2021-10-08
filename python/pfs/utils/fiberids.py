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
    """

    # When reading in the grand fiber map, missing
    # values for integer fields
    # are assigned this
    MISSING_VALUE = 65535
    EMPTY = MISSING_VALUE - 1
    ENGINEERING = EMPTY - 1

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
                 ('scienceFiberId', 'U4'),  # we change this to int after parsing the strings
                 ('fiberId', 'u2'),
                 ('sunssFiberId', 'U4'),
                 ('mtp_A', 'U15'),
                 ('mtp_C', 'U15'),
                 ('mtp_BA', 'U15'),
                 ('mtp_BC', 'U15'),
                 ]

        # Read in data from grand fiber map text file
        #
        # Note that providing an explicit list of fill values
        # rather than the default so that they can be checked in the unit test.
        # For string fields original value needs to be propagated.
        # To do that, using filling_value of np.nan
        # to instruct np.genfromtxt to just propagate the literal field value.
        self.data = np.genfromtxt(flist[0], dtype=dtype,
                                  filling_values=[self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  self.MISSING_VALUE,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan,
                                                  np.nan],
                                  comments='\\')
        #
        # Build a dtype with the type of scienceFiberId set to int
        #
        dt = self.data.dtype
        ind = [i for i, (k, t) in enumerate(dt.descr) if k == "scienceFiberId"][0]
        dt = dt.descr
        dt[ind] = (dt[ind][0], int)
        dt = np.dtype(dt)
        #
        # handle the emp/ang values (still as str!)
        #
        sfid = self.data["scienceFiberId"]
        sfid = np.where(sfid == "emp", str(self.EMPTY),
                        np.where(sfid == "eng", str(self.ENGINEERING), sfid))

        self.data["scienceFiberId"] = 0  # get rid of invalid literals such as 'eng'
        self.data = self.data.astype(np.dtype(dt))
        self.data["scienceFiberId"] = sfid
        #
        # Add properties
        #
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

        return self.data[['x', 'y']][cobras]

    def fiberIdToMTP(self, fiberIds, pfsConfig=None):
        """Return MTP information for the specified fiberIds

        Args
        ----
        fiberIds : array of 1-indexed fiberIds
        pfsConfig : `pfs.datamodel.PfsConfig`
           Tell us e.g. which fibres go to SuNSS

        Returns
        -------
        an array of ("MTPID", (holes), cobraId) where "holes" are for the A, BA, BC, and C connectors,
        and cobraId is the 0-indexed global cobra ID

        For SuNSS, the "cobra ID" is negative, and its absolute value is the ID in the
        ferrule (for the imaging leg) or 127 + ID (for the diffuse leg)
        """
        if pfsConfig is not None:
            try:
                from pfs.datamodel.pfsConfig import TargetType
            except ImportError:
                raise RuntimeError("You may not specify a pfsConfig file if pfs.datamodel is not setup")

        mtp = []
        for i, fid in zip(range(len(self.fiberId)), self.fiberId):
            if fid in fiberIds:
                holes = []
                for mtp_X in [self.mtp_A, self.mtp_BA, self.mtp_BC, self.mtp_C]:
                    segment, field, spectrograph, hole, cobra = self.splitConnector(mtp_X[i])
                    holes.append(hole)

                cobraId = self.cobraId[i]
                if pfsConfig is not None:
                    ll = pfsConfig.fiberId == fid
                    if sum(ll) == 0:
                        raise RuntimeError(f"fiberId {fid} is not found in the pfsConfig file")
                    tt = TargetType(pfsConfig.targetType[ll][0])
                    if tt in (TargetType.SUNSS_DIFFUSE, TargetType.SUNSS_IMAGING):
                        sid = self.sunssFiberId[i]
                        sunssId = int(sid[1:])
                        if sid[0] == 'd':
                            assert tt == TargetType.SUNSS_DIFFUSE
                            cobraId = -(127 + sunssId)
                        else:
                            assert tt == TargetType.SUNSS_IMAGING
                            cobraId = -sunssId

                mtp.append((f"{segment}-{field}-{spectrograph}", holes, cobraId))

        return mtp
    
    def mtpAToFiberId(self, mtpA, pfsConfig=None):
        """Return ScienceFiberId for the specified mtp_A 
        Args
        ----
        mtpA :  array of 1-indexed of mtp_A
        pfsConfig : `pfs.datamodel.PfsConfig`
           Tell us e.g. which fibres go to SuNSS
        Returns
        -------
        an array of  ("ScienceFiberId")
        """
        if pfsConfig is not None:
            try:
                from pfs.datamodel.pfsConfig import TargetType
            except ImportError:
                raise RuntimeError("You may not specify a pfsConfig file if pfs.datamodel is not setup")
        
        sfib = []
        for usconnect in mtpA:
            for mtp in self.data['mtp_A']:
                if (usconnect+'-' in mtp):
                    sfib.append(self.data[self.data['mtp_A'] == mtp]['scienceFiberId'][0])
            
        return sfib

    
    @staticmethod
    def splitConnector(connectorName):
        """Given a connectorName such as "U3-1-1-2-1" return the fields"""

        if connectorName == '-':
            return None, None, None, None, None

        fields = connectorName.split('-')
        fields[1:] = [int(i) for i in fields[1:]]
        segment, field, spectrograph, hole, cobra = fields

        return segment, field, spectrograph, hole, cobra
