import importlib
from importlib import reload

import opscore.utility.sdss3logging
import logging
import pathlib
import time

from . import spectroIds
reload(spectroIds)

logger = logging.getLogger('butler')

defaultDataRoot = pathlib.Path("/data")

class Butler(object):
    def __init__(self, dataRoot=None, configRoot=None, specIds=None):
        """A data and configuration manager, inspired by the LSST/DRP butler.

        This provides only the most minimal functionality required by the PFS ICS software.

        Args
        ----
        dataRoot : path-like
            The root of the data directory tree. Defaults to /data
        configRoot : path-like
            The root of the configuration directory tree.
            Defaults to $PFS_INSTDATA_DIR/data
        specIds : `pfs.utils.spectroIds.SpectroIds`
            Contains our identity: site, arm, module, etc.
            Usually created dynamically, from the hostname.
        """

        self.logger = logging.getLogger('butler')
        self.logger.setLevel(logging.DEBUG)

        self.dataRoot = pathlib.Path(dataRoot) if dataRoot is not None else defaultDataRoot
        if configRoot is not None:
            self.configRoot = pathlib.Path(configRoot)
        else:
            import eups

            eupsEnv = eups.Eups()
            eupsProd = eupsEnv.findSetupProduct('pfs_instdata')

            if eupsProd is None:
                raise ValueError("either configRoot must be passed in "
                                 "or the pfs_instdata product must be setup.")
            self.configRoot = pathlib.Path(eupsProd.dir) / "data"
        self._loadMaps(specIds)

    def _loadMaps(self, specIds=None):
        """Load the definitions of the maps which we can resolve. """

        if specIds is None:
            specIds = spectroIds.SpectroIds()

        from . import butlerMaps
        reload(butlerMaps)

        self.dataMap = butlerMaps.dataMap
        self.configMap = butlerMaps.configMap

        self.dataMap['dataRoot'] = dict(template=str(self.dataRoot))
        self.configMap['configRoot'] = dict(template=str(self.configRoot))

        self.addDict = specIds.idDict

        self.logger.debug(f'loaded butler from {self.configMap}, for {self.dataMap}')

    def addMaps(self, configMapDict=None, dataMapDict=None):
        """Add additional maps,

        Args
        ----
        configMapDict : dict()
        dataMapDict : dict()
          Two dictionaries to update the internal mapping dictionaries with.
          NOTE: this *overwrites* any values in the existing maps.
        """

        if configMapDict is not None:
            self.configMap.update(configMapDict)
        if dataMapDict is not None:
            self.dataMap.update(dataMapDict)

    def addKeys(self, addDict):
        """Add keys to the dict used to evaluate butler templates.
        
        Args
        ----
        addDict : dict
          Dictionary to _update_ our internal keys with.
          NOTE: this *overwrites* any values in the existing dict.
        """

        self.addDict.update(addDict)
        
    def getKnownDataMaps(self):
        return sorted(self.dataMap.keys())

    def getKnownConfigMaps(self):
        return sorted(self.configMap.keys())

    def getKnownMaps(self):
        return sorted(self.getKnownConfigMaps() + self.getKnownDataMaps())

    def getPfsDay(self, forTime=None):
        """ Get the PFS 'day', which determines where data gets stored.

        We use GMT: day rollover there is midday at Subaru.
        """

        if forTime is None:
            forTime = time.time()
        forTime = time.gmtime(forTime) # minus 86400?
        day = time.strftime('%Y%m%d', forTime)

        return day

    def _addInternalKeys(self, idDict):
        """ Add keys for the site and SPS ids, etc. Does not overwrite them if already set. """

        if idDict is None:
            idDict = dict()
        else:
            idDict = idDict.copy()

        if 'pfsDay' not in idDict:
            idDict['pfsDay'] = self.getPfsDay()
        for k in self.addDict.keys():
            if k not in idDict:
                idDict[k] = self.addDict[k]

        return idDict

    def getTray(self, objectType):
        """ Return an available configuration dictionary.

        Args
        ----
        objectType : `str`
           The type of the object we are looking for.

        Returns
        -------
        dict : "tray" of properties for the object type
        pathlib.Path : root to search under

        """

        try:
            tray = self.configMap[objectType]
            root = self.configRoot
        except KeyError:
            try:
                tray = self.dataMap[objectType]
                root = self.dataRoot
            except KeyError:
                raise KeyError(f"unknown data or configuration type: {objectType}")

        return tray, root

    def getPath(self, objectType, idDict=None, noJoinRoot=False):
        """ Return the path to an object.

        Args
        ----
        objectType : `str`
           The type of the object we are looking for.
        idDict : `dict`
           The type-specific keys which identify the specific object
           we are looking for.
        noJoinRoot : bool
           Do not append the template to the root, but return both

        Returns
        -------
        pathlib.Path : fully resolved path.

        """

        tray, root = self.getTray(objectType)
        try:
            template = tray['template']
        except NameError:
            raise KeyError(f"no path template available for: {objectType}")

        idDict = self._addInternalKeys(idDict)
        try:
            path = eval(f'f"{template}"', globals(), idDict)
        except NameError as e:
            raise KeyError(f"the path for {objectType} ({template}) could not be resolved: {e}")

        if noJoinRoot:
            return root, path
        else:
            return root / path

    def search(self, objectType, idDict=None):
        root, globPattern = self.getPath(objectType, idDict, noJoinRoot=True)

        matches = root.glob(globPattern)
        return sorted(matches)

    def getFromPath(self, objectType, path):
        """ Load an object given its path
        Args
        ----
        objectType : `str`
           The type of the object we are looking for.
        path : path-like
           Where to load from.

        Returns
        -------
        object : the loaded object


        The loader is searched in the following order:
        - if there is a 'loader' key in the tray, that is called with the path as the sole argument.
        - if there is a 'loaderModule' string key, that is imported and the 'load' attribute called as above.
        - else if nothing blew up the resolved path is returned as a string.
        """

        tray, _ = self.getTray(objectType)
        try:
            loader = tray['loader']
        except KeyError:
            try:
                loaderModuleName = tray['loaderModule']
            except KeyError:
                return path

            if not isinstance(loaderModuleName, str):
                raise ValueError(f'do not know what to do with a "loaderModule" name of {loaderModuleName}')

            try:
                loaderModule = importlib.import_module(loaderModuleName)
            except:
                raise

            try:
                loader = getattr(loaderModule, 'load')
            except:
                raise KeyError(f'no "load" attribute in {loaderModuleName}')

        self.logger.debug(f'loading {objectType} from {path}, using {loader}')
        return loader(path)

    def get(self, objectType, idDict=None):
        """ Find and load an object.

        Args
        ----
        objectType : `str`
           The type of the object we are looking for.
        idDict : `dict`
           The type-specific keys which identify the specific object
           we are looking for.

        Returns
        -------
        object : the loaded object
        """

        path = self.getPath(objectType, idDict)
        obj = self.getFromPath(objectType, path)

        return obj

    def put(self, obj, objectType=None, idDict=None):
        """Persist an object.

        Args
        ----
        obj : `object`
           The object to persist
        objectType : `str`
           The type of the object we are looking for.
        idDict : `dict`
           The type-specific keys which identify the specific object
           we are looking for.

        Todo
        ----

        If `objectType` and `idDict` are not passed in, we look for
        `butlerType` and `butlerDict` in the `obj`.
        """

        path = self.getPath(objectType, idDict)
        tray, root = self.getTray(objectType)

        if not hasattr(obj, 'dump'):
            raise KeyError(f"{objectType} object does not know how to write itself: {obj}")

        putDir = path.parent
        if not putDir.exists():
            putDir.mkdir(mode=0o2775, parents=True)

        self.logger.debug(f'dumping {objectType}({idDict}) = {type(obj)} to {path}')
        obj.dump(path)
