import os

import yaml


class InstData(object):
    varName = '$PFS_INSTDATA_DIR'

    def __init__(self, actor):
        """ Load /save mhs keywords values from/to disk.

        Args
        ----
        actor : actorcore.Actor object
            a running actor instance.
        """
        self.actor = actor

    @property
    def actorName(self):
        return self.actor.name

    @staticmethod
    def openFile(actorName, mode='r'):
        """ Open per-actor instdata file. """
        root = os.path.expandvars(InstData.varName)
        if root == InstData.varName:
            raise RuntimeError(f'{InstData.varName} is not defined')

        print(root, actorName)
        path = os.path.join(root, 'data/sps', f'{actorName}.yaml')
        print(path)
        return open(path, mode)

    @staticmethod
    def loadFile(actorName):
        """ Load per-actor instdata yaml file. 
        Returns python dictionary if file exists.
        """
        with InstData.openFile(actorName) as dataFile:
            return yaml.load(dataFile)

    @staticmethod
    def loadPersisted(actorName, keyName):
        """ Load persisted actor keyword from outside mhs world. """
        return InstData.loadFile(actorName)[keyName]

    def loadKey(self, keyName, actorName=None, cmd=None):
        """ Load mhs keyword values from disk.

        Args
        ----
        keyName : str
            keyword name.
        """
        cmd = self.actor.bcast if cmd is None else cmd
        actorName = self.actorName if actorName is None else actorName
        cmd.inform(f'text="loading {keyName} from instdata"')

        return InstData.loadPersisted(actorName, keyName)

    def loadKeys(self, actorName=None, cmd=None):
        """ Load all keys values from disk. """

        cmd = self.actor.bcast if cmd is None else cmd
        actorName = self.actorName if actorName is None else actorName
        cmd.inform(f'text="loading keys from instdata"')

        return InstData.loadFile(actorName)

    def _dump(self, data):
        """ Dump data dictionary to disk. """
        with self.openFile(self.actorName, mode='w') as dataFile:
            yaml.dump(data, dataFile)

    def _persist(self, keys, cmd=None):
        """ Load and update persisted data.
        Create a new file if it does not exist yet.

        Args
        ----
        keys : dict
            keyword dictionary.
        """
        cmd = self.actor.bcast if cmd is None else cmd

        try:
            data = self.loadKeys(self.actorName)
        except FileNotFoundError:
            cmd.warn(f'text="instdata : {self.actorName} file does not exist, creating empty file"')
            data = dict()

        data.update(keys)
        self._dump(data)

    def persistKey(self, keyName, *values, cmd=None):
        """ Save single mhs keyword values to disk.

        Args
        ----
        keyName : str
            keyword name.
        """
        cmd = self.actor.bcast if cmd is None else cmd
        data = dict([(keyName, values)])

        self._persist(data)
        cmd.inform(f'text="dumped {keyName} to instdata"')

    def persistKeys(self, keys, cmd=None):
        """ Save mhs keyword dictionary to disk.

        Args
        ----
        keys : dict
            keyword dictionary.
        """
        cmd = self.actor.bcast if cmd is None else cmd

        self._persist(keys)
        cmd.inform(f'text="dumped keys to instdata"')
