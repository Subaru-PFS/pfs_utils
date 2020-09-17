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

        return open(os.path.join(root, 'data/sps', f'{actorName}.yaml'), mode)

    @staticmethod
    def loadFile(actorName):
        """ Load per-actor instdata yaml file. 
        Returns python dictionary if file exists.
        """
        with InstData.openFile(actorName) as cfgFile:
            return yaml.load(cfgFile)

    @staticmethod
    def loadPersisted(actorName, keyName):
        """ Load persisted actor keyword from outside mhs world. """
        return InstData.loadFile(actorName)[keyName]

    def load(self, keyName, actorName=None, cmd=None):
        """ Load mhs keyword values from disk.

        Args
        ----
        keyName : str
            keyword name.
        """
        cmd = self.actor.bcast if cmd is None else cmd
        actorName = self.actorName if actorName is None else actorName
        cmd.inform(f'text="loading {keyName} from instdata"')
        return self.loadPersisted(actorName, keyName)

    def persist(self, keyName, *values, cmd=None):
        """ Save mhs keyword values to disk.
        Create per-actor file if it does not yet exist.

        Args
        ----
        keyName : str
            keyword name.
        """
        cmd = self.actor.bcast if cmd is None else cmd

        try:
            cfg = self.loadFile(self.actorName)
        except FileNotFoundError:
            cmd.warn(f'text="instdata : {self.actorName} file does not exist, creating empty file"')
            cfg = dict()

        cfg[keyName] = values

        with self.openFile(self.actorName, mode='w') as cfgFile:
            yaml.dump(cfg, cfgFile)

        cmd.inform(f'text="dumped {keyName} to instdata"')
