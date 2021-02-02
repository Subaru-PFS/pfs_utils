from pfs.utils.spectroIds import SpectroIds


class Part(object):
    knownStates = ['ok', 'broken', 'none']

    def __init__(self, state='none'):
        if state not in self.knownStates:
            raise ValueError(f'unknown state:{state}')
        self.state = state

    @property
    def operational(self):
        return self.state == 'ok'

    @property
    def setup(self):
        return self.state != 'none'


class Rda(Part):
    knownStates = ['ok', 'broken', 'none', 'low', 'med']

    def __init__(self, state='none'):
        Part.__init__(self, state=state)

    def __str__(self):
        return 'rda'


class Fca(Part):
    knownStates = ['ok', 'broken', 'none', 'home']

    def __init__(self, state='none'):
        Part.__init__(self, state=state)

    def __str__(self):
        return 'fca'


class Bia(Part):
    knownStates = ['ok', 'broken', 'none']

    def __init__(self, state='none'):
        Part.__init__(self, state=state)

    def __str__(self):
        return 'bia'


class Iis(Part):
    knownStates = ['ok', 'broken', 'none']

    def __init__(self, state='none'):
        Part.__init__(self, state=state)

    def __str__(self):
        return 'iis'


class Shutter(Part):
    knownStates = ['ok', 'broken', 'open', 'closed', 'none']

    def __init__(self, arm, state='none'):
        self.arm = arm
        Part.__init__(self, state=state)

    def __str__(self):
        return f'{self.arm}sh'

    def lightPath(self, inputLight, openShutter=True):

        if self.state == 'ok':
            if openShutter:
                outputLight = 'timed' if inputLight == 'continuous' else inputLight
            else:
                outputLight = 'none'

        elif self.state in ['open', 'none']:
            outputLight = inputLight

        elif self.state == 'closed':
            outputLight = 'none'

        elif self.state == 'broken':
            outputLight = inputLight if inputLight == 'none' else 'unknown'

        return outputLight


class Cam(SpectroIds, Part):
    knownStates = ['sci', 'eng', 'broken', 'none']

    def __init__(self, specModule, fpa, state='none'):
        SpectroIds.__init__(self, f'{fpa}{specModule.specNum}')
        self.specModule = specModule
        Part.__init__(self, state=state)

    def __str__(self):
        return self.camName

    @property
    def lightSource(self):
        return self.specModule.lightSource

    @property
    def operational(self):
        if 'dcb' in self.lightSource:
            return self.state in ['sci', 'eng']
        else:
            return self.state == 'sci'

    def dependencies(self, seqObj):
        return self.specModule.dependencies(self.arm, seqObj)
