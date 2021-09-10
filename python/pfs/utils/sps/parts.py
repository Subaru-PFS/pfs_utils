from pfs.utils.spectroIds import SpectroIds


class Part(object):
    """Placeholder to handle sps parts and their operational state.
    
    Attributes
    ----------
    state : `str`
        Current operation state.
    """
    knownStates = ['ok', 'broken', 'none']

    def __init__(self, specModule, state='none'):
        if state not in self.knownStates:
            raise ValueError(f'unknown state:{state}')

        self.specModule = specModule
        self.state = state

    @property
    def operational(self):
        return self.state == 'ok'


class Rda(Part):
    """Placeholder to handle red exchange mechanism operational state and special rules that apply to it.
    low/med would mean that the motor is broken, but the disperser is still in a usable position.
    
    Not doing anything special yet.
    
    Attributes
    ----------
    state : `str`
        Current operation state.
    """
    knownStates = ['ok', 'broken', 'none', 'low', 'med']

    def __init__(self, specModule, state='none'):
        Part.__init__(self, specModule, state=state)

    def __str__(self):
        """Part identifier."""
        return f'rda_{self.specModule.specName}'

    def checkTarget(self, targetPosition):
        """Check if rda operational state allow reaching the targetPosition. """
        if self.state == targetPosition:
            return True
        else:
            return self.operational


class Fca(Part):
    """Placeholder to handle fiber cable A operational state and special rules that apply to it.
    home would mean that hexapod is not working, but slit is still in focus, so could be used.
    
    Not doing anything special yet.
    
    Attributes
    ----------
    state : `str`
        Current operation state.
    """
    knownStates = ['ok', 'broken', 'none', 'home']

    def __init__(self, specModule, state='none'):
        Part.__init__(self, specModule, state=state)

    def __str__(self):
        """Part identifier."""
        return f'fca_{self.specModule.specName}'


class Bia(Part):
    """Placeholder to handle Back Illumination Assembly operating state and special rules that apply to it.
    
    Not doing anything special yet.
    
    Attributes
    ----------
    state : `str`
        Current operation state.
    """
    knownStates = ['ok', 'broken', 'none']

    def __init__(self, specModule, state='none'):
        Part.__init__(self, specModule, state=state)

    def __str__(self):
        """Part identifier."""
        return f'bia_{self.specModule.specName}'


class Iis(Part):
    """Placeholder to Internal Illumination Sources(engineering fibers) operating state and its special rules.
    
    Not doing anything special yet.
    
    Attributes
    ----------
    state : `str`
        Current operation state.
    """
    knownStates = ['ok', 'broken', 'none']

    def __init__(self, specModule, state='none'):
        Part.__init__(self, specModule, state=state)

    def __str__(self):
        """Part identifier."""
        return f'iis_{self.specModule.specName}'


class Shutter(Part):
    """Placeholder to handle shutter operating state and special rules that apply to it.
    open/close would mean that shutter is not working and stuck in a known position, which allow us to continue data
    acquisition in some cases.
    
    Attributes
    ----------
    state : `str`
        Current operation state.
    """
    knownStates = ['ok', 'broken', 'open', 'closed', 'none']

    def __init__(self, specModule, arm, state='none'):
        self.arm = arm
        Part.__init__(self, specModule, state=state)

    def __str__(self):
        """Part identifier."""
        return f'{self.arm}sh_{self.specModule.specName}'

    def lightPath(self, inputLight, openShutter=True):
        """Simulate the output light, given the inputLight and the shutter operating state.

        Parameters
        ----------
        inputLight : `str`
            Input light beam(continuous, timed, none, unknown).
        openShutter : `bool`
            Shutter is required to open.

        Returns
        -------
        outputLight : `str`
             Output light beam(continuous, timed, none, unknown).
        """

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
    """Placeholder to handle camera operating state and special rules that apply to it.
    state=sci means that cam can be operated nominally.
    state=eng means that some operations are authorized using dcbLike source.
    
    Attributes
    ----------
    state : `str`
        Current operation state.
    """
    knownStates = ['sci', 'eng', 'broken', 'none']

    def __init__(self, specModule, fpa, state='none'):
        SpectroIds.__init__(self, f'{fpa}{specModule.specNum}')
        self.specModule = specModule
        Part.__init__(self, specModule, state=state)

    def __str__(self):
        """Part identifier."""
        return self.camName

    @property
    def lightSource(self):
        """The light source which is feeding the corresponding spectrograph module."""
        return self.specModule.lightSource

    @property
    def operational(self):
        """sci/eng is considered as an operational state with dcb-dcb2, but not for other sources."""
        if 'dcb' in self.lightSource:
            return self.state in ['sci', 'eng']
        else:
            return self.state == 'sci'

    def dependencies(self, seqObj):
        """Retrieve the spectrograph dependencies given the data acquisition sequence type.

        Parameters
        ----------
        seqObj : `iicActor.sps.sequence.Sequence`
           Sequence instance.

        Returns
        -------
        names : `list` of `Part`
            List of required parts.
        """
        return self.specModule.dependencies(self.arm, seqObj)
