from pfs.utils.spectroIds import SpectroIds
from pfs.utils.sps.parts import Cam, Shutter, Rda, Fca, Bia, Iis


class NoShutterException(Exception):
    """Exception raised when an exposure is required without any working shutter to ensure exposure time.

    Attributes
    ----------
    text : `str`
       Exception text.
    """

    def __init__(self, text):
        Exception.__init__(self, text)


class SpecModule(SpectroIds):
    """Placeholder to handle a single spectrograph module configuration, lightSource, parts...
    It also describe if this module is part of the spectrograph system or standalone module.

    Attributes
    ----------
    specName : `str`
        Spectrograph module identifier (sm1, sm2, ...).
    spsModule : `bool`
        Is this module actually part of the spectrograph system (sps), or standalone.
    lightSource : `str`
        The light source which is feeding the spectrograph module.
    """
    knownParts = ['bcu', 'rcu', 'ncu', 'bsh', 'rsh', 'fca', 'rda', 'bia', 'iis']
    armToFpa = dict(b='b', m='r', r='r', n='n')
    lightSources = ['dcb', 'dcb2', 'sunss', 'pfi']
    validNames = [f'sm{specNum}' for specNum in SpectroIds.validModules]

    def __init__(self, specName, spsModule=True, lightSource=None):
        SpectroIds.__init__(self, specName)
        self.spsModule = spsModule
        self.lightSource = lightSource

    @property
    def cams(self):
        """Camera dictionary for a given spectrograph module."""
        return dict([(cam.arm, cam) for cam in [self.bcu, self.rcu, self.ncu]])

    @property
    def parts(self):
        """All existing spectrograph module parts, basically camera + entrance unit parts."""
        return list(self.cams.values()) + [self.bsh, self.rsh, self.fca, self.rda, self.bia, self.iis]

    @property
    def opeCams(self):
        """Camera considered operational, Will be exposed if no arm/camera are specified."""
        return [cam for cam in self.cams.values() if cam.operational]

    @property
    def genSpecParts(self):
        """Generate string that describe the spectrograph module parts."""
        return f'{self.specName}Parts={",".join([part.state for part in self.parts])}'

    @property
    def genLightSource(self):
        """Generate string that describe the spectrograph light source."""
        return f'{self.specName}LightSource={self.lightSource}'

    @classmethod
    def fromConfig(cls, specName, config, spsData):
        """Instantiate SpecModule class from spsActor.configParser.

        Parameters
        ----------
        specName : `str`
            Spectrograph module identifier (sm1, sm2, ...).
        config : `spsActor.configParser`
            ConfigParser object from spsActor.
        spsData : `pfs.utils.instdata.InstData`
            Sps instrument data object.

        Returns
        -------
        specModule : `SpecModule`
            SpecModule object.
        """
        try:
            lightSource, = spsData.loadKey(f'{specName}LightSource')
        except:
            lightSource = None

        try:
            spsModules = [specName.strip() for specName in config.get('sps', 'spsModules').split(',')]
        except:
            spsModules = [specName for specName in SpecModule.validNames if specName in config.sections()]

        spsModule = specName in spsModules
        specModule = cls(specName, spsModule=spsModule, lightSource=lightSource)

        parts = dict()
        for partName in SpecModule.knownParts:
            try:
                state = config.get(specName, partName)
            except:
                state = 'none'

            parts[partName] = state

        specModule.assign(**parts)
        return specModule

    @classmethod
    def fromModel(cls, specName, spsModel):
        """Instantiate SpecModule class from spsActor model.

        Parameters
        ----------
        specName : `str`
            Spectrograph module identifier (sm1, sm2, ...).
        spsModel : `opscore.actor.model.Model`
            SpsActor model.

        Returns
        -------
        specModule : `SpecModule`
            SpecModule object.
        """
        spsModule = specName in spsModel.keyVarDict['spsModules'].getValue()
        specParts = spsModel.keyVarDict[f'{specName}Parts'].getValue()
        lightSource = spsModel.keyVarDict[f'{specName}LightSource'].getValue()

        specModule = cls(specName, spsModule=spsModule, lightSource=lightSource)
        specModule.assign(*specParts)

        return specModule

    def assign(self, bcu='none', rcu='none', ncu='none', bsh='none', rsh='none', fca='none', rda='none', bia='none',
               iis='none'):
        """Instantiate and assign each part from the provided operating state.

        Parameters
        ----------
        bcu : `str`
            Blue camera operating state.
        rcu : `str`
            Red camera operating state.
        ncu : `str`
            Nir camera operating state.
        bsh : `str`
            Blue shutter operating state.
        rsh : `str`
            Red Shutter operating state.
        fca : `str`
            Fiber Cable A (hexapod) operating state.
        rda : `str`
            Red exchange mechanism operating state.
        bia : `str`
            Back Illumination Assembly operating state
        iis : `str`
            Internal Illumination Sources operating state.
        """
        self.bcu = Cam(self, 'b', bcu)
        self.rcu = Cam(self, 'r', rcu)
        self.ncu = Cam(self, 'n', ncu)
        self.bsh = Shutter(self, 'b', bsh)
        self.rsh = Shutter(self, 'r', rsh)
        self.fca = Fca(self, fca)
        self.rda = Rda(self, rda)
        self.bia = Bia(self, bia)
        self.iis = Iis(self, iis)

    def camera(self, arm):
        """Return Cam object from arm.

        Parameters
        ----------
        arm : `str`
            spectrograph module arm(b,r,n,m)

        Returns
        -------
        cam : `pfs.utils.sps.part.Cam`
            Cam object.
        """
        if arm not in SpecModule.validArms:
            raise RuntimeError(f'arm {arm} must be one of: {list(SpecModule.validArms.keys())}')

        fpa = SpecModule.armToFpa[arm]
        cam = self.cams[fpa]

        if not cam.operational:
            raise RuntimeError(f'{str(cam)} cam state: {cam.state}, not operational ...')

        return cam

    def lightSolver(self, arm, openShutter=True):
        """ In the spectrograph, for blue arm light goes through two shutters but only one for the other arms.
        This function simulate what's the output light for a given set of shutters and a continuous input light.

        Parameters
        ----------
        arm : `str`
            Spectrograph arm.
        openShutter : `bool`
            Shutter is required to open.

        Returns
        -------
        outputLight : `str`
             Output light beam(continuous, timed, none, unknown).
        shutterSet : list of `pfs.utils.sps.part.Shutter`
            List of matching shutters.

        """
        inputLight = 'continuous'
        shutterSet = [self.rsh, self.bsh] if arm == 'b' else [self.rsh]

        for shutter in shutterSet:
            outputLight = shutter.lightPath(inputLight, openShutter=openShutter)
            inputLight = outputLight

        return outputLight, shutterSet

    def shutterSet(self, arm, lightBeam):
        """Return the required shutter set for a given arm and lightBeam.
        Check that what you want to measure is actually what you get.

        Parameters
        ----------
        arm : `str`
            Spectrograph arm.
        lightBeam : `bool`
            Are you measuring photons ?

        Returns
        -------
        requiredShutters : list of `pfs.utils.sps.part.Shutter`
            List of required shutters.
        """
        outputLight, shutterSet = self.lightSolver(arm, openShutter=lightBeam)

        if outputLight == 'continuous':
            raise NoShutterException(f"cannot control exposure on {arm} arm...")
        elif outputLight == 'none':
            if lightBeam:
                raise RuntimeError(f'light cant reach {arm} arm...')
        elif outputLight == 'unknown':
            raise RuntimeError(f'cannot guaranty anything on {arm} arm')

        opeShutters = [shutter for shutter in shutterSet if shutter.operational]
        requiredShutters = opeShutters if lightBeam else opeShutters[-1:]

        return requiredShutters

    def dependencies(self, arm, seqObj):
        """Retrieve the spectrograph dependencies given the arm and the data acquisition sequence type.
        Only implemented shutter dependencies for know.

        Parameters
        ----------
        arm : `str`
            Spectrograph arm.
        seqObj : `iicActor.sps.sequence.Sequence`
           Sequence instance.

        Returns
        -------
        names : `list` of `Part`
            List of required parts.
        """
        try:
            deps = self.shutterSet(arm, seqObj.lightBeam)
        except NoShutterException:
            deps = self.askAnEngineer(seqObj)

        return deps

    def askAnEngineer(self, seqObj):
        """If NoShutterException is raised, check for special cases, timed dcb exposure is one of them.

        Parameters
        ----------
        seqObj : `iicActor.sps.sequence.Sequence`
           Sequence instance.

        Returns
        -------
        names : `list` of `Part`
            List of required parts.
        """
        if 'dcb' in self.lightSource:
            if seqObj.lightBeam:
                if not seqObj.shutterRequired:
                    return []
            else:
                return [self.lightSource]

        elif 'sunss' in self.lightSource:
            return []

        raise


class SpsConfig(object):
    """Placeholder spectrograph system configuration in mhs world.

    Attributes
    ----------
    specModules : list of `SpecModule`
        List of described and instanciated spectrograph module.
    """
    validCams = [f'{arm}{specNum}' for arm in SpectroIds.validArms.keys() for specNum in SpectroIds.validModules]

    def __init__(self, specModules):
        self.specModules = dict()
        for specModule in specModules:
            self.specModules[specModule.specName] = specModule

    @property
    def spsModules(self):
        """Spectrograph modules labelled as part of the spectrograph system(sps)"""
        return dict([(name, module) for name, module in self.specModules.items() if module.spsModule])

    @classmethod
    def fromConfig(cls, spsActor):
        """Instantiate SpsConfig class from spsActor.configParser.
        Instantiate only SpecModule which are described in the configuration file.

        Parameters
        ----------
        spsActor : `actorcore.ICC.ICC`
            spsActor.

        Returns
        -------
        spsConfig : `SpsConfig`
            SpsConfig object.
        """
        specNames = [specName for specName in SpecModule.validNames if specName in spsActor.config.sections()]
        specModules = [SpecModule.fromConfig(specName, spsActor.config, spsActor.instData) for specName in specNames]

        return cls([specModule for specModule in specModules])

    @classmethod
    def fromModel(cls, spsModel):
        """Instantiate SpsConfig class from spsActor model.
        Instantiate only SpecModule which are in specModules.

        Parameters
        ----------
        spsModel : `opscore.actor.model.Model`
            SpsActor model.

        Returns
        -------
        spsConfig : `SpsConfig`
            SpsConfig object.
        """
        specNames = spsModel.keyVarDict['specModules'].getValue()
        return cls([SpecModule.fromModel(specName, spsModel) for specName in specNames])

    def identify(self, sm=None, arm=None, cams=None):
        """Identify which camera(s) to expose from outer product(sm*arm) or cams.
        If no sm if provided then we're assuming modules labelled as sps.
        If no arm if provided then we're assuming all arms.

        Parameters
        ----------
        sm : list of `int`
            List of required spectrograph module number (1,2,..).
        arm : list of `str`
            List of required arm (b,r,n,m)
        cams : list of `str`
            List of camera names.

        Returns
        -------
        cams : `list` of `Cam`
            List of Cam object.
        """
        if cams is None:
            specModules = self.selectModules(sm) if sm is not None else list(self.spsModules.values())
            cams = self.selectArms(specModules, arm)
        else:
            cams = [self.selectCam(camName) for camName in cams]

        return cams

    def selectModules(self, specNums):
        """Select spectrograph modules for a given list of spectrograph number.

        Parameters
        ----------
        specNums : list of `int`
            List of required spectrograph module number (1,2,..).

        Returns
        -------
        specModules : `list` of `SpecModule`
            List of SpecModule object.
        """
        specModules = []
        for specNum in specNums:
            try:
                specModules.append(self.specModules[f'sm{specNum}'])
            except KeyError:
                raise RuntimeError(f'sm{specNum} is not wired in, specModules={",".join(self.specModules.keys())}')

        return specModules

    def selectArms(self, specModules, arms=None):
        """Return the outer product between provided specModules and arms.
        If no arms is provided, then assuming all arms.

        Parameters
        ----------
        specModules : list of `specModules`
            List of required spectrograph module.
        arm : list of `str`
            List of required arm (b,r,n,m)

        Returns
        -------
        cams : `list` of `Cam`
            List of Cam object.
        """
        if arms is not None:
            cams = [specModule.camera(arm) for specModule in specModules for arm in arms]
        else:
            cams = sum([specModule.opeCams for specModule in specModules], [])

        return cams

    def selectCam(self, camName):
        """Retrieve Cam object from camera name.

        Parameters
        ----------
        camName : `str`
            Camera name.

        Returns
        -------
        cam : `pfs.utils.sps.part.Cam`
            Cam object.
        """
        if camName not in SpsConfig.validCams:
            raise ValueError(f'{camName} is not a valid cam')

        arm, specNum = camName[0], int(camName[1])

        [cam] = self.identify(sm=[specNum], arm=[arm])
        return cam

    def declareLightSource(self, lightSource, specNum=None, spsData=None):
        """Declare light source for a given spectrograph number.
        if no spectrograph number is provided, then we're assuming that light is declared for sps.
        The only light source which can feed multiple spectrograph modules is pfi.
        The other sources are unassigned before getting reassigned to another spectrograph module.

        Parameters
        ----------
        lightSource : `str`
        The light source which is feeding the spectrograph module.
        specNum : `int`
            Spectrograph module number (1,2,..).
        spsData : `pfs.utils.instdata.InstData`
            Sps instrument data object.
        """
        if lightSource not in SpecModule.lightSources:
            raise RuntimeError(f'lightSource: {lightSource} must be one of: {",".join(SpecModule.lightSources)}')

        specModules = self.selectModules([specNum]) if specNum is not None else self.spsModules.values()
        if lightSource != 'pfi':
            if len(specModules) > 1:
                raise RuntimeError(f'{lightSource} can only be plugged to a single SM')

            toUndeclare = [module for module in self.specModules.values() if module.lightSource == lightSource]
            for specModule in toUndeclare:
                spsData.persistKey(f'{specModule.specName}LightSource', None)

        for specModule in specModules:
            spsData.persistKey(f'{specModule.specName}LightSource', lightSource)
