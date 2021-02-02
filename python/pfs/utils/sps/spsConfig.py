from pfs.utils.spectroIds import SpectroIds
from pfs.utils.sps.parts import Cam, Shutter, Rda, Fca, Bia, Iis


class NoShutterException(Exception):
    """Base class for other exceptions"""

    def __init__(self, text):
        Exception.__init__(self, text)


class SpecModule(SpectroIds):
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
        return [self.bcu, self.rcu, self.ncu]

    @property
    def shutters(self):
        return [self.bsh, self.rsh]

    @property
    def parts(self):
        return self.cams + self.shutters + [self.rda]

    @property
    def allCams(self):
        return dict([(cam.arm, cam) for cam in self.cams if cam.operational])

    @property
    def genSpecParts(self):
        return f'{self.specName}Parts={",".join([part.state for part in self.parts])}'

    @property
    def genLightSource(self):
        return f'{self.specName}LightSource={self.lightSource}'

    @classmethod
    def fromConfig(cls, specName, config, spsData):

        try:
            lightSource, = spsData.loadKey(f'{specName}LightSource')
        except:
            lightSource = None

        try:
            spsModules = [specName.strip() for specName in config.get('sps', 'spsModules').split(',')]
        except:
            spsModules = SpecModule.validNames

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
        try:
            spsModules = spsModel.keyVarDict['spsModules'].getValue()
        except:
            spsModules = SpecModule.validNames

        spsModule = specName in spsModules
        specParts = spsModel.keyVarDict[f'{specName}Parts'].getValue()
        lightSource = spsModel.keyVarDict[f'{specName}LightSource'].getValue()

        specModule = cls(specName, spsModule=spsModule, lightSource=lightSource)
        specModule.assign(*specParts)

        return specModule

    def assign(self, bcu='none', rcu='none', ncu='none', bsh='none', rsh='none', fca='none', rda='none',
               bia='none', iis='none'):
        self.bcu = Cam(self, 'b', bcu)
        self.rcu = Cam(self, 'r', rcu)
        self.ncu = Cam(self, 'n', ncu)
        self.bsh = Shutter('b', bsh)
        self.rsh = Shutter('r', rsh)
        self.fca = Fca(fca)
        self.rda = Rda(rda)
        self.bia = Bia(bia)
        self.iis = Iis(iis)

    def declareLightSource(self, lightSource, spsData=None):
        if lightSource not in SpecModule.lightSources:
            raise RuntimeError(f'lightSource: {lightSource} must be one of: {",".join(SpecModule.lightSources)}')

        return spsData.persistKey(f'{self.specName}LightSource', lightSource)

    def camFromArm(self, arm):
        if arm not in SpecModule.validArms:
            raise RuntimeError(f'arm {arm} must be one of: {list(SpecModule.validArms.keys())}')

        fpa = SpecModule.armToFpa[arm]
        if fpa not in self.allCams.keys():
            raise RuntimeError(f'{fpa}{self.specNum} cam is not operational ({",".join(self.allCams.keys())})')

        return self.allCams[fpa]

    def lightSolver(self, arm, openShutter, inputLight='continuous'):
        shutters = [self.rsh, self.bsh] if arm == 'b' else [self.rsh]

        for shutter in shutters:
            outputLight = shutter.lightPath(inputLight, openShutter=openShutter)
            inputLight = outputLight

        return outputLight, shutters

    def shutterSet(self, arm, lightBeam):
        outputLight, shutters = self.lightSolver(arm, openShutter=lightBeam)

        if outputLight == 'continuous':
            raise NoShutterException(f"cannot control exposure on {arm} arm...")
        elif outputLight == 'none':
            if lightBeam:
                raise RuntimeError(f'light cant reach {arm} arm...')
        elif outputLight == 'unknown':
            raise RuntimeError(f'cannot guaranty anything on {arm} arm')

        shutters = [shutter for shutter in shutters if shutter.operational]
        shutters = shutters if lightBeam else shutters[-1:]

        return shutters

    def dependencies(self, arm, seqObj):
        try:
            shutters = self.shutterSet(arm, seqObj.lightBeam)
        except NoShutterException:
            shutters = self.askAnEngineer(seqObj)

        return shutters

    def askAnEngineer(self, seqObj):

        if 'dcb' in self.lightSource:
            if seqObj.lightBeam:
                if not seqObj.shutterRequired:
                    return []
            else:
                return [self.lightSource]

        raise


class SpsConfig(object):
    validCams = [f'{arm}{specNum}' for arm in SpectroIds.validArms.keys() for specNum in SpectroIds.validModules]

    def __init__(self, specModules):
        self.specModules = dict()
        for specModule in specModules:
            self.specModules[specModule.specName] = specModule

    @classmethod
    def fromModel(cls, spsModel):
        specModules = []
        for specName in SpecModule.validNames:
            try:
                specModules.append(SpecModule.fromModel(specName, spsModel))
            except ValueError:
                continue

        return cls(specModules)

    @classmethod
    def fromConfig(cls, spsActor):
        described = [specName for specName in SpecModule.validNames if specName in spsActor.config.sections()]
        specModules = [SpecModule.fromConfig(specName, spsActor.config, spsActor.instData) for specName in described]

        return cls([specModule for specModule in specModules])

    @property
    def spsModules(self):
        return dict([(name, module) for name, module in self.specModules.items() if module.spsModule])

    def identify(self, sm=None, arm=None, cams=None):
        if cams is None:
            specModules = self.selectModules(sm) if sm is not None else list(self.spsModules.values())
            specs = self.selectArms(specModules, arm)
        else:
            specs = [self.selectCam(c) for c in cams]

        return specs

    def selectModules(self, specNums):
        specModules = []
        for specNum in specNums:
            try:
                specModules.append(self.specModules[f'sm{specNum}'])
            except KeyError:
                raise RuntimeError(f'sm{specNum} is not wired in, specModules={",".join(self.specModules.keys())}')

        return specModules

    def selectArms(self, specModules, arms=None):
        if arms is not None:
            cams = [specModule.camFromArm(arm) for specModule in specModules for arm in arms]
        else:
            cams = sum([list(specModule.allCams.values()) for specModule in specModules], [])

        return cams

    def selectCam(self, cam):
        if cam not in SpsConfig.validCams:
            raise ValueError(f'{cam} is not a valid cam')

        arm, specNum = cam[0], int(cam[1])

        [spec] = self.identify(sm=[specNum], arm=[arm])
        return spec
