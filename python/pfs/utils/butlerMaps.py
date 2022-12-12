import yaml
import pandas as pd

def load_yaml(fname):
    """Load yaml from file fname"""
    with open(fname) as fd:
        return yaml.full_load(fd)

configMap = dict()
dataMap = dict()

dataMap['fpsRun'] = dict(template="raw/{pfsDay}/fps/{visit:06d}")

dataMap['spsFile'] = dict(template="raw/{pfsDay}/sps/PF{site}A{visit:06d}{spectrograph}{armNum}.fits")
dataMap['ccdFile'] = dataMap['spsFile']
dataMap['rampFile'] = dict(template="raw/{pfsDay}/sps/PF{site}B{visit:06d}{spectrograph}{armNum}.fits")
dataMap['mcsFile'] = dict(template="raw/{pfsDay}/mcs/PF{site}C{visit:06d}{frame:02d}.fits")
dataMap['agccFile'] = dict(template="raw/{pfsDay}/mcs/PF{site}D{visit:06d}{frame:02d}.fits")

dataMap['pfsConfig'] = dict(template="raw/{pfsDay}/pfsConfig/pfsConfig-{pfsConfigId:#016x}-{visit:06d}.fits")
configMap['pfi'] = dict(template="pfi/PFI.yaml", loader=load_yaml)
configMap['modulePath'] = dict(template="pfi/cobras/{moduleName}")
configMap['cobraGeometry'] = dict(template="pfi/cobras/{moduleName}/{cobraInModule}/geometry.yaml",
                                  loaderModule='ics.cobraCharmer.cobra')
configMap['moduleXml'] = dict(template="pfi/modules/{moduleName}/{moduleName}{version}.xml",
                              loaderModule='ics.cobraCharmer.pfiDesign')
configMap['motorMap'] = \
    dict(template="pfi/cobras/{moduleName}/{cobraModuleId}/maps/{motor}_{direction}_{mapName}.yaml",
         loaderModule='ics.cobraCharmer.motormap')
configMap['fiducials'] = dict(template="pfi/fiducial_positions.csv",
                              loader=lambda fname: pd.read_csv(fname, comment='#'))
configMap['black_dots'] = dict(template="pfi/dot/black_dots_mm.csv",
                               loader=lambda fname: pd.read_csv(fname, comment='#'))
