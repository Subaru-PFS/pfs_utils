import ruamel_yaml

yaml = ruamel_yaml.YAML(typ='safe')

configMap = dict()
dataMap = dict()

dataMap['fpsRun'] = dict(template="pfs/{pfsDay}/fps/{visit:06d}")

dataMap['spsFile'] = dict(template="pfs/{pfsDay}/sps/PF{site}A{visit:06d}{spectrograph}{armNum}.fits")
dataMap['rampFile'] = dict(template="pfs/{pfsDay}/sps/PF{site}B{visit:06d}{spectrograph}{armNum}.fits")
dataMap['mcsFile'] = dict(template="pfs/{pfsDay}/mcs/PF{site}C{visit:06d}{frame:02d}.fits")

configMap['pfi'] = dict(template="pfi/PFI.yaml",
                        loader=yaml.load)
configMap['modulePath'] = dict(template="pfi/cobras/{moduleName}")
configMap['cobraGeometry'] = dict(template="pfi/cobras/{moduleName}/{cobraInModule}/geometry.yaml",
                                  loaderModule='ics.cobraCharmer.cobra')
configMap['moduleXml'] = dict(template="pfi/modules/{moduleName}/{moduleName}{version}.xml",
                              loaderModule='ics.cobraCharmer.pfiDesign')
configMap['motorMap'] = \
    dict(template="pfi/cobras/{moduleName}/{cobraModuleId}/maps/{motor}_{direction}_{mapName}.yaml",
         loaderModule='ics.cobraCharmer.motormap')
