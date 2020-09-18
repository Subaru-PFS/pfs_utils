import ruamel_yaml

yaml = ruamel_yaml.YAML(typ='safe')

configMap = dict()
dataMap = dict()

dataMap['fpsRun'] = dict(template="raw/{pfsDay}/fps/{visit:06d}")

dataMap['spsFile'] = dict(template="raw/{pfsDay}/sps/PF{site}A{visit:06d}{spectrograph}{armNum}.fits")
dataMap['rampFile'] = dict(template="raw/{pfsDay}/sps/PF{site}B{visit:06d}{spectrograph}{armNum}.fits")
dataMap['mcsFile'] = dict(template="raw/{pfsDay}/mcs/PF{site}C{visit:06d}{frame:02d}.fits")

configMap['pfi'] = dict(template="pfi/PFI{version}.yaml",
                        loader=yaml.load)
configMap['modulePath'] = dict(template="pfi/cobras/{moduleName}")
configMap['cobraGeometry'] = dict(template="pfi/cobras/{moduleName}/{cobraInModule}/geometry.yaml",
                                  loaderModule='ics.cobraCharmer.cobra')
configMap['moduleXml'] = dict(template="pfi/modules/{moduleName}/{moduleName}{version}.xml",
                              loaderModule='ics.cobraCharmer.pfiDesign')
configMap['motorMap'] = \
    dict(template="pfi/cobras/{moduleName}/{cobraInModule}/maps/{motor}_{direction}_{mapName}.yaml",
         loaderModule='ics.cobraCharmer.motormap')
