""" SPS-specific FITS routines. """

# These should come from some proper data product, but I would be
# *very* surprised if the values matter much. They certainly do
# not matter for PFS itself.
#
# The table values were gathered from ETC data and some grating design docs.
#
armSpecs = dict(b=dict(wavemin=380.0,
                       wavemax=650.0,
                       wavemid=519.0,
                       fringe=711.0),
                r=dict(wavemin=630.0,
                       wavemax=970.0,
                       wavemid=806.0,
                       fringe=557.0),
                m=dict(wavemin=710.0,
                       wavemax=885.0,
                       wavemid=1007.0,
                       fringe=1007.0),
                n=dict(wavemin=940.0,
                       wavemax=1260.0,
                       wavemid=1107.0,
                       fringe=1007.0))

def getSpsSpectroCards(arm):
    """Return the Subaru-specific spectroscopy cards.

    See INSTRM-1022 for gory discussion. We might need to add other
    cards.

    Args
    ----
    arm : `str`
      the letter for the arm we are interested in. 'brnm'

    Returns
    -------
    cards : list of fitsio-compliant card dicts.

    """

    cards = []
    try:
        specs = armSpecs[arm]
    except KeyError:
        raise ValueError(f'arm must be one of "brnm", not {arm}')

    disperserName = f'VPH_{arm}_{int(specs["fringe"])}_{int(specs["wavemid"])}nm'
    cards.append(dict(name='DISPAXIS', value=(1 if arm == 'n' else 2),
                      comment='Dispersion axis (along %s)' % ('rows' if arm == 'n' else 'columns')))
    cards.append(dict(name='DISPRSR', value=disperserName, comment='Disperser name (arm_fringe/mm_centralNm)'))
    cards.append(dict(name='WAV-MIN', value=specs['wavemin'], comment='[nm] Blue edge of the bandpass'))
    cards.append(dict(name='WAV-MAX', value=specs['wavemax'], comment='[nm] Red edge of the bandpass'))
    cards.append(dict(name='WAVELEN', value=specs['wavemid'], comment='[nm] Middle of the bandpass'))

    return cards

def getSpsWcs(arm):
    """Return a Subaru-compliant WCS solution."""

    raise NotImplementedError("Sorry, no SPS WCS yet!")
