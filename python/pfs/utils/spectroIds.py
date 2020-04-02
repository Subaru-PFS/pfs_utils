import logging
import os
import re
import socket

def idFromHostname(hostname=None):
    """ Return the module or cryostat id by looking at the hostname. 

    Returns:
    id : str
      Either "smM" or "[brmn]N", or None
    """

    if hostname is None:
        hostname = socket.gethostname()
    hostname = os.path.splitext(hostname)[0]

    m = re.match('^(enu)-(sm[1-4])', hostname)
    if m is not None:
        return m.groups[-1]

    m = re.match('^(xcu|ccd)-([brnm][1-489])', hostname)
    if m is not None:
        return m.groups[-1]

    return None

def getSite():
    """ Return the site name. Extracted from a DNS TXT record. """

    defaultSite = 'Z'

    import dns.resolver
    try:
        ans = dns.resolver.query('pfs-site.', 'TXT')
        site = ans[0].strings[0].decode('latin-1')
    except dns.resolver.NXDOMAIN:
        logging.warn(f'no "pfs-site." record found in DNS, using "{defaultSite}"')
        site = defaultSite

    return site

class SpectroIds(object):
    validArms = dict(b=1, r=2, n=3, m=4)
    validSites = {'A','J','L','S','C','X','Z'}
    validModules = tuple(range(1,5))

    def __init__(self, partName=None, hostname=None, site=None):
        """ Track, and optionally detect, spectrograph/camera names and ids. 

        Args
        ----
        partName : str
            A name of the form 'b2' or 'sm3'.
            None to autodetect from the hostname
        site : str
            One of the PFS site names.
            None to autodetect using the DNS pfs-site. record

        Internally we track the spectrograph module number, the site, and (optionally) the arm letter.

        Examples
        --------

        >>> ids = SpectroIds()
        >>> ids.makeFitsName(999, 'A')
        'PFJA00099912.fits'

        """

        self.arm = None
        self.specNum = None
        if site is None:
            site = getSite()

        if site not in self.validSites:
            raise RuntimeError('site (%s) must one of: %s' % (site, self.validSites))
        self.site = site

        if self.site == 'J':
            self.validModules += (8,9)

        if partName is None:
            partName = idFromHostname(hostname=hostname)

        # Not an SPS host, but we did get site and perhaps other things.
        if partName is None:
            return

        if len(partName) == 2:
            if partName[0] not in self.validArms:
                raise RuntimeError('arm (%s) must one of: %s' % (partName[0], list(self.validArms.keys())))
            self.arm = partName[0]
        elif len(partName) == 3:
            if partName[:2] != 'sm':
                raise RuntimeError('module (%s) must be smN' % (partName))
        else:
            raise RuntimeError('part name (%s) must be of the form "r1" or "sm2"' % (partName))

        try:
            self.specNum = int(partName[-1])
        except ValueError:
            raise RuntimeError('spectrograph number (%s) must be an integer' % (partName[-1]))
        if self.specNum not in self.validModules:
            raise RuntimeError('spectrograph number (%s) must be in %s' % (self.specNum,
                                                                           self.validModules))

    def __str__(self):
        return "SpectroIds(site=%s cam=%s arm=%s spec=%s)" % (self.site,
                                                              self.cam, self.arm, self.specNum)

    @property
    def camName(self):
        if self.arm is None:
            return None
        return "%s%d" % (self.arm, self.specNum)

    @property
    def specName(self):
        if self.specNum is None:
            return None
        return f'sm{self.specNum}'

    @property
    def armNum(self):
        if self.arm is None:
            return None
        return self.validArms[self.arm]

    @property
    def idDict(self):
        """ Return a dictionary of our valid fields. Intended for name interpolation, etc. """
        _idDict = dict(site=self.site)

        if self.specNum is not None:
            _idDict['spectrograph'] = self.specNum
            _idDict['specNum'] = self.specNum
            _idDict['specName'] = self.specName
        if self.arm is not None:
            _idDict['arm'] = self.arm
            _idDict['armNum'] = self.armNum
            _idDict['camName'] = self.camName
            _idDict['cam'] = self.camName

        return _idDict

    def makeSpsFitsName(self, visit, fileType='A', extension='.fits'):
        """ Create a complete filename 

        Args
        ====

        visit : int
          Between 0..999999
        fileType : str 
          'A' or 'B', for now. Only required if arm == 'n'

        By default, returns a fully fleshed out filename appropriate for visible or NIR cameras.
        """

        if self.arm is None:
            raise RuntimeError('cannot generate a filename without an arm')

        if not isinstance(visit, int) or visit < 0 or visit > 999999:
            raise ValueError('visit must be an integer between 0 and 999999')

        if self.arm == 'n':
            if fileType not in {'A', 'B'}:
                raise ValueError('fileType must be A or B')
        else:
            if fileType not in {'A'}:
                raise ValueError('fileType must be A')

        return "PF%1s%1s%06d%1d%1d%s" % (self.site,
                                         fileType,
                                         visit,
                                         self.specNum,
                                         self.armNum,
                                         extension)
