import os
from collections import defaultdict
from collections.abc import Iterable

import yaml
from typing import List

class HeaderFixDatabase:
    """Database of header fixes to apply

    Header fixes are a set of `dict`s indexed by ``visit``. These get written
    in YAML format for the astro_metadata_translator package to read.

    The standard set of header fixes is defined in the ``addStandardFixes``
    method. Further fixes can be added using the ``add`` method. New fixes can
    clobber (or supplement) previous fixes.
    """

    def __init__(self):
        # fixes[site][visit] --> header fixes to apply
        self.fixes = {site: defaultdict(dict) for site in "JLXIASPF"}
        self.addStandardFixes()

    def add(self, site, visits, **fixes):
        """Add a set of fixes for a list of visits

        Parameters
        ----------
        site : `str`
            Character indicating the site at which the file was generated.
            Supported characters include ``JLXIASPF``, but usually this is
            ``S`` for Subaru, ``L`` for LAM, and ``F`` for fake (i.e.,
            simulator).
        visits : iterable or `int`
            List of visits (or a single visit) for which the fixes apply.
        **fixes : `dict` mapping `str` to `str`/`int`/`float`
            Header fixes to apply.
        """
        if site not in "JLXIASPF":
            raise RuntimeError("Site must be one of JLXIASPF")
        for vv in visits if isinstance(visits, Iterable) else [visits]:
            self.fixes[site][vv].update(fixes)

    def write(self, path):
        """Write the database

        The database is written as one YAML file for each visit; this is the
        format mandated by astro_metadata_translator.

        Parameters
        ----------
        path : `str`
            Path to which to write correction files.
        """
        for site in self.fixes:
            for visit, fixes in self.fixes[site].items():
                instrument = "PFS" if site == "S" else f"PFS-{site}"
                filename = os.path.join(path, f"{instrument}-{visit}.yaml")
                with open(filename, "w") as ff:
                    yaml.dump(fixes, ff)

    def inclRange(self, start: int, stop: int) -> List[int]:
        """Create a list of integers corresponding to the input range.
        Start and stop values are inclusive.

        Parameters
        ----------
        start : `int`
            startvalue in range
        stop : `int`
            end value in range

        Returns
        -------
        visits : `list`[`int`]
            list of visits in inclusive range.
        """
        return list(range(start, stop + 1))

    def addStandardFixes(self):
        """Add the standard set of fixes

        This is the official list of header fixes to apply.
        """
        # Fix duplicate no-value (DM-23928)
        # Undefined values occurred in LAM data
        # between 2019-05-03 and 2019-06-14
        self.add("L", self.inclRange(16804, 20960), W_XHP2FR=0)

        # For Subaru exposures taken during Dec 2019, exposures
        # labelled as Neon were actually Krypton
        self.add("S", self.inclRange(423, 441) + [43, 54, 60], W_AITNEO=False, W_AITKRY=True)

        # Early SuNSS observations without the proper pfsDesignId
        self.add("S", self.inclRange(45750, 45852), W_PFDSGN=0xdeadbeef)
        self.add("S", self.inclRange(45750, 46080), W_LGTSRC='sunss')

        pfiEngineering = self.inclRange(67569, 67573) + [67587, 67588, 67594, 67605] + \
            list(set(self.inclRange(67611, 67650)) ^ set([67614, 67615, 67620])) + [67685, 67692, 67693, 67694] + \
            self.inclRange(67707, 67722) + self.inclRange(67739, 67743) + self.inclRange(67752, 67759) + \
            self.inclRange(67781, 67796) + self.inclRange(67809, 67838) + self.inclRange(67953, 67961) + \
            list(set(self.inclRange(68072, 68108)) ^ set([68095, 68097, 68098, 68102, 68103, 68105])) + \
            [68303, 68306, 68307, 68308] + self.inclRange(68323, 68327) + [68344, 68345, 68349, 68351] + \
            self.inclRange(68417, 68420) + self.inclRange(68424, 68427)

        pfiEven = [68360, 68361, 68373, 68388, 68391, 68398, 68399] + self.inclRange(68480, 68488) + \
            self.inclRange(68491, 68493) + self.inclRange(68499, 68501)
        pfiOdd = self.inclRange(68402, 68405) + self.inclRange(68504, 68506) + self.inclRange(68509, 68511) + \
            self.inclRange(68514, 68516)
        pfiBlack = self.inclRange(68410, 68412) + self.inclRange(68417, 68420) + self.inclRange(68424, 68427) + \
            list(set(self.inclRange(68519, 68527)) ^ set([68522]))
        pfiAll = [68432, 68433, 68434, 68475, 68476, 68477]

        # First night with PFI installed without the proper pfsDesignId
        self.add("S", pfiEngineering, W_PFDSGN=0x71c4c79bd4b5d30)
        self.add("S", pfiEven, W_PFDSGN=0x1d317decdf389ea8)
        self.add("S", pfiOdd, W_PFDSGN=0x31b5b0ab7d661e15)
        self.add("S", pfiBlack, W_PFDSGN=0x2ed537b1a10ff1b2)
        self.add("S", pfiAll, W_PFDSGN=0x40dbf5546df0d55e)

        # Wrong hgcd status for all these visits.
        self.add("S", self.inclRange(79990, 80852), W_AITHGC=False)

        # Only one lamp was turned on each time.
        allLamps = ['W_AITNEO', 'W_AITKRY', 'W_AITXEN', 'W_AITARG']
        wrongs = self.inclRange(79990, 79997)
        onlyLamps = allLamps * 2
        for visit, onlyLamp in zip(wrongs, onlyLamps):
            lampDict = dict([(lamp, False) for lamp in allLamps])
            lampDict[onlyLamp] = True
            self.add("S", [visit], **lampDict)

        # Correcting dcb header keys.
        self.add("S", self.inclRange(81862, 81865) + [81872, 81873] + self.inclRange(81879, 81882), W_AITQTH=True)
        self.add("S", [81869, 81874], W_AITKRY=True)
        self.add("S", [81870], W_AITNEO=True)
        self.add("S", [81871], W_AITARG=True)

        # Correcting dcb header keys.
        self.add("S", self.inclRange(83098, 83100) + self.inclRange(83304, 83365), W_AITQTH=True)
        self.add("S", self.inclRange(83422, 83523) + [83652] + self.inclRange(83673, 83952), W_AITQTH=True)
        self.add("S", [84004,84005], W_AITQTH=True)
        self.add("S", self.inclRange(83058, 83060) + self.inclRange(83524, 83548), W_AITKRY=True)
        self.add("S", [83651] + self.inclRange(83653, 83662) + [83367], W_AITKRY=True)
        self.add("S", self.inclRange(83977, 83400), W_AITKRY=True)
        self.add("S", self.inclRange(83368, 83373) + self.inclRange(83549, 83578), W_AITARG=True)
        self.add("S", self.inclRange(83648, 83650) + self.inclRange(83663, 83672), W_AITARG=True)
        self.add("S", self.inclRange(83953, 83976), W_AITARG=True)
        self.add("S", [83367] + self.inclRange(83374, 83379), W_AITHGA=True)
        self.add("S", self.inclRange(83579, 83608) + self.inclRange(83637, 83647), W_AITHGA=True)

        # Correcting domeFlat header keys
        self.add("S", [46084, 46085], W_AITQTH=True)
        self.add("S", self.inclRange(46228, 46262), W_AITQTH=True)
        self.add("S", self.inclRange(63046, 63106), W_AITQTH=True)
        self.add("S", [67709, 67710], W_AITQTH=True)
        self.add("S", self.inclRange(68523, 68527), W_AITQTH=True)
        self.add("S", self.inclRange(75954, 75958), W_AITQTH=True)
        self.add("S", [75966, 76143, 76144], W_AITQTH=True)
        self.add("S", self.inclRange(76466, 76468), W_AITQTH=True)
        self.add("S", [76559, 76560, 76763], W_AITQTH=True)
        self.add("S", self.inclRange(77173, 77175), W_AITQTH=True)
        self.add("S", [77404, 77405, 77457, 77464, 77466, 78196, 78248, 78705, 78759], W_AITQTH=True)
        self.add("S", self.inclRange(80621, 80640), W_AITQTH=True)
        self.add("S", self.inclRange(81979, 81988), W_AITQTH=True)
        self.add("S", self.inclRange(82113, 82127), W_AITQTH=True)
        self.add("S", self.inclRange(82230, 82238), W_AITQTH=True)
        self.add("S", [82428, 82429], W_AITQTH=True)
        self.add("S", self.inclRange(82527, 82533), W_AITQTH=True)

        # Correcting pfsDesignId header keys for joint DCB/SuNSS exposures
        designIdDCBSuNSS = 0x5cab8319135e443f
        self.add("S", self.inclRange(84574, 84578), W_PFDSGN=designIdDCBSuNSS)
        self.add("S", self.inclRange(84580, 84632), W_PFDSGN=designIdDCBSuNSS)
        self.add("S", self.inclRange(84666, 84695), W_PFDSGN=designIdDCBSuNSS)
        self.add("S", self.inclRange(84720, 84737), W_PFDSGN=designIdDCBSuNSS)
        self.add("S", self.inclRange(84742, 84753), W_PFDSGN=designIdDCBSuNSS)
        self.add("S", self.inclRange(84763, 84792), W_PFDSGN=designIdDCBSuNSS)
        self.add("S", self.inclRange(84795, 84824), W_PFDSGN=designIdDCBSuNSS)
        self.add("S", self.inclRange(84827, 84886), W_PFDSGN=designIdDCBSuNSS)
        self.add("S", self.inclRange(84905, 84934), W_PFDSGN=designIdDCBSuNSS)

        # Engineering run april 23.
        self.add("S", [91057, 91681], W_PFDSGN=0x662cf9deec5c1ce9)
        self.add("S", [92481], W_PFDSGN=0x15adaf923f34fb55)
