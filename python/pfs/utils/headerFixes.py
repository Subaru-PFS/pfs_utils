import os
from collections import defaultdict
from collections.abc import Iterable

import yaml


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

    def addStandardFixes(self):
        """Add the standard set of fixes

        This is the official list of header fixes to apply.
        """
        # Fix duplicate no-value (DM-23928)
        # Undefined values occurred in LAM data
        # between 2019-05-03 and 2019-06-14
        self.add("L", range(16804, 20961), W_XHP2FR=0)

        # For Subaru exposures taken during Dec 2019, exposures
        # labelled as Neon were actually Krypton
        self.add("S", list(range(423, 442)) + [43, 54, 60], W_AITNEO=False, W_AITKRY=True)

        # Early SuNSS observations without the proper pfsDesignId
        self.add("S", range(45750, 45853), W_PFDSGN=0xdeadbeef)
        self.add("S", range(45750, 46081), W_LGTSRC='sunss')

        pfiEngineering = list(range(67569, 67574)) + [67587, 67588, 67594, 67605] + \
            list(set(range(67611, 67651)) ^ set([67614, 67615, 67620])) + [67685, 67692, 67693, 67694] + \
            list(range(67707, 67723)) + list(range(67739, 67744)) + list(range(67752, 67760)) + \
            list(range(67781, 67797)) + list(range(67809, 67839)) + list(range(67953, 67962)) + \
            list(set(list(range(68072, 68109))) ^ set([68095, 68097, 68098, 68102, 68103, 68105])) + \
            [68303, 68306, 68307, 68308] + list(range(68323, 68328)) + [68344, 68345, 68349, 68351] + \
            list(range(68417, 68421)) + list(range(68424, 68428))

        pfiEven = [68360, 68361, 68373, 68388, 68391, 68398, 68399] + list(range(68480, 68489)) + \
            list(range(68491, 68494)) + list(range(68499, 68502))
        pfiOdd = list(range(68402, 68406)) + list(range(68504, 68507)) + list(range(68509, 68512)) + \
            list(range(68514, 68517))
        pfiBlack = list(range(68410, 68413)) + list(range(68417, 68421)) + list(range(68424, 68428)) + \
            list(set(list(range(68519, 68528))) ^ set([68522]))
        pfiAll = [68432, 68433, 68434, 68475, 68476, 68477]

        # First night with PFI installed without the proper pfsDesignId
        self.add("S", pfiEngineering, W_PFDSGN=0x71c4c79bd4b5d30)
        self.add("S", pfiEven, W_PFDSGN=0x1d317decdf389ea8)
        self.add("S", pfiOdd, W_PFDSGN=0x31b5b0ab7d661e15)
        self.add("S", pfiBlack, W_PFDSGN=0x2ed537b1a10ff1b2)
        self.add("S", pfiAll, W_PFDSGN=0x40dbf5546df0d55e)

        # Wrong hgcd status for all these visits.
        self.add("S", list(range(79990, 80852 + 1)), W_AITHGC=False)

        # Only one lamp was turned on each time.
        allLamps = ['W_AITNEO', 'W_AITKRY', 'W_AITXEN', 'W_AITARG']
        wrongs = list(range(79990, 79997 + 1))
        onlyLamps = allLamps * 2
        for visit, onlyLamp in zip(wrongs, onlyLamps):
            lampDict = dict([(lamp, False) for lamp in allLamps])
            lampDict[onlyLamp] = True
            self.add("S", [visit], **lampDict)

        # Correcting dcb header keys.
        self.add("S", list(range(81862, 81865 + 1)) + [81872, 81873] + list(range(81879, 81882 + 1)), W_AITQTH=True)
        self.add("S", [81869, 81874], W_AITKRY=True)
        self.add("S", [81870], W_AITNEO=True)
        self.add("S", [81871], W_AITARG=True)

        # Correcting dcb header keys.
        self.add("S", list(range(83098, 83100 + 1)) + list(range(83304, 83365 + 1)), W_AITQTH=True)
        self.add("S", list(range(83422, 83523 + 1)) + [83652] + list(range(83673, 83952 + 1)), W_AITQTH=True)
        self.add("S", [84004,84005], W_AITQTH=True)
        self.add("S", list(range(83058, 83060 + 1)) + list(range(83524, 83548 + 1)), W_AITKRY=True)
        self.add("S", [83651] + list(range(83653, 83662 + 1)) + [83367], W_AITKRY=True)
        self.add("S", list(range(83977, 83400 + 1)), W_AITKRY=True)
        self.add("S", list(range(83368, 83373 + 1)) + list(range(83549, 83578 + 1)), W_AITARG=True)
        self.add("S", list(range(83648, 83650 + 1)) + list(range(83663, 83672 + 1)), W_AITARG=True)
        self.add("S", list(range(83953, 83976 + 1)), W_AITARG=True)
        self.add("S", [83367] + list(range(83374, 83379 + 1)), W_AITHGA=True)
        self.add("S", list(range(83579, 83608 + 1)) + list(range(83637, 83647 + 1)), W_AITHGA=True)

        # Correcting domeFlat header keys
        self.add("S", [46084, 46085], W_AITQTH=True)
        self.add("S", list(range(46228, 46262 + 1)), W_AITQTH=True)
        self.add("S", list(range(63046, 63106 + 1)), W_AITQTH=True)
        self.add("S", [67709, 67710], W_AITQTH=True)
        self.add("S", list(range(68523, 68527 + 1)), W_AITQTH=True)
        self.add("S", list(range(75954, 75958 + 1)), W_AITQTH=True)
        self.add("S", [75966, 76143, 76144], W_AITQTH=True)
        self.add("S", list(range(76466, 76468 + 1)), W_AITQTH=True)
        self.add("S", [76559, 76560, 76763], W_AITQTH=True)
        self.add("S", list(range(77173, 77175 + 1)), W_AITQTH=True)
        self.add("S", [77404, 77405, 77457, 77464, 77466, 78196, 78248, 78705, 78759], W_AITQTH=True)
        self.add("S", list(range(80621, 80640 + 1)), W_AITQTH=True)
        self.add("S", list(range(81979, 81988 + 1)), W_AITQTH=True)
        self.add("S", list(range(82113, 82127 + 1)), W_AITQTH=True)
        self.add("S", list(range(82230, 82238 + 1)), W_AITQTH=True)
        self.add("S", [82428, 82429], W_AITQTH=True)
        self.add("S", list(range(82527, 82533 + 1)), W_AITQTH=True)
