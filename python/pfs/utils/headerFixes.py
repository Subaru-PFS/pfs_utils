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
        self.fixes = defaultdict(dict)  # visit --> header fixes to apply
        self.addStandardFixes()

    def add(self, visits, **fixes):
        """Add a set of fixes for a list of visits

        Parameters
        ----------
        visits : iterable or `int`
            List of visits (or a single visit) for which the fixes apply.
        **fixes : `dict` mapping `str` to `str`/`int`/`float`
            Header fixes to apply.
        """
        for vv in visits if isinstance(visits, Iterable) else [visits]:
            self.fixes[vv].update(fixes)

    def write(self, path):
        """Write the database

        The database is written as one YAML file for each visit; this is the
        format mandated by astro_metadata_translator.

        Parameters
        ----------
        path : `str`
            Path to which to write correction files.
        """
        for visit, fixes in self.fixes.items():
            filename = os.path.join(path, "PFS-%s.yaml" % (visit,))
            with open(filename, "w") as ff:
                yaml.dump(fixes, ff)

    def addStandardFixes(self):
        """Add the standard set of fixes

        This is the official list of header fixes to apply.
        """
        # Fix duplicate no-value (DM-23928)
        # Undefined values occurred in LAM data
        # between 2019-05-03 and 2019-06-14
        self.add(range(16804, 20961), W_XHP2FR=0)

        # For Subaru exposures taken during Dec 2019, exposures
        # labelled as Neon were actually Krypton
        self.add(list(range(423, 442)) + [43, 54, 60], W_AITNEO=False, W_AITKRY=True)

        # Early SuNSS observations without the proper pfsDesignId
        self.add(range(45752, 45853), W_PFDSGN=0xdeadbeef)

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
                 list( range(68514, 68517))
        pfiBlack = list(range(68410, 68413)) + list(range(68417, 68421)) + list(range(68424, 68428)) + \
                   list(set(list(range(68519, 68528))) ^ set([68522]))
        pfiAll = [68432, 68433, 68434, 68475, 68476, 68477]

        # First night with PFI installed without the proper pfsDesignId
        self.add(pfiEngineering, W_PFDSGN=0x71c4c79bd4b5d30)
        self.add(pfiEven, W_PFDSGN=0x1d317decdf389ea8)
        self.add(pfiOdd, W_PFDSGN=0x31b5b0ab7d661e15)
        self.add(pfiBlack, W_PFDSGN=0x2ed537b1a10ff1b2)
        self.add(pfiAll, W_PFDSGN=0x40dbf5546df0d55e)
