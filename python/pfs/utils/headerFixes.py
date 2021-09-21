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

        pfiFirstNight = list(range(67569, 67574)) + [67587, 67588, 67594, 67605] + \
                        list(set(range(67611, 67651)) ^ set([67614, 67615, 67620])) + [67685, 67692, 67693]

        # First night with PFI installed without the proper pfsDesignId
        self.add(pfiFirstNight, W_PFDSGN=0x1f8dc068ce7f1647)
