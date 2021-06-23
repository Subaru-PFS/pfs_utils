#!/usr/bin/env python

"""
Executable script to create a PfsDesign for a LAM exposure, given the colors
of fibers used.
"""

import sys
import numpy as np
from pfs.datamodel import PfsDesign, TargetType, FiberStatus, GuideStars
from .fibers import calculateFiberId

__all__ = ["HexIterator", "DummyCableBDatabase", "makePfsDesign"]


class HexIterator:
    """Iterator to provide hexadecimal values

    We use a particular digit (1, 2, 4, 8) and iterate first over where that
    digit is in the hexadecimal representation, and then over the digits. This
    isn't the way we normally count, but it provides a visual way of
    distinguishing elements by their position. It's also how we originally
    implemented the pfsDesignIds for the LAM colored cables (which can be
    combined, so the position of the 1s made it very easy to tell which is
    which), and maintaining backwards compatibility is important.

    Parameters
    ----------
    maxHexDigits : `int` Maximum number of hex digits to allow (attempts to use
        more will result in ``StopIteration`` when iterating).

    Examples
    --------

    >>> ["0x%03x" % vv for vv in HexIterator(3)]
    ['0x001', '0x010', '0x100', '0x002', '0x020', '0x200', '0x004', '0x040', \
'0x400', '0x008', '0x080', '0x800']
    """
    def __init__(self, maxHexDigits=16):
        self._exponent = 0
        self._maxHexDigits = maxHexDigits

    def __iter__(self):
        """Iterator"""
        return self

    def __next__(self):
        """Provide next value"""
        if self._exponent >= 4*self._maxHexDigits:
            raise StopIteration
        value = self.get()
        self.increment()
        return value

    def get(self):
        """Provide current value"""
        return 1 << (4*(self._exponent % self._maxHexDigits) + self._exponent//self._maxHexDigits)

    def increment(self):
        """Increment to next value"""
        self._exponent += 1


class DummyCableBDatabase:
    """Database of setups for Dummy Cable B

    Dummy Cable B (DCB) defines which fibers are illuminated on the spectrograph
    during engineering. We use this database of potential DCB setups to provide
    ``pfsDesignId`` values and fiberId lists for the construction of the
    `pfs.datamodel.PfsDesign`.

    Some DCB setups can be combined (i.e., multiple cables used simultaneously),
    so the ``pfsDesignId`` is the ``OR`` of the individual ``pfsDesignId``s.

    Attributes
    ----------
    names : `list` of `str`
        List of setup names.
    values : `list` of `int`
        List of associated ``pfsDesignId``s.
    fiberIds : `list` of iterables of `int`
        List of associated fiberIds.
    descriptions : `list` of `str`
        List of associated descriptions.
    """
    def __init__(self):
        self.names = []
        self.values = []
        self.fiberIds = []
        self.descriptions = []

        self.addStandardSet()
        self.validate()

    def validate(self):
        """Ensure everything is unique"""
        num = len(self.names)
        assert len(self.values) == num
        assert len(self.descriptions) == num
        assert len(self.fiberIds) == num
        assert len(set(self.names)) == num
        assert len(set(self.values)) == num
        assert len(set(self.descriptions)) == num
        value = 0
        for vv in self.values:
            assert (value & vv) == 0
            value |= vv

    def __len__(self):
        """Length"""
        return len(self.names)

    def add(self, name, value, description, fiberIds):
        """Add a setup

        Parameters
        ----------
        name : `str`
            Name of the setup.
        value : `int`
            Associated ``pfsDesignId`` value.
        description : `str`
            Description of the setup.
        fiberIds : iterable of `int`
            List of associated fiberIds.
        """
        self.names.append(name)
        self.values.append(value)
        self.descriptions.append(description)
        self.fiberIds.append(fiberIds)

    def addStandardSet(self):
        """Add the standard set of setups"""
        # NOTE: We use an iterator to produce the pfsDesignId values. If you are
        # going to use this scheme with new values, you should *ADD NEW ENTRIES
        # AT THE END* in order to maintain backward compatibility.
        hexIt = HexIterator()

        # Colored cables at LAM
        # Constructed from a snippet by Fabrice Madec, "dummy cable B fibers"
        # https://sumire-pfs.slack.com/files/U3MLENNHH/FFS6P4UR5/dummy_cable_b_fibers.txt
        self.add("blue", next(hexIt), "LAM blue cable", [32, 111, 223, 289, 418, 518, 621])
        self.add("green", next(hexIt), "LAM green cable", [63, 192, 255, 401, 464, 525, 587])
        self.add("orange", next(hexIt), "LAM orange cable", [12, 60, 110, 161, 210, 259, 341])
        self.add("red1", next(hexIt), "LAM red #1 cable", [2])
        self.add("red2", next(hexIt), "LAM red #2 cable", [3])
        self.add("red3", next(hexIt), "LAM red #3 cable", [308])
        self.add("red4", next(hexIt), "LAM red #4 cable", [339])
        self.add("red5", next(hexIt), "LAM red #5 cable", [340])
        self.add("red6", next(hexIt), "LAM red #6 cable", [342])
        self.add("red7", next(hexIt), "LAM red #7 cable", [649])
        self.add("red8", next(hexIt), "LAM red #8 cable", [650])
        self.add("yellow", next(hexIt), "LAM yellow cable", [360, 400, 449, 497, 545, 593, 643])

        # Additional LAM setups
        allFibers = list(range(1, 652))
        blank = ([44, 91, 93, 136, 183, 185, 228, 272] + list(range(317, 336)) +
                 [383, 427, 470, 472, 516, 559, 561, 608])
        engineering = [1, 45, 92, 137, 184, 229, 273, 316, 336, 382, 426, 471, 515, 560, 607, 651]
        science = [ff for ff in allFibers if ff not in set(engineering + blank)]
        self.add("engineering", next(hexIt), "Engineering", engineering)
        self.add("9mtp", next(hexIt), "Right science fibers", [ff for ff in science if ff < 273])
        self.add("12mtpS12", next(hexIt), "Left science fibers", [ff for ff in science if ff > 273])
        self.add("12mtpS34", next(hexIt), "Left science fibers without some",
                 [ff for ff in science if ff > 273 and ff not in set([281, 309, 359])])

    def check(self, *names):
        """Check that all the provided names are defined

        Parameters
        ----------
        *names : `str`
            Setups.

        Raises
        ------
        RuntimeError
            If any of the names aren't defined.
        """
        bad = set(names) - set(self.names)
        if bad:
            raise RuntimeError(f"Unrecognised setup names: {bad}")

    def getFiberIds(self, *names):
        """Convert a list of setups to an array of fiber IDs

        Parameters
        ----------
        *names : `str`
            Setups.

        Returns
        -------
        fiberId : `numpy.ndarray`
            Array of fiber IDs.
        """
        self.check(*names)
        names = set(names)
        return np.array(sorted(set(sum([ff for nn, ff in zip(self.names, self.fiberIds) if nn in names],
                                       []))))

    def getHash(self, *names):
        """Convert a list of setups to a hash for the pfsDesignId

        Parameters
        ----------
        *names : `str`
            Setups.

        Returns
        -------
        hash : `int`
            Hash, for the pfsDesignId.
        """
        self.check(*names)
        names = set(names)
        return sum(vv for nn, vv in zip(self.names, self.values) if nn in names)

    def interpret(self, value):
        """Interpret a hash value as a list of setups

        Parameters
        ----------
        value : `int`
            Value to interpret.

        Returns
        -------
        names : `list` of `str`
            List of setup names.
        """
        return [nn for nn, vv in zip(self.names, self.values) if (value & vv) > 0]


def makePfsDesign(pfsDesignId, fiberId, arms):
    """Build a ``Pfsdesign``

    Parameters
    ----------
    pfsDesignId : `int`
        Identifier for the top-end design. For our purposes, this is just a
        unique integer.
    fiberId : `numpy.ndarray` of `int`
        Array of identifiers for fibers that will be lit.
    arms : `str`
        One-character identifiers for arms that will be exposed,
        eg 'brn'.

    Returns
    -------
    design : `pfs.datamodel.Pfsdesign`
        designuration of the top-end.
    """
    raBoresight = 0.0
    decBoresight = 0.0
    posAng = 0.0
    tract = np.zeros_like(fiberId, dtype=int)
    patch = ["0,0" for _ in fiberId]

    num = len(fiberId)
    catId = np.zeros_like(fiberId, dtype=int)
    objId = fiberId
    targetTypes = TargetType.SCIENCE*np.ones_like(fiberId, dtype=int)
    fiberStatus = np.full_like(targetTypes, FiberStatus.GOOD)
    ra = np.zeros_like(fiberId, dtype=float)
    dec = np.zeros_like(fiberId, dtype=float)
    pfiNominal = np.zeros((num, 2), dtype=float)

    empty = [[] for _ in fiberId]

    return PfsDesign(pfsDesignId, raBoresight, decBoresight,
                     posAng, arms,
                     fiberId, tract, patch, ra, dec, catId, objId, targetTypes,
                     fiberStatus,
                     empty, empty, empty, empty, empty, empty, empty,
                     pfiNominal,
                     GuideStars.empty())


def main(args=None):
    """Command-line interface to create PfsDesign files

    Parameters
    ----------
    args : `list` of `str`, optional
        Command-line arguments to parse. If ``None``, will use ``sys.argv``.
    """
    import argparse

    dcb = DummyCableBDatabase()

    epilog = "Setups are\n" + "\n".join("    %s: %s" % vv for vv in zip(dcb.names, dcb.descriptions))
    parser = argparse.ArgumentParser(description="Create a PfsDesign for a LAM exposure, "
                                                 "given the Dummy Cable B setup.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=epilog)
    parser.add_argument("--directory", default=".", help="Directory in which to write file")
    parser.add_argument("setups", nargs="*", type=str,
                        help="Setup(s) specifying fibers that were lit")
    parser.add_argument("--arms", type=str, default='brn',
                        help="single-character identifier for arms that will be exposed, eg 'brn'.")
                
    args = parser.parse_args(args=args)

    nbad = 0
    for s in args.setups:
        if s not in dcb.names:
            print(f"Setup {s} is not valid (please choose one of {' '.join(dcb.names)})", file=sys.stderr)
            nbad += 1
    if nbad:
        sys.exit(1)

    fiberHoles = dcb.getFiberIds(*args.setups)
    numFibers = len(fiberHoles)
    spectrograph = np.array([1]*numFibers + [2]*numFibers + [3]*numFibers + [4]*numFibers)
    fiberId = calculateFiberId(spectrograph, np.concatenate([fiberHoles, fiberHoles, fiberHoles, fiberHoles]))

    pfsDesignId = dcb.getHash(*args.setups)
    design = makePfsDesign(pfsDesignId, fiberId, args.arms)
    design.write(dirName=args.directory)
    print("Wrote %s" % (design.filename,))


if __name__ == "__main__":
    main()
