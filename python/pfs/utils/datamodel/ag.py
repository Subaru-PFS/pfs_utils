import warnings
from enum import IntFlag


class SourceCatalogFlags(IntFlag):
    """
    Represents a bitmask for catalog guide star properties.

    This was formerly named AutoGuiderStarMask.

    See:

    https://irsa.ipac.caltech.edu/data/Gaia/dr3/gaia_dr3_source_colDescriptions.html
    https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue
    /ssec_dm_gaia_source.html

    Attributes:
        NONE: No properties. There shouldn't be any of these as everything is
            either GAIA or HSC.
        GAIA: Gaia DR3 catalog.
        HSC: HSC PDR3 catalog.
        PMRA: Proper motion RA is measured.
        PMRA_SIG: Proper motion RA measurement is significant (SNR>5).
        PMDEC: Proper motion Dec is measured.
        PMDEC_SIG: Proper motion Dec measurement is significant (SNR>5).
        PARA: Parallax is measured.
        PARA_SIG: Parallax measurement is significant (SNR>5).
        ASTROMETRIC: Astrometric excess noise is small (astrometric_excess_noise<1.0).
        ASTROMETRIC_SIG: Astrometric excess noise is significant (astrometric_excess_noise_sig>2.0).
        NON_BINARY: Not a binary system (RUWE<1.4).
        PHOTO_SIG: Photometric measurement is significant (SNR>5).
        GALAXY: Is a galaxy candidate.
    """

    NONE = 0x00000
    GAIA = 0x00001
    HSC = 0x00002
    PMRA = 0x00004
    PMRA_SIG = 0x00008
    PMDEC = 0x00010
    PMDEC_SIG = 0x00020
    PARA = 0x00040
    PARA_SIG = 0x00080
    ASTROMETRIC = 0x00100
    ASTROMETRIC_SIG = 0x00200
    NON_BINARY = 0x00400
    PHOTO_SIG = 0x00800
    GALAXY = 0x01000

    def __str__(self):
        if self == SourceCatalogFlags.NONE:
            return "NONE"
        flags = [flag.name for flag in SourceCatalogFlags if flag in self and flag != SourceCatalogFlags.NONE]
        return "|".join(flags)


class SourceDetectionFlags(IntFlag):
    """
    Represents a bitmask for detection properties.

    This was formerly named SourceDetectionFlag (singular).

    Attributes:
        NONE: No issues detected.
        RIGHT: Source is detected on the right side of the image, which has
            a piece of glass in front of the sensor.
        EDGE: Source is detected at the edge of the image.
        SATURATED: Source is saturated.
        BAD_SHAPE: Source has a bad shape.
        BAD_ELLIP: Source has a bad ellipticity.
        FLAT_TOP: Source has a flat top profile.
        BAD_SIZE: Source has a bad size.
    """

    NONE = 0x0000
    RIGHT = 0x0001
    EDGE = 0x0002
    SATURATED = 0x0004
    BAD_SHAPE = 0x0008
    BAD_ELLIP = 0x0010
    FLAT_TOP = 0x0020
    BAD_SIZE = 0x0040

    GOOD_DETECTION = NONE | RIGHT
    BAD_DETECTION = EDGE | SATURATED | BAD_SHAPE | BAD_ELLIP | FLAT_TOP | BAD_SIZE

    def __str__(self):
        if self == SourceDetectionFlags.NONE:
            return "NONE"
        flags = [flag.name for flag in SourceDetectionFlags if flag in self and flag != SourceDetectionFlags.NONE]
        return "|".join(flags)


class SourceMatchingFlags(IntFlag):
    """
    Represents a bitmask for matching properties between detected and catalog sources.

    Attributes:
        NONE: No matching attempted or unknown result.
        GOOD_MATCH: Source successfully matched with exactly one catalog entry
            within tolerance.
        NO_MATCH: Source has no matching catalog entry within tolerance.
        BAD_RESIDUAL: Source matched but has residual position difference
            larger than allowed threshold.
        UNUSED_MULTI_MATCH: Source was matched to a catalog entry, but wasn't the
            closest source so is unused.
    """

    NONE = 0x0000
    GOOD_MATCH = 0x0001
    NO_MATCH = 0x0002
    BAD_RESIDUAL = 0x0004
    UNUSED_MULTI_MATCH = 0x0008

    def __str__(self):
        if self == SourceMatchingFlags.NONE:
            return "NONE"
        flags = [flag.name for flag in SourceMatchingFlags if flag in self and flag != SourceMatchingFlags.NONE]
        return "|".join(flags)


def __getattr__(name):
    if name == "AutoGuiderStarMask":
        warnings.warn(
            "AutoGuiderStarMask is deprecated and will be removed in a future version. "
            "Please use SourceCatalogFlags instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SourceCatalogFlags

    if name == "SourceDetectionFlag":
        warnings.warn(
            "SourceDetectionFlag is deprecated and will be removed in a future version. "
            "Please use SourceDetectionFlags (plural) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SourceDetectionFlags

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return sorted([*list(globals().keys()), "AutoGuiderStarMask", "SourceDetectionFlag"])
