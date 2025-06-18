from enum import IntFlag


class AutoGuiderStarMask(IntFlag):
    """
    Represents a bitmask for guide star properties.

    Attributes:
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


class SourceDetectionFlag(IntFlag):
    """
    Represents a bitmask for detection properties.

    Attributes:
        RIGHT: Source is detected on the right side of the image.
        EDGE: Source is detected at the edge of the image.
        SATURATED: Source is saturated.
        BAD_SHAPE: Source has a bad shape.
        BAD_ELLIP: Source has a bad ellipticity.
        FLAT_TOP: Source has a flat top profile.
    """
    RIGHT = 0x0001
    EDGE = 0x0002
    SATURATED = 0x0004
    BAD_SHAPE = 0x0008
    BAD_ELLIP = 0x0010
    FLAT_TOP = 0x0020
