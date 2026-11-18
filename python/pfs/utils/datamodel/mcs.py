from enum import IntFlag


class MCSSpotFlags(IntFlag):
    """
    Represents a bitmask for the saturation of the MCS spots: ``mcs_data.flags`` in opdb, and the
    ``flags`` column of the SPOTS HDU of a PFSC file, whose header names each bit (``FLAG<n>``)
    and gives the thresholds of the frame.

    A pixel is saturated when at least one of the reads summed in the coadd clipped. The flags of
    a spot count the saturated pixels within a radius of its centroid (the photometry aperture) and
    compare the count with three thresholds; a spot carrying BAD_CENTROID also carries
    BAD_PHOTOMETRY, and one carrying SATURATED carries both.

    Attributes:
        NONE: No saturated pixel near the spot, or fewer than the BAD_PHOTOMETRY threshold.
        NO_SAT_CHECK: The saturation of the frame was not measured; the other bits are then unset
            and say nothing.
        BAD_PHOTOMETRY: Enough saturated pixels that the flux is low by more than about 0.5%; the
            flux error includes a term for the saturated pixels.
        BAD_CENTROID: Enough saturated pixels that the centroid is biased by more than about 1 um.
            The mean bias is constant at a fixed flux and the frame-to-frame scatter grows by about
            half, so the spot remains usable.
        SATURATED: Enough saturated pixels that the frame-to-frame scatter of the centroid is at
            least twice that of an unsaturated spot.
    """

    NONE = 0x0000
    NO_SAT_CHECK = 0x0001
    BAD_PHOTOMETRY = 0x0002
    BAD_CENTROID = 0x0004
    SATURATED = 0x0008

    def __str__(self):
        if self == MCSSpotFlags.NONE:
            return "NONE"
        flags = [flag.name for flag in MCSSpotFlags if flag in self and flag != MCSSpotFlags.NONE]
        return "|".join(flags)
