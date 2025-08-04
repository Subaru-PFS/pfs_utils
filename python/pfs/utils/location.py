from dataclasses import dataclass

from astropy import units as u
from astropy.coordinates import Angle, EarthLocation


@dataclass(frozen=True)
class TelescopeLocation:
    """A dataclass representing the location of a telescope on Earth.

    This class stores the name, height, latitude, and longitude of a telescope
    and provides a property to convert these values to an astropy EarthLocation object.

    Note
    ----
    Most users should not create an instance of this class but instead use the
    pre-defined SUBARU instance, i.e. `from pfs.utils.location import SUBARU`.

    Parameters
    ----------
    name : str
        The name of the telescope.
    height : u.Quantity
        The height of the telescope above sea level, as an astropy Quantity with
        units of length (e.g., meters).
    latitude : Angle
        The latitude of the telescope, as an astropy Angle.
    longitude : Angle
        The longitude of the telescope, as an astropy Angle.

    """

    name: str
    height: u.Quantity
    latitude: Angle
    longitude: Angle

    @property
    def location(self) -> EarthLocation:
        """Convert to an astropy EarthLocation object.

        Returns
        -------
        EarthLocation
            An astropy EarthLocation object representing the telescope's position
            on Earth.
        """
        return EarthLocation(lat=self.latitude, lon=self.longitude, height=self.height)


# Create an instance for Subaru
SUBARU = TelescopeLocation(
    name="Subaru Telescope",
    height=4163 * u.m,
    latitude=Angle("+19:49:31.8", unit=u.deg),
    longitude=Angle("-155:28:33.7", unit=u.deg),
)
