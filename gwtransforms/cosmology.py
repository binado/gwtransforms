import astropy.cosmology
import astropy.units as u
from numpy.typing import ArrayLike


def dl_to_z(dl: float | ArrayLike, cosmology: astropy.cosmology.FLRW):
    return astropy.cosmology.z_at_value(cosmology.luminosity_distance, dl * u.Mpc).value
