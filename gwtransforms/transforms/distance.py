import astropy.cosmology
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gwtransforms.transforms.base import ParameterTransform


def dl_to_z(dl: float | ArrayLike, cosmology: astropy.cosmology.FLRW):
    import astropy.units as u

    return astropy.cosmology.z_at_value(cosmology.luminosity_distance, dl * u.Mpc).value


class RedshiftToLuminosityDistance(ParameterTransform):
    inputs = ("redshift",)
    outputs = ("luminosity_distance",)

    def __init__(self, cosmology: astropy.cosmology.FLRW):
        self.cosmology = cosmology

    def ddl_dz(self, z: ArrayLike) -> NDArray:
        z = np.asanyarray(z)
        dc = self.cosmology.comoving_distance(z)
        dm = self.cosmology.comoving_transverse_distance(z)
        dh = self.cosmology.hubble_distance
        Ok0 = self.cosmology.Ok0
        sqrt_Ok = np.sqrt(abs(Ok0))
        ddc_dz = dh * self.cosmology.inv_efunc(z)
        if Ok0 == 0:
            curv_factor = 1
        else:
            rescaled_dh = dh / sqrt_Ok
            curv_factor = np.where(
                Ok0 > 0, np.cos(dc / rescaled_dh), np.cosh(dc / rescaled_dh)
            )

        ddm_dz = ddc_dz * curv_factor
        return ((1 + z) * ddm_dz + dm).value

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.cosmology.luminosity_distance(x).value

    def _inverse(self, y: ArrayLike) -> ArrayLike:
        return dl_to_z(y, self.cosmology)

    def jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return self.ddl_dz(y).reshape(-1, 1, 1)

    def _inverse_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return (1 / self.ddl_dz(y)).reshape(-1, 1, 1)

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        jacobian = self.jacobian(x, y)
        return np.log(np.absolute(jacobian))
