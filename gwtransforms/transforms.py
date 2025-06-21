import weakref
from typing import Hashable, Sequence

import astropy.cosmology
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gwtransforms.conversions import (
    chirp_mass_and_symmetric_mass_ratio_to_component_masses,
    chirp_mass_and_symmetric_mass_ratio_to_total_mass_and_mass_ratio,
    component_masses_to_chirp_mass_and_symmetric_mass_ratio,
    component_masses_to_mass_ratio,
    component_masses_to_total_mass,
    primary_mass_and_mass_ratio_to_component_masses,
    symmetric_mass_ratio_to_mass_ratio,
    total_mass_and_mass_ratio_to_chirp_mass_and_symmetric_mass_ratio,
    total_mass_and_mass_ratio_to_component_masses,
)
from gwtransforms.cosmology import dl_to_z
from gwtransforms.utils import (
    combine_into_jacobian,
    combine_into_transform,
    stack_dict_keys_into_array,
    unpack_parameter_dim,
    unstack_array_to_dict,
)


class ParameterTransform:
    _inv = None
    inputs = None
    outputs = None

    def __init__(
        self,
        inputs: tuple[str, ...] | None = None,
        outputs: tuple[str, ...] | None = None,
    ) -> None:
        if inputs is not None:
            self.inputs = inputs

        if outputs is not None:
            self.outputs = outputs

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = weakref.ref(inv)
        return inv

    def __call__(self, x: NDArray) -> NDArray:
        raise NotImplementedError

    def _inverse(self, y: NDArray) -> NDArray:
        raise NotImplementedError

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        raise NotImplementedError

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        raise NotImplementedError

    def log_abs_det_jacobian(
        self, x: NDArray, y: NDArray, intermediates=None
    ) -> NDArray:
        raise NotImplementedError

    def transform_and_jacobian(self, x: NDArray) -> tuple[NDArray, NDArray]:
        y = self(x)
        jacobian = self.jacobian(x, y)
        return y, jacobian

    def apply_to_dict(self, d: dict[Hashable, ArrayLike]):
        if self.inputs is None:
            raise ValueError("inputs attribute should be set")
        x = stack_dict_keys_into_array(d, *self.inputs)
        y = self(x)
        return unstack_array_to_dict(d, y, *self.inputs)


class ComposeTransform(ParameterTransform):
    _inv = None

    def __init__(self, *transforms: ParameterTransform):
        self.transforms = transforms
        self.inputs = self.transforms[0].inputs
        self.outputs = self.transforms[-1].outputs

    def __call__(self, x: NDArray) -> NDArray:
        for transform in self.transforms:
            x = transform(x)
        return x

    def _inverse(self, y: NDArray) -> NDArray:
        for transform in reversed(self.transforms):
            y = transform._inverse(y)
        return y

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        jac = np.eye(x.shape[0])
        for transform in self.transforms:
            y_aux = transform(x)
            jac = transform.jacobian(x, y_aux) @ jac
            x = y_aux
        return jac

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        jac = np.eye(x.shape[0], x.shape[0])
        for transform in self.transforms:
            y_aux = transform(x)
            jac = jac @ transform.inverse_jacobian(x, y_aux)
            x = y_aux
        return jac


class _InverseTransform(ParameterTransform):
    def __init__(self, transform: ParameterTransform) -> None:
        super().__init__()
        self._inv = transform
        self.inputs = self.inv.outputs
        self.outputs = self.inv.inputs

    @property
    def inv(self):
        return self._inv

    def __call__(self, x: NDArray) -> NDArray:
        return self._inv._inverse(x)

    def _inverse(self, y: NDArray) -> NDArray:
        return self._inv(y)

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        return self._inv.inverse_jacobian(y, x)

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        return self._inv.jacobian(y, x)

    def log_abs_det_jacobian(
        self, x: NDArray, y: NDArray, intermediates=None
    ) -> NDArray:
        # NB: we don't use intermediates for inverse transform
        return -self._inv.log_abs_det_jacobian(y, x, None)


class ComponentMassesToTotalMassAndMassRatio(ParameterTransform):
    inputs = ("mass_1", "mass_2")
    outputs = ("total_mass", "mass_ratio")

    def __call__(self, x: NDArray) -> NDArray:
        mass_1, mass_2 = unpack_parameter_dim(x)
        total_mass = component_masses_to_total_mass(mass_1, mass_2)
        mass_ratio = component_masses_to_mass_ratio(mass_1, mass_2)
        return combine_into_transform(total_mass, mass_ratio)

    def _inverse(self, y: NDArray) -> NDArray:
        total_mass, mass_ratio = unpack_parameter_dim(y)
        mass_1, mass_2 = total_mass_and_mass_ratio_to_component_masses(
            total_mass, mass_ratio
        )
        return combine_into_transform(mass_1, mass_2)

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        mass_1, _ = unpack_parameter_dim(x)
        _, mass_ratio = unpack_parameter_dim(y)
        d_total_mass_d_mass_1 = d_total_mass_d_mass_2 = np.ones_like(mass_1)
        d_mass_ratio_d_mass_1 = -mass_ratio / mass_1
        d_mass_ratio_d_mass_2 = 1 / mass_1
        return combine_into_jacobian(
            d_total_mass_d_mass_1,
            d_total_mass_d_mass_2,
            d_mass_ratio_d_mass_1,
            d_mass_ratio_d_mass_2,
        )

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        total_mass, mass_ratio = unpack_parameter_dim(y)
        d_mass_1_d_total_mass = 1 / (1 + mass_ratio)
        d_mass_1_d_mass_ratio = -total_mass / (1 + mass_ratio) ** 2
        d_mass_2_d_total_mass = mass_ratio / (1 + mass_ratio)
        d_mass_2_d_mass_ratio = -d_mass_1_d_mass_ratio
        return combine_into_jacobian(
            d_mass_1_d_total_mass,
            d_mass_1_d_mass_ratio,
            d_mass_2_d_total_mass,
            d_mass_2_d_mass_ratio,
        )


class ComponentMassesToPrimaryMassAndMassRatio(ParameterTransform):
    inputs = ("mass_1", "mass_2")
    outputs = ("mass_1", "mass_ratio")

    def __call__(self, x: NDArray) -> NDArray:
        mass_1, mass_2 = unpack_parameter_dim(x)
        mass_ratio = component_masses_to_mass_ratio(mass_1, mass_2)
        return combine_into_transform(mass_1, mass_ratio)

    def _inverse(self, y: NDArray) -> NDArray:
        mass_1, mass_ratio = unpack_parameter_dim(y)
        mass_1, mass_2 = primary_mass_and_mass_ratio_to_component_masses(
            mass_1, mass_ratio
        )
        return combine_into_transform(mass_1, mass_2)

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        mass_1, mass_2 = unpack_parameter_dim(x)
        _, mass_ratio = unpack_parameter_dim(y)
        d_mass_1_d_mass_1 = np.ones_like(mass_1)
        d_mass_1_d_mass_2 = np.zeros_like(mass_2)
        d_mass_ratio_d_mass_1 = -mass_ratio / mass_1
        d_mass_ratio_d_mass_2 = 1 / mass_1
        return combine_into_jacobian(
            d_mass_1_d_mass_1,
            d_mass_1_d_mass_2,
            d_mass_ratio_d_mass_1,
            d_mass_ratio_d_mass_2,
        )

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        mass_1, _ = unpack_parameter_dim(x)
        _, mass_ratio = unpack_parameter_dim(y)
        d_mass_1_d_mass_1 = np.ones_like(mass_1)
        d_mass_1_d_mass_ratio = np.zeros_like(mass_1)
        d_mass_2_d_mass_1 = mass_ratio
        d_mass_2_d_mass_ratio = mass_1
        return combine_into_jacobian(
            d_mass_1_d_mass_1,
            d_mass_1_d_mass_ratio,
            d_mass_2_d_mass_1,
            d_mass_2_d_mass_ratio,
        )


class TotalMassAndMassRatioToChirpMassAndSymmetricMassRatio(ParameterTransform):
    inputs = ("total_mass", "mass_ratio")
    outputs = ("chirp_mass", "symmetric_mass_ratio")

    def __call__(self, x: NDArray) -> NDArray:
        total_mass, mass_ratio = unpack_parameter_dim(x)
        chirp_mass, symmetric_mass_ratio = (
            total_mass_and_mass_ratio_to_chirp_mass_and_symmetric_mass_ratio(
                total_mass, mass_ratio
            )
        )
        return combine_into_transform(chirp_mass, symmetric_mass_ratio)

    def _inverse(self, y: NDArray) -> NDArray:
        chirp_mass, symmetric_mass_ratio = unpack_parameter_dim(y)
        total_mass, mass_ratio = (
            chirp_mass_and_symmetric_mass_ratio_to_total_mass_and_mass_ratio(
                chirp_mass, symmetric_mass_ratio
            )
        )
        return combine_into_transform(total_mass, mass_ratio)

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        _, mass_ratio = unpack_parameter_dim(x)
        chirp_mass, symmetric_mass_ratio = unpack_parameter_dim(y)
        d_chirp_mass_d_total_mass = symmetric_mass_ratio ** (3 / 5)
        d_chirp_mass_d_mass_ratio = (
            -3 * chirp_mass * (mass_ratio - 1) / (5 * mass_ratio * (mass_ratio + 1))
        )
        d_symmetric_mass_ratio_d_total_mass = np.zeros_like(symmetric_mass_ratio)
        d_symmetric_mass_ratio_d_mass_ratio = (1 - mass_ratio) / (1 + mass_ratio) ** 3
        return combine_into_jacobian(
            d_chirp_mass_d_total_mass,
            d_chirp_mass_d_mass_ratio,
            d_symmetric_mass_ratio_d_total_mass,
            d_symmetric_mass_ratio_d_mass_ratio,
        )

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        _, mass_ratio = unpack_parameter_dim(x)
        chirp_mass, symmetric_mass_ratio = unpack_parameter_dim(y)
        d_total_mass_d_chirp_mass = symmetric_mass_ratio ** (-3 / 5)
        d_total_mass_d_symmetric_mass_ratio = (
            (-3 / 5) * chirp_mass * symmetric_mass_ratio ** (-8 / 5)
        )
        d_mass_ratio_d_chirp_mass = np.zeros_like(mass_ratio)
        d_mass_ratio_d_symmetric_mass_ratio = -mass_ratio / (
            symmetric_mass_ratio * np.sqrt(1 - 4 * symmetric_mass_ratio)
        )
        return combine_into_jacobian(
            d_total_mass_d_chirp_mass,
            d_total_mass_d_symmetric_mass_ratio,
            d_mass_ratio_d_chirp_mass,
            d_mass_ratio_d_symmetric_mass_ratio,
        )


class ComponentMassesToChirpMassAndSymmetricMassRatio(ParameterTransform):
    inputs = ("mass_1", "mass_2")
    outputs = ("chirp_mass", "symmetric_mass_ratio")

    def __call__(self, x: NDArray) -> NDArray:
        mass_1, mass_2 = unpack_parameter_dim(x)
        chirp_mass, symmetric_mass_ratio = (
            component_masses_to_chirp_mass_and_symmetric_mass_ratio(mass_1, mass_2)
        )
        return combine_into_transform(chirp_mass, symmetric_mass_ratio)

    def _inverse(self, y: NDArray) -> NDArray:
        chirp_mass, symmetric_mass_ratio = unpack_parameter_dim(y)
        mass_1, mass_2 = chirp_mass_and_symmetric_mass_ratio_to_component_masses(
            chirp_mass, symmetric_mass_ratio
        )
        return combine_into_transform(mass_1, mass_2)

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        mass_1, mass_2 = unpack_parameter_dim(x)
        _, symmetric_mass_ratio = unpack_parameter_dim(y)
        total_mass = component_masses_to_total_mass(mass_1, mass_2)
        d_chirp_mass_d_mass_1 = (
            symmetric_mass_ratio ** (3 / 5) * (2 * mass_1 + 3 * mass_2) / (5 * mass_1)
        )
        d_chirp_mass_d_mass_2 = (
            symmetric_mass_ratio ** (3 / 5) * (3 * mass_1 + 2 * mass_2) / (5 * mass_2)
        )
        d_symmetric_mass_ratio_mass_1 = mass_2 * (mass_2 - mass_1) / total_mass**3
        d_symmetric_mass_ratio_mass_2 = mass_1 * (mass_1 - mass_2) / total_mass**3
        return combine_into_jacobian(
            d_chirp_mass_d_mass_1,
            d_chirp_mass_d_mass_2,
            d_symmetric_mass_ratio_mass_1,
            d_symmetric_mass_ratio_mass_2,
        )

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        chirp_mass, nu = unpack_parameter_dim(y)
        total_mass, mass_ratio = (
            chirp_mass_and_symmetric_mass_ratio_to_total_mass_and_mass_ratio(
                chirp_mass, nu
            )
        )
        d_mass_1_d_chirp_mass = 2 * nu ** (2 / 5) / (np.sqrt(1 - 4 * nu) + 1)
        d_mass_2_d_chirp_mass = mass_ratio * d_mass_1_d_chirp_mass
        d_total_mass_d_symmetric_mass_ratio = -0.6 * total_mass / nu
        d_mass_1_d_symmetric_mass_ratio = d_total_mass_d_symmetric_mass_ratio / (
            1 + mass_ratio
        ) - total_mass * (1 + mass_ratio) / (1 - mass_ratio)
        d_mass_2_d_symmetric_mass_ratio = (
            d_total_mass_d_symmetric_mass_ratio - d_mass_1_d_symmetric_mass_ratio
        )
        return combine_into_jacobian(
            d_mass_1_d_chirp_mass,
            d_mass_1_d_symmetric_mass_ratio,
            d_mass_2_d_chirp_mass,
            d_mass_2_d_symmetric_mass_ratio,
        )


class SourceFrameToDetectorFrameMasses(ParameterTransform):
    dimensionful_detector_frame_mass_variables = (
        "mass_1",
        "mass_2",
        "total_mass",
        "chirp_mass",
    )

    def __init__(
        self,
        inputs: tuple[str, str],
        outputs: tuple[str, str],
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self._validate_inputs_and_outputs()

    @property
    def dimensionful_source_frame_mass_variables(self) -> tuple[str, ...]:
        _df_vars = self.dimensionful_detector_frame_mass_variables
        return tuple([f"{mass}_source" for mass in _df_vars])

    def _validate_inputs_and_outputs(self) -> None:
        df_masses = self.dimensionful_detector_frame_mass_variables
        sf_masses = self.dimensionful_source_frame_mass_variables
        if len(self.inputs) != 2:
            raise ValueError("Expected input argument to have two elements")
        if self.inputs[0] not in sf_masses:
            raise ValueError(
                f"First element of input argument should be in {sf_masses}"
            )
        if self.inputs[1] != "redshift":
            raise ValueError("Second element of input argument should be 'redshift'")

        if len(self.outputs) != 2:
            raise ValueError("Expected output argument to have two elements")
        if self.outputs[0] not in df_masses:
            raise ValueError(
                f"First element of output argument should be in {df_masses}"
            )
        if self.outputs[1] != "redshift":
            raise ValueError("Second element of output argument should be 'redshift'")

    def __call__(self, x: NDArray) -> NDArray:
        mass_source, redshift = unpack_parameter_dim(x)
        return combine_into_transform(mass_source * (1 + redshift), redshift)

    def inverse(self, y: NDArray) -> NDArray:
        mass, redshift = unpack_parameter_dim(y)
        return combine_into_transform(mass / (1 + redshift), redshift)

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        source_mass, redshift = unpack_parameter_dim(x)
        d_mass_d_source_mass = 1 + redshift
        d_mass_d_redshift = source_mass
        d_redshift_d_source_mass = np.zeros_like(redshift)
        d_redshift_d_redshift = np.ones_like(redshift)
        return combine_into_jacobian(
            d_mass_d_source_mass,
            d_mass_d_redshift,
            d_redshift_d_source_mass,
            d_redshift_d_redshift,
        )

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        mass, redshift = unpack_parameter_dim(y)
        d_source_mass_d_mass = 1 / (1 + redshift)
        d_source_mass_d_redshift = -mass / (1 + redshift) ** 2
        d_redshift_d_mass = np.zeros_like(redshift)
        d_redshift_d_redshift = np.ones_like(redshift)
        return combine_into_jacobian(
            d_source_mass_d_mass,
            d_source_mass_d_redshift,
            d_redshift_d_mass,
            d_redshift_d_redshift,
        )


class RedshiftToLuminosityDistance(ParameterTransform):
    inputs = ("redshift",)
    outputs = ("luminosity_distance",)

    def __init__(self, cosmology: astropy.cosmology.FLRW):
        self.cosmology = cosmology

    def ddl_dz(self, z: NDArray) -> NDArray:
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

    def __call__(self, x: NDArray) -> NDArray:
        return self.cosmology.luminosity_distance(x).value

    def _inverse(self, y: NDArray) -> NDArray:
        return dl_to_z(y, self.cosmology)

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        return self.ddl_dz(y).reshape(-1, 1, 1)

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        return (1 / self.ddl_dz(y)).reshape(-1, 1, 1)


def construct_jacobian(
    transforms: Sequence[ParameterTransform], params: NDArray, dims: dict[str, int]
) -> tuple[NDArray, dict[str, int], NDArray]:
    ndim = len(dims)
    _params = params.copy()
    if params.shape[-1] != ndim:
        raise ValueError("Last axis of params array should match dims")

    batch_shape = _params.shape[:-1]
    total_jac = np.broadcast_to(np.eye(ndim), (*batch_shape, ndim, ndim)).copy()
    _dims = dims.copy()

    for transform in transforms:
        inputs = transform.inputs
        outputs = transform.outputs
        if inputs is None or outputs is None:
            raise ValueError(
                f"inputs and outputs must be set for transform {transform}"
            )
        if len(inputs) != len(outputs):
            raise ValueError(
                f"inputs and outputs should have the same length for transform {transform}"
            )

        ntransform = len(inputs)
        input_dim_indices = [_dims[i] for i in inputs]
        x = _params[..., input_dim_indices]
        y, jac = transform.transform_and_jacobian(x)
        # Update total jacobian J_T with J_T = J_i @ J_T
        # J_i will be a block matrix, J_i = jac, identity elsewhere
        # If we compute J_T = (I + (J_i - I)) @ J_T = J_T + (J_I - I) @ J_T,
        # The matrix J_I - I is non-zero only in the input indices
        # Hence J_T += (jac - I) @ J_T
        # We update J_T using the sparse structure of (J_i - I):
        # J_T[inputs, :] += (J_i - I)[inputs, inputs] @ J_T[inputs, :]
        # In practice, jac has shape (..., ntransform, ntransform)
        # J_T (i, j) += jac_m_I (i, k) @ J_T (k, j)
        # j \in (0, ndim), i,k \in input_dim_indices
        # ik_slice = (..., i or k, j)
        ik_slice = (..., input_dim_indices, slice(None))
        total_jac[ik_slice] += (jac - np.eye(ntransform)) @ total_jac[ik_slice]

        _params[..., input_dim_indices] = y
        other_dims = {k: v for k, v in _dims.items() if k not in inputs}
        new_dims = {o: _dims[i] for i, o in zip(inputs, outputs)}
        _dims = {**other_dims, **new_dims}

    return _params, _dims, total_jac
