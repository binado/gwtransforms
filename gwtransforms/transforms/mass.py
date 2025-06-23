import numpy as np
from numpy.typing import ArrayLike

from gwtransforms.conversions import (
    chirp_mass_and_symmetric_mass_ratio_to_component_masses,
    chirp_mass_and_symmetric_mass_ratio_to_total_mass_and_mass_ratio,
    component_masses_to_chirp_mass_and_symmetric_mass_ratio,
    component_masses_to_mass_ratio,
    component_masses_to_total_mass,
    primary_mass_and_mass_ratio_to_component_masses,
    total_mass_and_mass_ratio_to_chirp_mass_and_symmetric_mass_ratio,
    total_mass_and_mass_ratio_to_component_masses,
)
from gwtransforms.transforms.base import ParameterTransform
from gwtransforms.utils import (
    combine_into_jacobian,
    combine_into_transform,
    unpack_parameter_dim,
)


class ComponentMassesToTotalMassAndMassRatio(ParameterTransform):
    inputs = ("mass_1", "mass_2")
    outputs = ("total_mass", "mass_ratio")

    def __call__(self, x: ArrayLike) -> ArrayLike:
        mass_1, mass_2 = unpack_parameter_dim(x)
        total_mass = component_masses_to_total_mass(mass_1, mass_2)
        mass_ratio = component_masses_to_mass_ratio(mass_1, mass_2)
        return combine_into_transform(total_mass, mass_ratio)

    def _inverse(self, y: ArrayLike) -> ArrayLike:
        total_mass, mass_ratio = unpack_parameter_dim(y)
        mass_1, mass_2 = total_mass_and_mass_ratio_to_component_masses(
            total_mass, mass_ratio
        )
        return combine_into_transform(mass_1, mass_2)

    def jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def _inverse_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def __call__(self, x: ArrayLike) -> ArrayLike:
        mass_1, mass_2 = unpack_parameter_dim(x)
        mass_ratio = component_masses_to_mass_ratio(mass_1, mass_2)
        return combine_into_transform(mass_1, mass_ratio)

    def _inverse(self, y: ArrayLike) -> ArrayLike:
        mass_1, mass_ratio = unpack_parameter_dim(y)
        mass_1, mass_2 = primary_mass_and_mass_ratio_to_component_masses(
            mass_1, mass_ratio
        )
        return combine_into_transform(mass_1, mass_2)

    def jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def _inverse_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def __call__(self, x: ArrayLike) -> ArrayLike:
        total_mass, mass_ratio = unpack_parameter_dim(x)
        chirp_mass, symmetric_mass_ratio = (
            total_mass_and_mass_ratio_to_chirp_mass_and_symmetric_mass_ratio(
                total_mass, mass_ratio
            )
        )
        return combine_into_transform(chirp_mass, symmetric_mass_ratio)

    def _inverse(self, y: ArrayLike) -> ArrayLike:
        chirp_mass, symmetric_mass_ratio = unpack_parameter_dim(y)
        total_mass, mass_ratio = (
            chirp_mass_and_symmetric_mass_ratio_to_total_mass_and_mass_ratio(
                chirp_mass, symmetric_mass_ratio
            )
        )
        return combine_into_transform(total_mass, mass_ratio)

    def jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def _inverse_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def __call__(self, x: ArrayLike) -> ArrayLike:
        mass_1, mass_2 = unpack_parameter_dim(x)
        chirp_mass, symmetric_mass_ratio = (
            component_masses_to_chirp_mass_and_symmetric_mass_ratio(mass_1, mass_2)
        )
        return combine_into_transform(chirp_mass, symmetric_mass_ratio)

    def _inverse(self, y: ArrayLike) -> ArrayLike:
        chirp_mass, symmetric_mass_ratio = unpack_parameter_dim(y)
        mass_1, mass_2 = chirp_mass_and_symmetric_mass_ratio_to_component_masses(
            chirp_mass, symmetric_mass_ratio
        )
        return combine_into_transform(mass_1, mass_2)

    def jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def _inverse_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def __call__(self, x: ArrayLike) -> ArrayLike:
        mass_source, redshift = unpack_parameter_dim(x)
        return combine_into_transform(mass_source * (1 + redshift), redshift)

    def inverse(self, y: ArrayLike) -> ArrayLike:
        mass, redshift = unpack_parameter_dim(y)
        return combine_into_transform(mass / (1 + redshift), redshift)

    def jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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

    def _inverse_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
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
