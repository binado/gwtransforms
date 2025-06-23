import numpy as np
import pytest
from astropy.cosmology import Planck18

from gwtransforms.transforms import (
    ComponentMassesToChirpMassAndSymmetricMassRatio,
    ComponentMassesToPrimaryMassAndMassRatio,
    ComponentMassesToTotalMassAndMassRatio,
    RedshiftToLuminosityDistance,
    TotalMassAndMassRatioToChirpMassAndSymmetricMassRatio,
    construct_jacobian,
)
from gwtransforms.utils import stack_dict_keys_into_array


@pytest.mark.parametrize(
    "transform,transform_name",
    [
        (
            ComponentMassesToPrimaryMassAndMassRatio(),
            "component_masses_to_primary_mass_and_mass_ratio",
        ),
        (
            ComponentMassesToTotalMassAndMassRatio(),
            "component_masses_to_total_mass_and_mass_ratio",
        ),
        (
            ComponentMassesToChirpMassAndSymmetricMassRatio(),
            "component_masses_to_chirp_mass_and_symmetric_mass_ratio",
        ),
    ],
)
def test_analytical_mass_transforms(
    transform, transform_name, symbolic_transforms, mass_arrays
):
    x = stack_dict_keys_into_array(mass_arrays, *transform.inputs)
    symbolic_transform = symbolic_transforms[transform_name]
    assert transform.inputs == symbolic_transform.inputs
    assert transform.outputs == symbolic_transform.outputs
    y, sym_y = transform(x), symbolic_transform(x)
    assert np.allclose(y, sym_y)
    x_from_inv = transform._inverse(y)
    x_sym_from_inv = symbolic_transform._inverse(sym_y)
    assert np.allclose(x_from_inv, x_sym_from_inv)
    jacobian = transform.jacobian(x, y)
    sym_jacobian = symbolic_transform.jacobian(x, sym_y)
    assert np.allclose(jacobian, sym_jacobian)
    inv_jacobian = transform._inverse_jacobian(x, y)
    sym_inv_jacobian = symbolic_transform._inverse_jacobian(x, sym_y)
    assert np.allclose(inv_jacobian, sym_inv_jacobian)


def test_construct_jacobian():
    batch_shape = (100,)
    m1 = 30 * np.ones(batch_shape)
    m2 = 20 * np.ones(batch_shape)
    redshift = np.geomspace(1e-4, 1e0, batch_shape[0])
    params = np.stack((m1, m2, redshift), axis=-1)
    dims = {"mass_1": 0, "mass_2": 1, "redshift": 2}
    redshift_to_luminosity_distance_transform = RedshiftToLuminosityDistance(
        cosmology=Planck18
    )
    transforms = [
        ComponentMassesToTotalMassAndMassRatio(),
        TotalMassAndMassRatioToChirpMassAndSymmetricMassRatio(),
        redshift_to_luminosity_distance_transform,
    ]
    combined_transform = ComponentMassesToChirpMassAndSymmetricMassRatio()
    new_params, new_dims, jacobian = construct_jacobian(transforms, params, dims)
    new_masses = combined_transform(params[:, :2])
    combined_mass_jacobian = combined_transform.jacobian(params[:, :2], new_masses)
    dl = redshift_to_luminosity_distance_transform(params[:, -1])
    ddl_dz = redshift_to_luminosity_distance_transform.jacobian(params[:, -1], dl)
    ddl_dz = np.asanyarray(ddl_dz)
    assert new_dims == {
        "chirp_mass": 0,
        "symmetric_mass_ratio": 1,
        "luminosity_distance": 2,
    }
    assert np.allclose(new_masses, new_params[:, :2])
    assert np.allclose(combined_mass_jacobian, jacobian[:, :2, :2])
    assert np.allclose(jacobian[:, :2, -1], 0)
    assert np.allclose(jacobian[:, -1, :2], 0)
    assert np.allclose(dl, new_params[:, -1])
    assert np.allclose(jacobian[:, -1, -1], ddl_dz[:, -1, -1])
