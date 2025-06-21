import numpy as np
import pytest
import sympy as sp
from astropy.cosmology import Planck18

from gwtransforms.symbolic import get_all_mass_symbolic_transforms
from gwtransforms.transforms import (
    ComponentMassesToChirpMassAndSymmetricMassRatio,
    ComponentMassesToPrimaryMassAndMassRatio,
    ComponentMassesToTotalMassAndMassRatio,
    RedshiftToLuminosityDistance,
    SourceFrameToDetectorFrameMasses,
    TotalMassAndMassRatioToChirpMassAndSymmetricMassRatio,
    construct_jacobian,
)
from gwtransforms.utils import stack_dict_keys_into_array

symbolic_transforms = get_all_mass_symbolic_transforms()


@pytest.fixture
def mass_arrays():
    m1 = 30
    m2 = 20
    q = m2 / m1
    mt = m1 + m2
    nu = q / (1 + q) ** 2
    mc = mt * nu**0.6
    batch_shape = (10,)
    inputs = (
        "mass_1",
        "mass_2",
        "mass_ratio",
        "total_mass",
        "symmetric_mass_ratio",
        "chirp_mass",
    )
    arrays = tuple(map(lambda x: x * np.ones(batch_shape), (m1, m2, q, mt, nu, mc)))
    return dict(zip(inputs, arrays))


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
def test_analytical_mass_transforms(transform, transform_name, mass_arrays):
    x = stack_dict_keys_into_array(mass_arrays, *transform.inputs)
    symbolic_transform = symbolic_transforms[transform_name]
    assert transform.inputs == symbolic_transform.inputs
    assert transform.outputs == symbolic_transform.outputs
    y, sym_y = transform(x), symbolic_transform(x)
    assert np.allclose(y, sym_y)
    x_from_inv = transform._inverse(y)
    x_sym_from_inv = symbolic_transform.inverse(sym_y)
    assert np.allclose(x_from_inv, x_sym_from_inv)
    jacobian = transform.jacobian(x, y)
    sym_jacobian = symbolic_transform.jacobian(x, sym_y)
    assert np.allclose(jacobian, sym_jacobian)
    inv_jacobian = transform.inverse_jacobian(x, y)
    sym_inv_jacobian = symbolic_transform.inverse_jacobian(x, sym_y)
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
    new_params, new_dims, jacobian = construct_jacobian(transforms, params, dims)
    new_masses, combined_mass_jacobian = (
        ComponentMassesToChirpMassAndSymmetricMassRatio().transform_and_jacobian(
            params[:, :2]  # (m1, m2)
        )
    )
    dl, ddl_dz = redshift_to_luminosity_distance_transform.transform_and_jacobian(
        params[:, -1]
    )
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
