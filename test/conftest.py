import numpy as np
import pytest

from gwtransforms.transforms.symbolic import get_all_mass_symbolic_transforms


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


@pytest.fixture
def symbolic_transforms():
    return get_all_mass_symbolic_transforms()
