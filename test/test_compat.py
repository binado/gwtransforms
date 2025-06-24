import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy.cosmology import Planck18

from gwtransforms.compat.dataarray import with_xarray
from gwtransforms.compat.dataframe import with_pandas
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
    "transform",
    [
        ComponentMassesToPrimaryMassAndMassRatio(),
        ComponentMassesToTotalMassAndMassRatio(),
        ComponentMassesToChirpMassAndSymmetricMassRatio(),
    ],
)
def test_with_pandas_dataframe(transform, mass_arrays):
    cols = transform.inputs
    x = stack_dict_keys_into_array(mass_arrays, *cols)
    x_df = pd.DataFrame(data=x, columns=cols)
    df_transform = with_pandas(transform, in_place=False)
    y = transform(x)
    y_df = df_transform(x_df)
    assert isinstance(y_df, pd.DataFrame)
    assert list(y_df.columns) == list(transform.outputs)
    assert np.allclose(y, y_df.to_numpy())

    x_inv = transform._inverse(y)
    x_df_inv = df_transform._inverse(y_df)
    assert isinstance(x_df_inv, pd.DataFrame)
    assert list(x_df_inv.columns) == list(transform.inputs)
    assert np.allclose(x_inv, x_df_inv.to_numpy())

    jacobian = transform.jacobian(x, y)
    jacobian_df = df_transform.jacobian(x_df, y_df)
    assert isinstance(jacobian_df, pd.DataFrame)
    for o, output_label in enumerate(transform.outputs):
        for i, input_label in enumerate(transform.inputs):
            assert np.allclose(
                jacobian[:, o, i], jacobian_df[output_label, input_label]
            )

    inverse_jacobian = transform._inverse_jacobian(x, y)
    inverse_jacobian_df = df_transform._inverse_jacobian(x_df, y_df)
    assert isinstance(inverse_jacobian_df, pd.DataFrame)
    for i, input_label in enumerate(transform.inputs):
        for o, output_label in enumerate(transform.outputs):
            assert np.allclose(
                inverse_jacobian[:, i, o],
                inverse_jacobian_df[input_label, output_label],
            )


def test_with_pandas_series(redshift_array):
    transform = RedshiftToLuminosityDistance(cosmology=Planck18)
    cols = transform.inputs
    x = redshift_array
    x_df = pd.Series(data=x)
    df_transform = with_pandas(transform, in_place=False)
    y = transform(x)
    y_df = df_transform(x_df)
    assert isinstance(y_df, pd.Series)
    assert np.allclose(y, y_df.to_numpy())

    x_inv = transform._inverse(y)
    x_df_inv = df_transform._inverse(y_df)
    assert isinstance(x_df_inv, pd.Series)
    assert np.allclose(x_inv, x_df_inv.to_numpy())

    jacobian = transform.jacobian(x, y)
    jacobian_df = df_transform.jacobian(x_df, y_df)
    assert isinstance(jacobian_df, pd.DataFrame)
    for o, output_label in enumerate(transform.outputs):
        for i, input_label in enumerate(transform.inputs):
            assert np.allclose(
                jacobian[:, o, i], jacobian_df[output_label, input_label]
            )

    inverse_jacobian = transform._inverse_jacobian(x, y)
    inverse_jacobian_df = df_transform._inverse_jacobian(x_df, y_df)
    assert isinstance(inverse_jacobian_df, pd.DataFrame)
    for i, input_label in enumerate(transform.inputs):
        for o, output_label in enumerate(transform.outputs):
            assert np.allclose(
                inverse_jacobian[:, i, o],
                inverse_jacobian_df[input_label, output_label],
            )


@pytest.mark.parametrize(
    "transform",
    [
        ComponentMassesToPrimaryMassAndMassRatio(),
        ComponentMassesToTotalMassAndMassRatio(),
        ComponentMassesToChirpMassAndSymmetricMassRatio(),
    ],
)
def test_with_xarray(transform, mass_arrays):
    cols = transform.inputs
    x = stack_dict_keys_into_array(mass_arrays, *cols)
    batch_size = x.shape[0]
    batch_dim = "batch"
    forward_dim = "inputs"
    inverse_dim = "outputs"
    forward_jacobian_dim = forward_dim
    inverse_jacobian_dim = inverse_dim
    x_da = xr.DataArray(
        data=x,
        dims=[batch_dim, forward_dim],
        coords=[np.arange(batch_size), list(transform.inputs)],
    )
    da_transform = with_xarray(
        transform, forward_dim, inverse_dim, forward_jacobian_dim, inverse_jacobian_dim
    )
    y = transform(x)
    y_da = da_transform(x_da)
    assert isinstance(y_da, xr.DataArray)
    assert np.allclose(y, y_da)
    assert set(y_da.dims) == {batch_dim, inverse_dim}

    jacobian = transform.jacobian(x, y)
    jacobian_da = da_transform.jacobian(x_da, y_da)
    assert isinstance(jacobian_da, xr.DataArray)
    assert np.allclose(jacobian, jacobian_da)
    assert set(jacobian_da.dims) == {
        batch_dim,
        forward_jacobian_dim,
        inverse_jacobian_dim,
    }

    inverse_jacobian = transform._inverse_jacobian(x, y)
    inverse_jacobian_da = da_transform._inverse_jacobian(x_da, y_da)
    assert isinstance(inverse_jacobian_da, xr.DataArray)
    assert np.allclose(inverse_jacobian, inverse_jacobian_da)
    assert set(inverse_jacobian_da.dims) == {
        batch_dim,
        forward_jacobian_dim,
        inverse_jacobian_dim,
    }
