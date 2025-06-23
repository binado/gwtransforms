from collections import OrderedDict
from typing import Hashable, Mapping, Sequence

import xarray as xr
from numpy.typing import ArrayLike

from gwtransforms.transforms.base import ParameterTransform, ParameterTransformT


class DataArrayParameterTransform(ParameterTransformT[xr.DataArray]):
    def __init__(
        self,
        transform: ParameterTransform,
        forward_dim: str,
        inverse_dim: str,
        forward_jacobian_dim: str,
        inverse_jacobian_dim: str,
    ) -> None:
        self._transform = transform
        self.inputs = self._transform.inputs
        self.outputs = self._transform.outputs
        self.forward_dim = forward_dim
        self.inverse_dim = inverse_dim
        self.forward_jacobian_dim = forward_jacobian_dim
        self.inverse_jacobian_dim = inverse_jacobian_dim

    def _replace_dims(
        self,
        x: xr.DataArray,
        old_dims: Sequence[Hashable],
        new_dims: Sequence[Hashable],
    ) -> tuple[Hashable, ...]:
        n = len(old_dims)
        if x.dims[-n:] != old_dims:
            raise ValueError(
                f"expected last dims to be {old_dims}, got {x.dims[-n:]} instead"
            )
        return tuple(list(x.dims[:-n]) + list(new_dims))

    def _merge_coords(
        self,
        target: xr.DataArray,
        old_dims: Sequence[Hashable],
        new_coords: Mapping[str, Sequence[Hashable]],
    ) -> tuple[tuple[Hashable, ...], xr.Coordinates]:
        new_dims = list(new_coords.keys())
        merged_dims = self._replace_dims(target, old_dims, new_dims)
        merged_coords = {
            **{k: list(v) for k, v in target.coords.items() if k in merged_dims},
            **{k: list(v) for k, v in new_coords.items()},
        }
        return merged_dims, xr.Coordinates(coords=merged_coords)

    def _wrap_array(
        self,
        x: ArrayLike,
        target: xr.DataArray,
        old_dims: Sequence[Hashable],
        new_coords: Mapping[str, Sequence[Hashable]],
    ) -> xr.DataArray:
        dims, coords = self._merge_coords(target, old_dims, new_coords)
        return xr.DataArray(x, dims=dims, coords=coords)

    def _get_at_coords(
        self, x: xr.DataArray, dim: str, coords: Sequence[str]
    ) -> xr.DataArray:
        coords_as_list = list(coords)
        return x.loc[{dim: coords_as_list}]

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        _x = self._get_at_coords(x, self.forward_dim, self.inputs)
        _y = self._transform(_x)
        old_dims = (self.forward_dim,)
        new_coords = OrderedDict([(self.inverse_dim, self.outputs)])
        return self._wrap_array(_y, x, old_dims, new_coords)

    def _inverse(self, y: xr.DataArray) -> xr.DataArray:
        _y = self._get_at_coords(y, self.inverse_dim, self.outputs)
        _x = self._transform.inv(_y)
        old_dims = (self.inverse_dim,)
        new_coords = OrderedDict([(self.forward_dim, self.inputs)])
        return self._wrap_array(_x, y, old_dims, new_coords)

    def jacobian(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        _x = self._get_at_coords(x, self.forward_dim, self.inputs)
        _y = self._get_at_coords(y, self.inverse_dim, self.outputs)
        jacobian = self._transform.jacobian(_x, _y)
        old_dims = (self.forward_dim,)
        new_coords = OrderedDict(
            [(self.inverse_dim, self.outputs), (self.forward_dim, self.inputs)]
        )
        return self._wrap_array(jacobian, x, old_dims, new_coords)

    def _inverse_jacobian(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        _x = self._get_at_coords(x, self.forward_dim, self.inputs)
        _y = self._get_at_coords(y, self.inverse_dim, self.outputs)
        inverse_jacobian = self._transform._inverse_jacobian(_x, _y)
        old_dims = (self.inverse_dim,)
        new_coords = OrderedDict(
            [(self.forward_dim, self.inputs), (self.inverse_dim, self.outputs)]
        )
        return self._wrap_array(inverse_jacobian, y, old_dims, new_coords)

    def log_abs_det_jacobian(self, x: xr.DataArray, y: xr.DataArray) -> ArrayLike:
        _x = self._get_at_coords(x, self.forward_dim, self.inputs)
        _y = self._get_at_coords(y, self.inverse_dim, self.outputs)
        return self._transform.log_abs_det_jacobian(_x, _y)


def with_xarray(
    transform: ParameterTransform,
    forward_dim: str,
    inverse_dim: str,
    forward_jacobian_dim: str,
    inverse_jacobian_dim: str,
) -> ParameterTransformT[xr.DataArray]:
    return DataArrayParameterTransform(
        transform, forward_dim, inverse_dim, forward_jacobian_dim, inverse_jacobian_dim
    )
