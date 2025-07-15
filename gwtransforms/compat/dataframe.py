from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from gwtransforms.transforms.base import ParameterTransform, ParameterTransformT
from gwtransforms.utils import unpack_parameter_dim


class DataFrameParameterTransform(ParameterTransformT[pd.Series | pd.DataFrame]):
    def __init__(self, transform: ParameterTransform, in_place: bool = True):
        self._transform = transform
        self.in_place = in_place
        self.inputs = self._transform.inputs
        self.outputs = self._transform.outputs

    def _wrap_array(
        self, array: ArrayLike, index: pd.Index, cols: Sequence[str]
    ) -> pd.Series | pd.DataFrame:
        ncols = len(cols)
        array = np.asanyarray(array)
        if ncols == 1:
            return pd.Series(array, index=index)

        if array.shape[-1] != ncols:
            raise ValueError(
                f"Expected last array dim to be {cols}, got {array.shape[-1]}"
            )
        values = unpack_parameter_dim(array)
        data = dict(zip(cols, values))
        return pd.DataFrame(data=data, index=index)

    def _get_dims(self, x: pd.Series | pd.DataFrame, dims: Sequence[str]) -> ArrayLike:
        x_at_dims = x[list(dims)] if isinstance(x, pd.DataFrame) else x
        return x_at_dims.to_numpy()

    def _check_inputs(self, df: pd.DataFrame):
        if not all([i in df.columns for i in self.inputs]):
            raise ValueError("DataFrame must contain all inputs as columns")

    def __call__(self, x: pd.Series | pd.DataFrame):
        _x = self._get_dims(x, self.inputs)
        _y = self._transform(_x)
        if self.in_place:
            x[list(self.outputs)] = _y
            return x
        else:
            return self._wrap_array(_y, x.index, self.outputs)

    def _inverse(self, y: pd.Series | pd.DataFrame):
        _y = self._get_dims(y, self.outputs)
        _x = self._transform._inverse(_y)
        if self.in_place:
            y[list(self.inputs)] = _x
            return y
        else:
            return self._wrap_array(_x, y.index, self.inputs)

    def jacobian(
        self, x: pd.Series | pd.DataFrame, y: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        _x = self._get_dims(x, self.inputs)
        _y = self._get_dims(y, self.outputs)
        jacobian = self._transform.jacobian(_x, _y)
        n = len(self.inputs) * len(self.outputs)
        jacobian = np.asanyarray(jacobian).reshape(-1, n)
        cols = pd.MultiIndex.from_product(
            [self.outputs, self.inputs], names=["outputs", "inputs"]
        )
        return pd.DataFrame(data=jacobian, index=x.index, columns=cols)

    def _inverse_jacobian(
        self, x: pd.Series | pd.DataFrame, y: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        _x = self._get_dims(x, self.inputs)
        _y = self._get_dims(y, self.outputs)
        inverse_jacobian = self._transform._inverse_jacobian(_x, _y)
        n = len(self.inputs) * len(self.outputs)
        inverse_jacobian = np.asanyarray(inverse_jacobian).reshape(-1, n)
        cols = pd.MultiIndex.from_product(
            [self.inputs, self.outputs], names=["inputs", "outputs"]
        )
        return pd.DataFrame(data=inverse_jacobian, index=y.index, columns=cols)

    def log_abs_det_jacobian(
        self, x: pd.Series | pd.DataFrame, y: pd.Series | pd.DataFrame
    ) -> ArrayLike:
        return self._transform.log_abs_det_jacobian(x, y)


def with_pandas(
    transform: ParameterTransform, in_place: bool = False
) -> ParameterTransformT[pd.Series | pd.DataFrame]:
    """
    Wrap `transform` into an object that can be used with
    `pandas.Series` and `pandas.DataFrame`.

    Parameters
    ----------
    transform: ParameterTransform
        A parameter transform acting on `numpy` arrays
    in_place: bool
        Whether to modifiy the `Series` or `DataFrame` in place
    when possible. Defaults to `False`

    Returns
    -------
    ParameterTransformT[pd.Series | pd.DataFrame]
        The wrapped transform

    Examples
    --------

    """
    return DataFrameParameterTransform(transform, in_place=in_place)
