from typing import Hashable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def stack_dict_keys_into_array(
    d: Mapping[Hashable, ArrayLike], *keys: Hashable, axis: int = -1
) -> NDArray:
    return np.stack([d[key] for key in keys], axis=axis)


def unstack_array_to_dict(
    d: dict[Hashable, ArrayLike], arr: NDArray, *keys: Hashable, axis: int = -1
) -> dict[Hashable, ArrayLike]:
    _arr = np.moveaxis(arr, axis, 0)
    return {**d, **{key: _arr[i, ...] for i, key in enumerate(keys)}}


def combine_arrays(arrays, shape):
    arr_shape = arrays[0].shape
    return np.stack(arrays, axis=-1).reshape(*arr_shape, *shape)


def unpack_parameter_dim(array: ArrayLike, axis: int = -1) -> NDArray:
    """
    Swap axes of an array so that the `axis` dimension comes first.

    Parameters
    ----------
    array : ArrayLike
        Input array
    axis : int, optional
        Axis to unpack, by default -1

    Returns
    -------
    NDArray
        Array with swapped axes

    Examples
    --------

    Trying to unpack an array along the wrong dimension:

    >>> # x, y lie along the last axis
    >>> a = np.ones((5, 3, 2))
    >>> x, y = a
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: too many values to unpack (expected 2)

    >>> # x, y lie along the last axis
    >>> a = np.ones((5, 3, 2))
    >>> x, y = unpack_parameter_dim(a, axis=-1)
    >>> x.shape, y.shape
    ((5, 3), (5, 3))
    """
    array = np.asanyarray(array)
    return np.moveaxis(array, axis, 0)


def combine_into_transform(*arrays):
    return np.stack(arrays, axis=-1)


def combine_into_jacobian(*arrays):
    ndim = int(np.sqrt(len(arrays)))
    return combine_arrays(arrays, shape=(ndim, ndim))
