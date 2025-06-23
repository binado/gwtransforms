import weakref
from typing import (
    Generic,
    Hashable,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gwtransforms.utils import stack_dict_keys_into_array, unstack_array_to_dict

T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class ParameterTransformT(Protocol[T]):
    _inv: Union[
        "ParameterTransformT", weakref.ReferenceType["ParameterTransformT"], None
    ] = None
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def ninputs(self) -> int:
        return len(self.inputs)

    @property
    def noutputs(self) -> int:
        return len(self.outputs)

    @property
    def inv(self) -> "ParameterTransformT":
        inv = None
        if self._inv is not None and isinstance(self._inv, weakref.ReferenceType):
            inv = self._inv()
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = weakref.ref(inv)
        return inv

    def __call__(self, x: T) -> T:
        raise NotImplementedError

    def _inverse(self, y: T) -> T:
        raise NotImplementedError

    def jacobian(self, x: T, y: T) -> T:
        raise NotImplementedError

    def _inverse_jacobian(self, x: T, y: T) -> T:
        raise NotImplementedError

    def log_abs_det_jacobian(
        self,
        x: T,
        y: T,
    ) -> ArrayLike:
        raise NotImplementedError
        # jacobian = self.jacobian(x, y)
        # det = np.linalg.det(jacobian)
        # return np.log(np.absolute(det))
        #

    # def apply_to_dict(self, d: dict[Hashable, T]):
    #     if self.inputs is None:
    #         raise ValueError("inputs attribute should be set")
    #     x = stack_dict_keys_into_array(d, *self.inputs)
    #     y = self(x)
    #     return unstack_array_to_dict(d, y, *self.inputs)


class _InverseTransform(ParameterTransformT[T]):
    def __init__(self, transform: ParameterTransformT[T]) -> None:
        super().__init__()
        self._inv = transform
        self.inputs = self.inv.outputs
        self.outputs = self.inv.inputs

    @property
    def inv(self) -> ParameterTransformT[T]:
        if not isinstance(self._inv, ParameterTransformT):
            raise ValueError("Unable to construct inverse object")
        return self._inv

    def __call__(self, x: T) -> T:
        return self.inv._inverse(x)

    def _inverse(self, y: T) -> T:
        return self.inv(y)

    def jacobian(self, x: T, y: T) -> T:
        return self.inv._inverse_jacobian(y, x)

    def _inverse_jacobian(self, x: T, y: T) -> T:
        return self.inv.jacobian(y, x)

    def log_abs_det_jacobian(self, x: T, y: T) -> ArrayLike:
        inv = self.inv.log_abs_det_jacobian(y, x)
        return -np.asanyarray(inv)


class ParameterTransform(ParameterTransformT[ArrayLike]):
    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        jacobian = np.asanyarray(self.jacobian(x, y))
        det = np.linalg.det(jacobian)
        return np.log(np.absolute(det))


class ComposeTransform(ParameterTransform):
    _inv = None

    def __init__(self, *transforms: ParameterTransform):
        self.transforms = transforms
        self.inputs = self.transforms[0].inputs
        self.outputs = self.transforms[-1].outputs

    def __call__(self, x: ArrayLike) -> ArrayLike:
        for transform in self.transforms:
            x = transform(x)
        return x

    def _inverse(self, y: ArrayLike) -> ArrayLike:
        for transform in reversed(self.transforms):
            y = transform._inverse(y)
        return y

    def jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        x = np.asanyarray(x)
        jac = np.eye(x.shape[0])
        for transform in self.transforms:
            y_aux = transform(x)
            current_jac = np.asanyarray(transform.jacobian(x, y_aux))
            jac = current_jac @ jac
            x = y_aux
        return jac

    def _inverse_jacobian(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        x = np.asanyarray(x)
        y = np.asanyarray(y)
        jac = np.eye(x.shape[0], x.shape[0])
        for transform in self.transforms:
            y_aux = transform(x)
            current_jac = np.asanyarray(transform._inverse_jacobian(x, y_aux))
            jac = jac @ current_jac
            x = y_aux
        return jac


def construct_jacobian(
    transforms: Sequence[ParameterTransform],
    params: NDArray,
    dims: dict[str, int],
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
        y = transform(x)
        jac = transform.jacobian(x, y)
        jac = np.asanyarray(jac)
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
