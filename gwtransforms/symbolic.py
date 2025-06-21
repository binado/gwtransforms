from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy.utilities.iterables import flatten

from gwtransforms.utils import combine_into_jacobian, unpack_parameter_dim


@dataclass
class SymbolicTransform:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    input_symbols: tuple[sp.Symbol, ...]
    output_symbols: tuple[sp.Symbol, ...]
    forward_exprs: sp.Matrix
    inverse_exprs: sp.Matrix
    jacobian_exprs: sp.Matrix
    inverse_jacobian_exprs: sp.Matrix

    def __post_init__(self):
        self._forward_numpy = self.lambdify(self.input_symbols, self.forward_exprs)
        self._inverse_numpy = self.lambdify(self.output_symbols, self.inverse_exprs)
        self._jacobian_numpy = self.lambdify(self.input_symbols, self.jacobian_exprs)
        self._inv_jacobian_numpy = self.lambdify(
            self.output_symbols, self.inverse_jacobian_exprs
        )

    @property
    def ninputs(self):
        return len(self.inputs)

    @property
    def noutputs(self):
        return len(self.outputs)

    def lambdify(self, args, expr):
        if isinstance(expr, sp.Matrix):
            _expr = expr.tolist()
        elif isinstance(expr, list | sp.Expr):
            _expr = expr
        else:
            raise ValueError("Wrong format for expr")
        return sp.lambdify(args, _expr, modules="numpy")

    def _vectorize(self, x: NDArray, lambdified_fn, dims: tuple[int, ...]) -> NDArray:
        res = lambdified_fn(*unpack_parameter_dim(x))
        flattened = flatten(res, cls=list)
        arrs = np.broadcast_arrays(*flattened)
        return np.stack(arrs, axis=-1).reshape(-1, *dims)

    def det(self) -> sp.Expr:
        return self.jacobian_exprs.det()

    def _log_abs_det_jacobian(self):
        return sp.log(abs(self.det()))

    def __call__(self, x: NDArray) -> NDArray:
        dims = (self.noutputs,)
        return self._vectorize(x, self._forward_numpy, dims=dims)

    def inverse(self, y: NDArray) -> NDArray:
        dims = (self.ninputs,)
        return self._vectorize(y, self._inverse_numpy, dims=dims)

    def jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        dims = (self.noutputs, self.ninputs)
        return self._vectorize(x, self._jacobian_numpy, dims=dims)

    def inverse_jacobian(self, x: NDArray, y: NDArray) -> NDArray:
        dims = (self.ninputs, self.noutputs)
        return self._vectorize(y, self._inv_jacobian_numpy, dims=dims)


def get_expression_and_jacobian(
    inputs: Sequence[sp.Symbol],
    outputs: Sequence[sp.Symbol],
    subs: Sequence[tuple[sp.Symbol, sp.Expr]],
):
    expressions = sp.Matrix(outputs).subs(subs[::-1])
    jacobian = expressions.jacobian(inputs)
    return expressions, jacobian


def get_symbolic_transform(
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    input_symbols: tuple[sp.Symbol, ...],
    output_symbols: tuple[sp.Symbol, ...],
    subs: Sequence[tuple[sp.Symbol, sp.Expr]],
    inverse_subs: Sequence[tuple[sp.Symbol, sp.Expr]],
) -> SymbolicTransform:
    forward_exprs, jacobian_exprs = get_expression_and_jacobian(
        input_symbols, output_symbols, subs
    )
    inverse_exprs, inverse_jacobian_exprs = get_expression_and_jacobian(
        output_symbols, input_symbols, inverse_subs
    )
    return SymbolicTransform(
        inputs,
        outputs,
        input_symbols,
        output_symbols,
        forward_exprs,
        inverse_exprs,
        jacobian_exprs,
        inverse_jacobian_exprs,
    )


def get_all_mass_symbolic_transforms():
    m_1, m_2, m_t, m_c, q, nu = sp.symbols("m_1 m_2 m_t m_c q nu")

    transform_dict = {}
    # (m_1, m_2) -> (m_1, q)
    inputs = ("mass_1", "mass_2")
    outputs = ("mass_1", "mass_ratio")
    input_symbols = (m_1, m_2)
    output_symbols = (m_1, q)
    subs = [(q, m_2 / m_1)]
    inverse_subs = [(m_2, m_1 * q)]
    transform_dict["component_masses_to_primary_mass_and_mass_ratio"] = (
        get_symbolic_transform(
            inputs, outputs, input_symbols, output_symbols, subs, inverse_subs
        )
    )

    # (m_1, m_2) -> (m_T, q)
    inputs = ("mass_1", "mass_2")
    outputs = ("total_mass", "mass_ratio")
    input_symbols = (m_1, m_2)
    output_symbols = (m_t, q)
    subs = [(m_t, m_1 + m_2), (q, m_2 / m_1)]
    inverse_subs = [(m_1, m_t / (1 + q)), (m_2, m_1 * q)]
    transform_dict["component_masses_to_total_mass_and_mass_ratio"] = (
        get_symbolic_transform(
            inputs,
            outputs,
            input_symbols,
            output_symbols,
            subs,
            inverse_subs,
        )
    )

    # (m_1, m_2) -> (m_C, nu)
    inputs = ("mass_1", "mass_2")
    outputs = ("chirp_mass", "symmetric_mass_ratio")
    input_symbols = (m_1, m_2)
    output_symbols = (m_c, nu)

    subs = [
        (m_t, m_1 + m_2),
        (q, m_2 / m_1),
        (nu, q / (1 + q) ** 2),
        (m_c, m_t * nu ** sp.Rational(3, 5)),
    ]
    inverse_subs = [
        (m_t, m_c * nu ** sp.Rational(-3, 5)),
        (q, -1 + sp.Rational(1, 2) / nu * (1 + sp.sqrt(1 - 4 * nu))),
        (m_1, m_t / (1 + q)),
        (m_2, m_1 * q),
    ]
    transform_dict["component_masses_to_chirp_mass_and_symmetric_mass_ratio"] = (
        get_symbolic_transform(
            inputs,
            outputs,
            input_symbols,
            output_symbols,
            subs,
            inverse_subs,
        )
    )

    # (m_T, q) -> (m_C, nu)
    inputs = ("total_mass", "mass_ratio")
    outputs = ("chirp_mass", "symmetric_mass_ratio")
    input_symbols = (m_t, q)
    output_symbols = (m_c, nu)
    subs = [
        (nu, q / (1 + q) ** 2),
        (m_c, m_t * nu ** sp.Rational(3, 5)),
    ]
    inverse_subs = [
        (m_t, m_c * nu ** sp.Rational(-3, 5)),
        (q, -1 + sp.Rational(1, 2) / nu * (1 + sp.sqrt(1 - 4 * nu))),
    ]
    transform_dict[
        "total_mass_and_mass_ratio_to_chirp_mass_and_symmetric_mass_ratio"
    ] = get_symbolic_transform(
        inputs,
        outputs,
        input_symbols,
        output_symbols,
        subs,
        inverse_subs,
    )

    return transform_dict
