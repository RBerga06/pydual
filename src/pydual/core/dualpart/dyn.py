"""Dynamically-sized dual basis where variables are indipendent by default."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Self, final, override
import numpy as np
from .._np import Shape, Tensor
from .abc import Callback1Scalar, Callback1Vector, Callback2, Callback2Scalar, Callback2Vector, DualBasis, DualPart, Callback1

__all__ = ["DynDualBasis", "DynDualPart"]


@final
class DynDualBasis(DualBasis):
    """
    Dynamically-sized dual basis where variables are indipendent by default.

    NOTE: All instances of this class are fully equivalent!
    """
    __slots__ = ()

    @override
    def zero[S: Shape](self, shape: S, /) -> "DynDualPart[S]":
        return DynDualPart({})

    @override
    def zero_like[S: Shape](self, x: "DualPart[Self, S]", /) -> "DynDualPart[S]":
        return DynDualPart({})

DYN_DUAL_BASIS: Final = DynDualBasis()
"""A `DynDualBasis` instance."""

_counter: int = 0

def _new_index() -> int:
    global _counter
    idx = _counter
    _counter += 1
    return idx

@final
@dataclass(slots=True, frozen=True)
class DynDualPart[S: Shape](DualPart[DynDualBasis, S]):
    """Dual parts (implemented as a Python dictionary)."""
    data: dict[int, DualPart[DualBasis, S]]

    @override
    def get_basis(self, /) -> DynDualBasis:
        return DYN_DUAL_BASIS

    @override
    def cov(self, rhs: Self, /) -> Tensor[S]:
        # missing values don't contribute to the covariance
        return np.sum([self.data[k].cov(rhs.data[k]) for k in self.data.keys() & rhs.data.keys()], axis=0)  # pyright: ignore[reportAny]

    @override
    def map_[Z: Shape](self, /, f_scalar: Callback1Scalar[S, Z], f_vector: Callback1Vector[S, Z]) -> "DynDualPart[Z]":
        return DynDualPart({k: v.map_(f_scalar, f_vector) for k, v in self.data.items()})

    if TYPE_CHECKING:
        @override
        def map[Z: Shape](self, f: Callback1[S, Z], /) -> "DynDualPart[Z]": ...


    @override
    def map2_[S2: Shape, Z: Shape](
        self, lhs_shape: S,
        rhs: "DualPart[DynDualBasis, S2]", rhs_shape: S2,
        /,
        f_scalar: Callback2Scalar[S, S2, Z], f_vector: Callback2Vector[S, S2, Z]
    ) -> "DynDualPart[Z]":
        assert isinstance(rhs, DynDualPart)
        # TODO: Find a way of changing output shape
        def callback[N: int](x: DualPart[DualBasis, S] | None, y: DualPart[DualBasis, S2] | None, /) -> DualPart[DualBasis, Z]:
            if x is None:
                assert y is not None
                x = y.get_basis().zero(lhs_shape)
            elif y is None:
                y = x.get_basis().zero(rhs_shape)
            return x.map2_(lhs_shape, y, rhs_shape, f_scalar, f_vector)
        return DynDualPart({k: callback(self.data.get(k), rhs.data.get(k)) for k in self.data.keys() | rhs.data.keys()})

    @override
    def map2(self, rhs: Self, f: Callback2[S], /) -> Self:
        # TODO: Find a way of changing output shape
        def callback[N: int](x: DualPart[DualBasis, S] | None, y: DualPart[DualBasis, S] | None, /) -> DualPart[DualBasis, S]:
            if x is None:
                assert y is not None
                x = y.get_basis().zero_like(y)
            elif y is None:
                y = x.get_basis().zero_like(x)
            return x.map2(y, f)
        return type(self)({k: callback(self.data.get(k), rhs.data.get(k)) for k in self.data.keys() | rhs.data.keys()})

    def map2alt(self, rhs: Self, f: Callback2[S], fl: Callback1[S], fr: Callback1[S], /) -> Self:
        """Apply the given functions to all dual parts in `self` and `rhs`."""
        # TODO: Find a way of changing output shape
        return type(self)(
            {k: self.data[k].map2(rhs.data[k], f) for k in self.data.keys() & rhs.data.keys()}
            | {k: self.data[k].map(fl) for k in self.data.keys() - rhs.data.keys()}
            | {k: rhs.data[k].map(fr) for k in rhs.data.keys() - self.data.keys()},
        )

    def zero(self, /) -> Self:
        """Return a zero with the same shape as `self` and the same basis."""
        return type(self)({})

    @override
    def clone(self, /) -> Self:
        return type(self)({_new_index(): v.clone() for _, v in self.data.items()})

    @staticmethod
    def from_dual_parts[Z: Shape](*duals: DualPart[Any, Z]) -> "DynDualPart[Z]":  # pyright: ignore[reportExplicitAny]
        return DynDualPart({_new_index(): d for d in duals})
