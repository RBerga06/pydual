"""Fixed-size dual basis relying on a covariance matrix."""

from dataclasses import dataclass
from typing import Self, cast, override
import numpy as np

from .._np import Mat, Shape, Tensor, Vec
from .abc import (
    Callback1Scalar,
    Callback1Vector,
    Callback2,
    Callback2Scalar,
    Callback2Vector,
    DualBasis,
    DualPart,
)


__all__ = ["FixedDualBasis", "FixedDualPart"]


@dataclass(slots=True, frozen=True)
class FixedDualBasis[N: int](DualBasis):
    cov_matrix: Mat[N]

    def __init__(self, cov_matrix: Mat[N] | N, /) -> None:
        """Create a fixed-size dual basis. If the covariance matrix is an integer, it is understood as `np.eye(<integer>)`."""
        if isinstance(cov_matrix, int):
            cov_matrix = np.eye(cov_matrix)
        object.__setattr__(self, "cov_matrix", cov_matrix)

    @property
    def n(self, /) -> N:
        return self.cov_matrix.shape[0]  # pyright: ignore[reportReturnType]

    @override
    def zero[S: Shape](self, shape: S, /) -> "FixedDualPart[S, N]":
        data = cast(Tensor[tuple[N, *S]], np.zeros((*shape, self.n), dtype=np.float64))
        return FixedDualPart(data, self)

    @override
    def zero_like[S: Shape](self, dual: DualPart[Self, S], /) -> "FixedDualPart[S, N]":
        assert isinstance(dual, FixedDualPart)
        shape = cast(FixedDualPart[S, N], dual).data.shape
        data = cast(Tensor[tuple[N, *S]], np.zeros(shape, dtype=np.float64))
        return FixedDualPart(data, self)

    def eye(
        self, delta: Vec[N] | None = None, /
    ) -> "FixedDualPart[tuple[N], N]":  # TODO: Support ndarrays?
        data: Mat[N]
        if delta is None:
            data = np.eye(self.n, dtype=np.float64)
        else:
            data = np.diag(delta)
        return FixedDualPart[tuple[N], N](data, self)


@dataclass(slots=True, frozen=True)
class FixedDualPart[S: Shape, N: int](DualPart[FixedDualBasis[N], S]):
    data: Tensor[tuple[N, *S]]
    basis: FixedDualBasis[N]

    @override
    def get_basis(self, /) -> FixedDualBasis[N]:
        return self.basis

    @override
    def cov(self, rhs: Self, /) -> Tensor[S]:
        return np.vecdot(np.matvec(self.basis.cov_matrix, self.data.T), rhs.data.T).T  # pyright: ignore[reportAny]

    @override
    def map_[Z: Shape](
        self, /, f_scalar: Callback1Scalar[S, Z], f_vector: Callback1Vector[S, Z]
    ) -> "FixedDualPart[Z, N]":
        return FixedDualPart(f_vector(self.data), self.basis)

    @override
    def map2(self, rhs: Self, f: Callback2[S], /) -> Self:
        raise NotImplementedError

    @override
    def map2_[S2: Shape, Z: Shape](
        self,
        lhs_shape: S,
        rhs: "DualPart[FixedDualBasis[N], S2]",
        rhs_shape: S2,
        /,
        f_scalar: Callback2Scalar[S, S2, Z],
        f_vector: Callback2Vector[S, S2, Z],
    ) -> "FixedDualPart[Z, N]":
        raise NotImplementedError

    @override
    def clone(self, /) -> Self:
        return type(self)(self.data, FixedDualBasis(self.basis.cov_matrix.copy()))
