"""Protocols."""
from dataclasses import dataclass
from typing import Any, Protocol, Self, override
import numpy as np

type Tensor[S: tuple[int, ...], T: np.generic = np.float64] = np.ndarray[S, np.dtype[T]]
type Vec[N: int, T: np.generic = np.float64] = Tensor[tuple[N], T]
type Mat[N: int, M: int = N, T: np.generic = np.float64] = Tensor[tuple[N, M], T]


class DualParts[B: DualBasis[Any], S: tuple[int, ...], N: int = int](Protocol):
    """The dual parts of a dual number."""


class DualBasis[N: int = int](Protocol):
    """Any basis for dual parts."""

    def cov[S: tuple[int, ...]](self, lhs: DualParts[Self, S, N], rhs: DualParts[Self, S, N], /) -> Tensor[S]:
        """Evaluate the covariance between the given dual parts."""
        raise NotImplementedError

    def std[S: tuple[int, ...]](self, dual: DualParts[Self, S, N], /) -> Tensor[S]:
        """Evaluate the standard deviation of the given dual parts."""
        return self.cov(dual, dual) ** 0.5  # pyright: ignore[reportReturnType]


@dataclass(slots=True, frozen=True)
class FixedDualBasis[N: int](DualBasis[N]):
    """A basis for dual parts (implemented as a fixed-size NumPy matrix)."""
    cov_matrix: Mat[N]

    @override
    def cov[S: tuple[int, ...]](self, lhs: DualParts[Self, S, N], rhs: DualParts[Self, S, N], /) -> Tensor[S]:
        """Evaluate the covariance between the given dual parts."""
        assert isinstance(lhs, FixedDualPart) and isinstance(rhs, FixedDualPart)
        return np.vecdot(np.matvec(self.cov_matrix, lhs.data), rhs.data)  # pyright: ignore[reportAny]

    def __init__(self, cov_matrix: Mat[N] | N, /) -> None:
        """Create a fixed-size dual basis. If the covariance matrix is an integer, it is understood as `np.eye(<integer>)`."""
        if isinstance(cov_matrix, int):
            cov_matrix = np.eye(cov_matrix)
        object.__setattr__(self, "cov_matrix", cov_matrix)

@dataclass(slots=True, frozen=True)
class FixedDualPart[S: tuple[int, ...], N: int](DualParts[FixedDualBasis[N], S, N]):
    data: Tensor[tuple[*S, N]]


class DynDualBasis(DualBasis):
    """A dynamically-sized dual basis (all variables are independent here)."""

    @override
    def cov[S: tuple[int, ...]](self, lhs: DualParts[Self, S], rhs: DualParts[Self, S], /) -> Tensor[S]:
        """Evaluate the covariance between the given dual parts."""
        assert isinstance(lhs, DynDualPart) and isinstance(rhs, DynDualPart)
        total: Tensor[S] = 0.0  # pyright: ignore[reportAssignmentType]
        for k in lhs.data | rhs.data:
            x = lhs.data.get(k, 0.0)
            y = rhs.data.get(k, 0.0)
            total += x * y
        return total

@dataclass(slots=True, frozen=True)
class DynDualPart[S: tuple[int, ...]](DualParts[DynDualBasis, S]):
    data: dict[int, Tensor[S]]
