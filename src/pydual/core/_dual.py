"""Protocols."""
from dataclasses import dataclass
from typing import Callable, Protocol, Self, cast, override
import numpy as np
from ._np import Shape, Tensor, Mat, Vec


class DualBasis(Protocol):
    """A basis for a space of dual parts."""

    def zero[S: Shape](self, shape: S, /) -> "DualParts[Self, S]":
        """Return the zero with the given shape and `self` as a basis."""
        raise NotImplementedError


class DualParts[B: DualBasis, S: Shape](Protocol):
    """The dual parts of a dual number."""
    basis: B
    """The basis where `self` lives."""

    def cov(self, rhs: Self, /) -> Tensor[S]:
        """Evaluate the covariance between `self` and `rhs`."""
        raise NotImplementedError

    def std(self, /) -> Tensor[S]:
        """Evaluate the standard deviation of `self`."""
        return self.cov(self) ** 0.5  # pyright: ignore[reportReturnType]

    def map(self, f: Callable[[Tensor[S]], Tensor[S]], /) -> Self:
        """Apply the given function to all dual parts in `self`."""
        raise NotImplementedError

    def map2(self, rhs: "Self | DualParts[B, tuple[()]]", f: Callable[[Tensor[S], Tensor[S] | Tensor[tuple[()]]], Tensor[S]], /) -> Self:
        """Apply the given function to all dual parts in `self` and `rhs`."""
        raise NotImplementedError

    def map2alt(self,
        rhs: "Self | DualParts[B, tuple[()]]",
        f: Callable[[Tensor[S], Tensor[S]], Tensor[S]],
        fl: Callable[[Tensor[S]], Tensor[S]],
        fr: Callable[[Tensor[S]], Tensor[S]],
        /
    ) -> Self:
        """Apply the given functions to all dual parts in `self` and `rhs`."""
        raise NotImplementedError  # TODO

    def zero(self, /) -> Self:
        """Return a zero with the same shape as `self` and the same basis."""
        raise NotImplementedError

    def clone(self, /) -> Self:
        """Create a new copy that is (stochastically) indipendent from `self`."""
        raise NotImplementedError



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
        data = cast(Tensor[tuple[*S, N]], np.zeros((*shape, self.n), dtype=np.float64))
        return FixedDualPart[S, N](data, self)

    def eye(self, delta: Vec[N] | None = None, /) -> "FixedDualPart[tuple[N], N]":
        data: Mat[N]
        if delta is None:
            data = np.eye(self.n, dtype=np.float64)
        else:
            data = np.diag(delta)
        return FixedDualPart[tuple[N], N](data, self)

@dataclass(slots=True, frozen=True)
class FixedDualPart[S: Shape, N: int](DualParts[FixedDualBasis[N], S]):
    """Dual parts (implemented as a fixed-size NumPy covariance matrix)."""
    data: Tensor[tuple[*S, N]]
    basis: FixedDualBasis[N]

    @override
    def cov(self, rhs: Self, /) -> Tensor[S]:
        return np.vecdot(np.matvec(self.basis.cov_matrix, self.data), rhs.data)  # pyright: ignore[reportAny]

    @override
    def map(self, f: Callable[[Tensor[S]], Tensor[S]], /) -> Self:
        raise NotImplementedError  # TODO

    @override
    def map2(self, rhs: "Self | DualParts[FixedDualBasis[N], tuple[()]]", f: Callable[[Tensor[S], Tensor[S] | Tensor[tuple[()]]], Tensor[S]], /) -> Self:
        """Apply the given function to all dual parts in `self` and `rhs` (treats missing values as 0.0)."""
        assert isinstance(rhs, FixedDualPart)
        raise NotImplementedError  # TODO

    @override
    def map2alt(self,
        rhs: "Self | DualParts[FixedDualBasis[N], tuple[()]]",
        f: Callable[[Tensor[S], Tensor[S]], Tensor[S]],
        fl: Callable[[Tensor[S]], Tensor[S]],
        fr: Callable[[Tensor[S]], Tensor[S]],
        /
    ) -> Self:
        """Apply the given functions to all dual parts in `self` and `rhs`."""
        assert isinstance(rhs, FixedDualPart)
        raise NotImplementedError  # TODO

    @override
    def zero(self, /) -> Self:
        """Return a zero with the same shape as `self` and the same basis."""
        raise NotImplementedError  # TODO

    @override
    def clone(self, /) -> Self:
        """Create a new copy that is (stochastically) indipendent from `self`."""
        raise TypeError("You cannot clone a `FixedDualPart`.")


class DynDualBasis(DualBasis):
    """A dynamically-sized dual basis (all variables are independent here)."""

    @override
    def zero[S: Shape](self, shape: S, /) -> "DynDualPart[S]":
        return DynDualPart({}, self)

@dataclass(slots=True, frozen=True)
class DynDualPart[S: Shape](DualParts[DynDualBasis, S]):
    """Dual parts (implemented as a Python dictionary)."""
    data: dict[int, Tensor[S]]
    basis: DynDualBasis

    @override
    def cov(self, rhs: Self, /) -> Tensor[S]:
        total: Tensor[S] = 0.0  # pyright: ignore[reportAssignmentType]
        for k in self.data | rhs.data:
            x = self.data.get(k, 0.0)
            y = rhs.data.get(k, 0.0)
            total += x * y
        return total

    @override
    def map(self, f: Callable[[Tensor[S]], Tensor[S]], /) -> Self:
        return type(self)({k: f(v) for k, v in self.data.items()}, self.basis)

    @override
    def map2(self, rhs: "Self | DualParts[DynDualBasis, tuple[()]]", f: Callable[[Tensor[S], Tensor[S] | Tensor[tuple[()]]], Tensor[S]], /) -> Self:
        """Apply the given function to all dual parts in `self`."""
        assert isinstance(rhs, DynDualPart)
        return type(self)({k: f(self.data.get(k, 0.0), rhs.data.get(k, 0.0)) for k in self.data.keys() | rhs.data.keys()}, self.basis)  # pyright: ignore[reportArgumentType]

    @override
    def map2alt(self,
        rhs: "Self | DualParts[DynDualBasis, tuple[()]]",
        f: Callable[[Tensor[S], Tensor[S]], Tensor[S]],
        fl: Callable[[Tensor[S]], Tensor[S]],
        fr: Callable[[Tensor[S]], Tensor[S]],
        /
    ) -> Self:
        """Apply the given functions to all dual parts in `self` and `rhs`."""
        assert isinstance(rhs, DynDualPart)
        return type(self)(
            {k: f(self.data[k], rhs.data[k]) for k in self.data.keys() & rhs.data.keys()}  # pyright: ignore[reportArgumentType]
            | {k: fl(self.data[k]) for k in self.data.keys() - rhs.data.keys()}
            | {k: fr(rhs.data[k]) for k in rhs.data.keys() - self.data.keys()},  # pyright: ignore[reportArgumentType]
            self.basis
        )

    @override
    def zero(self, /) -> Self:
        """Return a zero with the same shape as `self` and the same basis."""
        raise NotImplementedError  # TODO

    @override
    def clone(self, /) -> Self:
        """Create a new copy that is (stochastically) indipendent from `self`."""
        raise NotImplementedError  # TODO
