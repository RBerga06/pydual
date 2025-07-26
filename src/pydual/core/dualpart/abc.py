"""ABCs / Protocols for dual parts."""
from typing import Callable, Protocol, Self, overload
from .._np import Shape, Tensor

__all__ = [
    "DualBasis", "DualPart",
    "Callback1Scalar", "Callback1Vector", "Callback1",
]


type Callback1Scalar[S: Shape, Z: Shape = S] = Callable[[Tensor[S]], Tensor[Z]]

class Callback1Vector[S: Shape, Z: Shape = S](Protocol):
    def __call__[N: int](self, arg: Tensor[tuple[*S, N]], /) -> Tensor[tuple[*Z, N]]: ...

class Callback1[S: Shape, Z: Shape = S](Protocol):
    @overload
    def __call__(self, arg: Tensor[S], /) -> Tensor[Z]: ...
    @overload
    def __call__[N: int](self, arg: Tensor[tuple[*Z, N]], /) -> Tensor[tuple[*Z, N]]: ...


class Callback2[S: Shape](Protocol):
    @overload
    def __call__(self, lhs: Tensor[S], rhs: Tensor[S], /) -> Tensor[S]: ...
    @overload
    def __call__[N: int](self, lhs: Tensor[tuple[*S, N]], rhs: Tensor[tuple[*S, N]], /) -> Tensor[tuple[*S, N]]: ...


class DualPart[Basis: DualBasis, S: Shape](Protocol):
    def get_basis(self, /) -> Basis:
        """The basis where `self` lives."""
        raise NotImplementedError

    def cov(self, rhs: Self, /) -> Tensor[S]:
        """
        Evaluate the covariance between `self` and `rhs` (under the basis `self.basis`).

        NOTE: You should first make sure that `rhs` really has the same basis as `self`!
        """
        raise NotImplementedError

    def std(self, /) -> Tensor[S]:
        """Evaluate the standard deviation of `self` (under the basis `self.basis`)."""
        return self.cov(self) ** 0.5  # pyright: ignore[reportReturnType]

    def map_[Z: Shape](self, /, f_scalar: Callback1Scalar[S, Z], f_vector: Callback1Vector[S, Z]) -> "DualPart[Basis, Z]":
        """Apply the given functions to all elements in `self`."""
        raise NotImplementedError

    def map[Z: Shape](self, f: Callback1[S, Z], /) -> "DualPart[Basis, Z]":
        """Apply the given function to all elements in `self`."""
        return self.map_(f, f)  # pyright: ignore[reportArgumentType]  # TODO: pyright bug?

    def map2(self, rhs: Self, f: Callback2[S], /) -> Self:
        """
        Apply the given function to all elements in `self` and `rhs`.

        NOTE: You should first make sure that `rhs` really has the same basis as `self`!
        """
        raise NotImplementedError

    def clone(self, /) -> Self:
        """Create a new copy that is (stochastically) indipendent from `self`."""
        raise NotImplementedError


class DualBasis(Protocol):
    """A basis for a space of dual parts."""

    def zero_like[S: Shape](self, dual: DualPart[Self, S], /) -> DualPart[Self, S]:
        """Return a zero/empty dual part with the same shape as `dual` and `self` as a basis."""
        raise NotImplementedError
