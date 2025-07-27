"""NumPy arrays of dual numbers."""

# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
from collections.abc import Iterator
from typing import Any, Literal, Self, cast, final, overload
from dataclasses import dataclass
import numpy as np

from ._np import Shape, Tensor, Mat, Vec
from .dualpart import DualPart
from .dualpart.dyn import DynDualPart, DynDualBasis
from .dualpart.fixed import FixedDualBasis

type dTensor[S: Shape] = dual[S]
type dMat[N: int, M: int = N] = dTensor[tuple[N, M]]
type dVec[N: int] = dTensor[tuple[N]]
type dScalar = dTensor[tuple[()]]


@final
@dataclass(slots=True, frozen=True, eq=False, match_args=True)
class dual[S: Shape]:
    """A NumPy ndarray of dual numbers."""

    dreal: Tensor[S]
    """The real part."""
    ddual: DynDualPart[S]
    """The dual part."""

    @property
    def shape(self, /) -> S:
        return self.dreal.shape

    def display[X: np.number](
        self,
        *,
        fmt: str = "",
        ufmt: str | None = None,
        packed: bool = True,
    ) -> str:
        if packed and np.sum(self.shape) != 0:
            self = cast(
                dTensor["tuple[int, *S]"], self
            )  # inform Pyright that `self` is at least 1-dimensional
            x = np.array([x.display(fmt=fmt, ufmt=ufmt, packed=packed) for x in self])
            return np.array2string(
                x,
                separator=",",
                formatter={"str_kind": lambda x: x},  # pyright: ignore[reportArgumentType]
            )
        if ufmt is None:
            ufmt = fmt
        return f"{self.dreal:{fmt}} ± {self.ddual.std():{ufmt}}"

    def clone(self, /) -> Self:
        """Create a completely indipendent copy of `self`."""
        return type(self)(self.dreal, self.ddual.clone())

    @property
    def dreal_and_std[X: np.number](self, /) -> tuple[Tensor[S], Tensor[S]]:
        return (self.dreal, self.ddual.std())

    def __pos__(self, /) -> Self:
        return self

    def __neg__(self, /) -> Self:
        return type(self)(-self.dreal, self.ddual.map(lambda x: -x))

    def __add__(self, rhs: dTensor[S] | Tensor[S] | dScalar | float) -> dTensor[S]:
        if isinstance(rhs, dual):
            return type(self)(
                self.dreal + rhs.dreal,  # pyright: ignore[reportArgumentType]
                self.ddual.map2(rhs.ddual, lambda x, y: x + y),  # pyright: ignore[reportArgumentType]
            )
        return type(self)(self.dreal + rhs, self.ddual)  # pyright: ignore[reportArgumentType]

    __radd__ = __add__

    def __sub__(
        self: dTensor[S] | dScalar, rhs: dTensor[S] | Tensor[S] | dScalar | float
    ) -> dTensor[S]:
        if isinstance(rhs, dual):
            return dual[S](
                self.dreal - rhs.dreal,  # pyright: ignore[reportArgumentType]
                self.ddual.map2_(
                    self.shape,  # pyright: ignore[reportArgumentType]
                    rhs.ddual,  # pyright: ignore[reportArgumentType]
                    rhs.shape,
                    lambda x, y: x - y,
                    lambda x, y: x - y,
                ),
            )
        return dual[S](self.dreal + rhs, self.ddual.map(lambda x: -x))  # pyright: ignore[reportArgumentType]

    def __rsub__(self, lhs: Tensor[S] | float) -> dTensor[S]:
        return type(self)(lhs - self.dreal, self.ddual.map(lambda x: -x))  # pyright: ignore[reportArgumentType]

    @overload  # (scalar * array)
    def __mul__[Z: tuple[int, ...]](
        self: dScalar, rhs: dTensor[Z] | Tensor[Z]
    ) -> dTensor[Z]: ...
    @overload  # (array * array) or (array * scalar)
    def __mul__(self, rhs: dTensor[S] | Tensor[S] | dScalar | float) -> dTensor[S]: ...
    def __mul__[Z: tuple[int, ...]](
        self: dTensor[Z] | dScalar, rhs: dTensor[Z] | Tensor[Z] | dScalar | float
    ) -> dTensor[Z]:
        if isinstance(rhs, dual):
            return dual[Z](
                self.dreal * rhs.dreal,  # pyright: ignore[reportArgumentType]
                self.ddual.map2(
                    rhs.ddual,  # pyright: ignore[reportArgumentType]
                    lambda dl, dr: dl * rhs.dreal + dr * self.dreal,
                ),
            )
        return dual[Z](self.dreal * rhs, self.ddual.map(lambda x: x * rhs))  # pyright: ignore[reportArgumentType]

    __rmul__ = __mul__

    @overload  # (scalar / array)
    def __truediv__[Z: tuple[int, ...]](
        self: dScalar, rhs: dTensor[Z] | Tensor[Z]
    ) -> dTensor[Z]: ...
    @overload  # (array / array) or (array / scalar)
    def __truediv__(
        self, rhs: dTensor[S] | Tensor[S] | dScalar | float
    ) -> dTensor[S]: ...
    def __truediv__[Z: tuple[int, ...]](
        self: dTensor[Z] | dScalar, rhs: dTensor[Z] | Tensor[Z] | dScalar | float
    ) -> dTensor[Z]:
        if isinstance(rhs, dual):
            return dual[Z](
                self.dreal / rhs.dreal,  # pyright: ignore[reportArgumentType]
                self.ddual.map2(
                    rhs.ddual,  # pyright: ignore[reportArgumentType]
                    lambda dl, dr: dl / rhs.dreal - dr * self.dreal / rhs.dreal**2,
                ),
            )
        return dual[Z](self.dreal * rhs, self.ddual.map(lambda x: x * rhs))  # pyright: ignore[reportArgumentType]

    def __rtruediv__(self, lhs: Tensor[S] | float) -> dTensor[S]:
        return dual[S](
            lhs / self.dreal,  # pyright: ignore[reportArgumentType]
            self.ddual.map(lambda dr: -dr * lhs / self.dreal**2),
        )

    def __pow__(self, rhs: dTensor[S] | Tensor[S] | float) -> dTensor[S]:
        if isinstance(rhs, dual):
            real: Tensor[S] = self.dreal**rhs.dreal  # pyright: ignore[reportAssignmentType]
            return dual(
                real,
                self.ddual.map2(
                    rhs.ddual,
                    lambda dl, dr: real
                    * (rhs.dreal * dl / self.dreal + dr * np.log(self.dreal)),
                ),
            )
        elif rhs == 0.0:
            # TODO: We should also apply this special case element-wise: for example, [4.2, -0.0] ** [3.2, 0.0] should not fail!
            return dual[S](np.ones(self.dreal.shape), self.ddual.zero())
        else:
            return dual[S](
                self.dreal**rhs,  # pyright: ignore[reportArgumentType]
                self.ddual.map(lambda dl: rhs * self.dreal ** (rhs - 1) * dl),
            )

    def __rpow__(self, lhs: float) -> dTensor[S]:
        real: Tensor[S] = lhs**self.dreal  # pyright: ignore[reportAssignmentType]
        return type(self)(real, self.ddual.map(lambda dr: real * dr * np.log(lhs)))

    @overload
    def __matmul__[N: int](self: dVec[N], rhs: dVec[N] | Vec[N]) -> dScalar: ...
    @overload
    def __matmul__[N: int, M: int](
        self: dVec[N], rhs: dMat[N, M] | Mat[N, M]
    ) -> dVec[M]: ...
    @overload
    def __matmul__[N: int, M: int](
        self: dMat[N, M], rhs: dVec[M] | Vec[M]
    ) -> dVec[N]: ...
    @overload
    def __matmul__[N: int, M: int, K: int](
        self: dMat[N, M], rhs: dMat[M, K] | Mat[M, K]
    ) -> dMat[N, K]: ...
    def __matmul__(  # pyright: ignore[reportInconsistentOverload]  # TODO: make shape parameter covariant somehow
        self: dTensor[Shape], rhs: dTensor[Shape] | Tensor[Shape]
    ) -> dTensor[Shape]:
        if isinstance(rhs, dual):
            return type(self)(
                self.dreal @ rhs.dreal,
                self.ddual.map2alt(
                    rhs.ddual,
                    lambda dl, dr: dl @ rhs.dreal + self.dreal @ dr,
                    lambda dl: dl @ rhs.dreal,
                    lambda dr: self.dreal @ dr,
                ),
            )
        return type(self)(self.dreal @ rhs, self.ddual.map(lambda dl: dl @ rhs))

    @property
    def mT[N: int, M: int, Z: Shape](
        self: dTensor["tuple[*Z, N, M]"], /
    ) -> dTensor["tuple[*Z, M, N]"]:
        """Transpose `self` as a matrix."""
        return dual(
            cast(Tensor["tuple[*Z, M, N]"], self.dreal.mT),
            self.ddual.map(lambda dl: dl.mT),  # pyright: ignore[reportUnknownMemberType]
        )

    @classmethod
    def _from_real_and_derivative(
        cls,
        real: Tensor[S],
        derivative: Tensor[S],
        dual: DualPart[DynDualBasis, S],
    ) -> dTensor[S]:
        return cls(real, dual.map(lambda d: d * derivative))  # pyright: ignore[reportArgumentType]

    def exp(self: dTensor[S], /) -> dTensor[S]:
        real = np.exp(self.dreal)
        return self._from_real_and_derivative(real, real, self.ddual)  # pyright: ignore[reportArgumentType]

    def sin(self: dTensor[S], /) -> dTensor[S]:
        return self._from_real_and_derivative(
            np.sin(self.dreal),  # pyright: ignore[reportArgumentType]
            np.cos(self.dreal),  # pyright: ignore[reportArgumentType]
            self.ddual,
        )

    def cos(self: dTensor[S], /) -> dTensor[S]:
        return self._from_real_and_derivative(
            np.cos(self.dreal),  # pyright: ignore[reportArgumentType]
            -np.sin(self.dreal),  # pyright: ignore[reportArgumentType]
            self.ddual,
        )

    def tan(self: dTensor[S], /) -> dTensor[S]:
        real = cast(Tensor[S], np.tan(self.dreal))
        diff = cast(Tensor[S], real**2 + 1.0)
        return self._from_real_and_derivative(real, diff, self.ddual)

    def sinh(self: dTensor[S], /) -> dTensor[S]:
        real = cast(Tensor[S], np.sinh(self.dreal))
        diff = cast(Tensor[S], np.cosh(self.dreal))
        return self._from_real_and_derivative(real, diff, self.ddual)

    def cosh(self: dTensor[S], /) -> dTensor[S]:
        real = cast(Tensor[S], np.cosh(self.dreal))
        diff = cast(Tensor[S], np.sinh(self.dreal))
        return self._from_real_and_derivative(real, diff, self.ddual)

    def tanh[X: np.floating](self: "dTensor[S]", /) -> "dTensor[S]":
        real = cast(Tensor[S], np.tanh(self.dreal))
        diff = cast(Tensor[S], 1.0 - np.tanh(self.dreal) ** 2)
        return self._from_real_and_derivative(real, diff, self.ddual)

    def matinv[N: int](
        self: dMat[N, N], /
    ) -> dMat[N, N]:  # TODO: Type this for higher-dimensional objects
        """Matrix inverse."""
        real = np.linalg.inv(self.dreal)
        return dual[tuple[N, N]](real, self.ddual.map(lambda d: -real @ d @ real))  # pyright: ignore[reportArgumentType]

    def _matpow_positive[N: int](self: dMat[N, N], n: int, /) -> dMat[N, N]:
        # TODO: Make this recursive algorithm into an iterative one
        if n == 1:
            return self
        n, bit = divmod(n, 2)
        y = self._matpow_positive(n)
        y @= y
        if bit != 0:
            y @= self
        return y

    def matpow[N: int](self: dMat[N, N], n: int, /) -> dMat[N, N]:
        if n > 0:
            return self._matpow_positive(n)
        elif n < 0:
            return self.matinv()._matpow_positive(-n)
        else:  # n == 0
            return dual(
                np.eye(self.dreal.shape[0], dtype=np.float64),
                self.ddual.zero(),
            )

    @overload
    def __iter__[N: int](self: dVec[N], /) -> Iterator[dScalar]: ...
    @overload
    def __iter__[Z: Shape](
        self: dTensor["tuple[int, *Z]"], /
    ) -> Iterator[dTensor[Z]]: ...
    def __iter__[Z: Shape](self: dTensor["tuple[int, *Z]"], /) -> Iterator[dTensor[Z]]:  # pyright: ignore[reportInconsistentOverload]
        for i, x in enumerate(self.dreal):
            yield dual(x, self.ddual.map(lambda d: d[i]))  # pyright: ignore[reportReturnType]

    @overload  # element access (vector)
    def __getitem__[N: int](self: dVec[N], key: N, /) -> dScalar: ...
    @overload  # element access (any shape)
    def __getitem__[Z: tuple[int, ...]](self: dTensor[Z], key: Z, /) -> dScalar: ...
    @overload  # slicing (vector)
    def __getitem__(
        self: dVec[int], key: "slice[int | None, int | None, int | None]", /
    ) -> dVec[int]: ...
    @overload  # mask (vector)
    def __getitem__(self: dVec[int], key: Vec[int, np.bool_], /) -> dVec[int]: ...
    @overload  # mask (matrix)
    def __getitem__(
        self: dMat[int, int], key: Mat[int, int, np.bool_], /
    ) -> dMat[int, int]: ...
    def __getitem__(self, key: Any) -> float | dTensor[Shape]:  # pyright: ignore[reportExplicitAny, reportAny]
        return dual[tuple[int, ...]](self.dreal[key], self.ddual.map(lambda x: x[key]))

    @overload
    def as_tuple(self: dVec[Literal[0]], /) -> tuple[()]: ...
    @overload
    def as_tuple(self: dVec[Literal[1]], /) -> tuple[dScalar]: ...
    @overload
    def as_tuple(self: dVec[Literal[2]], /) -> tuple[dScalar, dScalar]: ...
    @overload
    def as_tuple(self: dVec[Literal[3]], /) -> tuple[dScalar, dScalar, dScalar]: ...
    @overload
    def as_tuple(
        self: dVec[Literal[4]], /
    ) -> tuple[dScalar, dScalar, dScalar, dScalar]: ...
    @overload
    def as_tuple(self: dVec[int], /) -> tuple[dScalar, ...]: ...
    def as_tuple(self: dVec[int], /) -> tuple[dScalar, ...]:  # pyright: ignore[reportInconsistentOverload]
        return tuple(self)

    def sum(self, /) -> dScalar:
        """Sum all elements in the array."""
        return dual[tuple[()]](
            np.sum(self.dreal),  # pyright: ignore[reportArgumentType]
            self.ddual.map_(
                f_scalar=lambda dx: np.sum(dx),  # pyright: ignore[reportArgumentType]
                f_vector=lambda dx: np.sum(dx, axis=tuple(range(1, dx.ndim))),  # pyright: ignore[reportAny]
            ),
        )

    def average(self, /, *, weights: Tensor[S] | bool = True) -> dScalar:
        """
        Average all elements in the array.

        Arguments:
        - weights (optional): \
            if `False`, calculate the arithmetic mean; \
            if `True` (the default), calculate the weighted mean; \
            else, the given weights (a `ndarray` with the same shape as `self`) will be used.
        """
        if weights is True:
            weights = cast(Tensor[S], self.ddual.std() ** -2)
        elif weights is False:
            weights = np.ones(self.shape)
        return (self * weights).sum() / np.sum(weights)

    @staticmethod  # TODO: Can this be generalized to N-dimensional tensors somehow?
    def from_data[N: int](
        data: Vec[N], /, *, sigma: Vec[N] | None = None, cov: Mat[N] | None = None
    ) -> dVec[N]:
        """
        Convert `data` to a vector of dual numbers.

        Arguments:
        - data: the estimated/average values for N random variables
        - sigma: factors that should multiply the standard deviations of those N random variables (defined by `cov`)
        - cov: the covariance matrix between those N random variables (defaults to the identity)
        """
        n = cast(N, data.shape[-1])
        if cov is None:
            cov = cast(Mat[N], np.eye(n))
        return dual(data, DynDualPart.from_dual_parts(FixedDualBasis(cov).eye(sigma)))

    @staticmethod
    def from_real[Z: Shape](real: Tensor[Z]) -> dTensor[Z]:
        """Construct a dual number where all dual parts are set to zero."""
        return dual(real, DynDualBasis.INSTANCE.zero(real.shape))


def dreal_and_std[S: Shape](
    x: dTensor[S] | Tensor[S], /
) -> tuple[Tensor[S], Tensor[S] | None]:
    if isinstance(x, dual):
        return x.dreal_and_std
    return (x, None)


def cov[S: Shape](a: dTensor[S], b: dTensor[S]) -> Tensor[S]:
    """Evaluate the covariance between `a` and `b` (element by element)."""
    return a.ddual.cov(b.ddual)
