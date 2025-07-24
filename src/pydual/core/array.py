"""NumPy arrays of dual numbers."""
from collections.abc import Iterator
from typing import Any, Literal, Self, cast, final, overload
from .abc import DualParts, Mat, Vec, Shape, Tensor
from dataclasses import dataclass
import numpy as np

type dTensor[B, S: Shape] = dual[B, S]
type dMat[B, N: int = int, M: int = N] = dTensor[B, tuple[N, M]]
type dVec[B, N: int = int] = dTensor[B, tuple[N]]
type dScalar[B] = dTensor[B, tuple[()]]


@final
@dataclass(slots=True, frozen=True, eq=False, match_args=True)
class dual[B, S: Shape]:
    """A NumPy ndarray of dual numbers."""
    dreal: Tensor[S]
    """The real part."""
    ddual: DualParts[B, S]
    """The dual part."""

    @property
    def shape(self, /) -> S:
        return self.dreal.shape

    def display[X: np.number](self,
        *,
        fmt: str = "",
        ufmt: str | None = None,
        packed: bool = True,
    ) -> str:
        if packed and np.sum(self.shape) != 0:
            x = np.array([x.display(fmt=fmt, ufmt=ufmt, packed=packed) for x in self])  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues, reportUnknownArgumentType, reportUnknownMemberType]
            return np.array2string(x, separator=",", formatter={"str_kind": lambda x: x})  # pyright: ignore[reportArgumentType, reportUnknownLambdaType, reportUnknownArgumentType]
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

    def __add__(self, rhs: dTensor[B, S] | Tensor[S] | dScalar[B] | float) -> dTensor[B, S]:
        if isinstance(rhs, dual):
            return type(self)(
                self.dreal + rhs.dreal,  # pyright: ignore[reportArgumentType]
                self.ddual.map2(rhs.ddual, lambda x, y: x + y),  # pyright: ignore[reportArgumentType]
            )
        return type(self)(self.dreal + rhs, self.ddual)  # pyright: ignore[reportArgumentType]

    __radd__ = __add__

    def __sub__(self: dTensor[B, S] | dScalar[B], rhs: dTensor[B, S] | Tensor[S] | dScalar[B] | float) -> dTensor[B, S]:
        if isinstance(rhs, dual):
            return dual[B, S](
                self.dreal - rhs.dreal,  # pyright: ignore[reportArgumentType]
                self.ddual.map2(rhs.ddual, lambda x, y: x - y),  # pyright: ignore[reportArgumentType]
            )
        return dual[B, S](self.dreal + rhs, self.ddual.map(lambda x: -x))  # pyright: ignore[reportArgumentType]

    def __rsub__(self, lhs: Tensor[S] | float) -> dTensor[B, S]:
        return type(self)(lhs - self.dreal, self.ddual.map(lambda x: -x))  # pyright: ignore[reportArgumentType]

    @overload  # (scalar * array)
    def __mul__[Z: tuple[int, ...]](self: dScalar[B], rhs: dTensor[B, Z] | Tensor[Z]) -> dTensor[B, Z]: ...
    @overload  # (array * array) or (array * scalar)
    def __mul__(self, rhs: dTensor[B, S] | Tensor[S] | dScalar[B] | float) -> dTensor[B, S]: ...
    def __mul__[Z: tuple[int, ...]](  # pyright: ignore[reportInconsistentOverload]
        self: dTensor[B, Z] | dScalar[Z], rhs: dTensor[B, Z] | Tensor[Z] | dScalar[B] | float
    ) -> dTensor[B, Z]:
        if isinstance(rhs, dual):
            return dual[B, Z](
                self.dreal * rhs.dreal,  # pyright: ignore[reportArgumentType]
                self.ddual.map2(rhs.ddual, lambda dl, dr: dl * rhs.dreal + dr * self.dreal),  # pyright: ignore[reportArgumentType]
            )
        return dual[B, Z](self.dreal * rhs, self.ddual.map(lambda x: x * rhs))  # pyright: ignore[reportArgumentType]

    __rmul__ = __mul__

    @overload  # (scalar / array)
    def __truediv__[Z: tuple[int, ...]](self: dScalar[B], rhs: dTensor[B, Z] | Tensor[Z]) -> dTensor[B, Z]: ...
    @overload  # (array / array) or (array / scalar)
    def __truediv__(self, rhs: dTensor[B, S] | Tensor[S] | dScalar[B] | float) -> dTensor[B, S]: ...
    def __truediv__[Z: tuple[int, ...]](  # pyright: ignore[reportInconsistentOverload]
        self: dTensor[B, Z] | dScalar[Z], rhs: dTensor[B, Z] | Tensor[Z] | dScalar[B] | float
    ) -> dTensor[B, Z]:
        if isinstance(rhs, dual):
            return dual[B, Z](
                self.dreal / rhs.dreal,  # pyright: ignore[reportArgumentType]
                self.ddual.map2(rhs.ddual, lambda dl, dr: dl / rhs.dreal - dr * self.dreal / rhs.dreal**2),  # pyright: ignore[reportArgumentType]
            )
        return dual[B, Z](self.dreal * rhs, self.ddual.map(lambda x: x * rhs))  # pyright: ignore[reportArgumentType]

    def __rtruediv__(self, lhs: Tensor[S] | float) -> dTensor[B, S]:
        return dual[B, S](
            lhs / self.dreal,  # pyright: ignore[reportArgumentType]
            self.ddual.map(lambda dr: -dr * lhs / self.dreal ** 2)  # pyright: ignore[reportArgumentType]
        )

    def __pow__(self, rhs: dTensor[B, S] | Tensor[S] | float) -> dTensor[B, S]:
        if isinstance(rhs, dual):
            real: Tensor[S] = self.dreal**rhs.dreal  # pyright: ignore[reportAssignmentType]
            return dual(
                real,
                self.ddual.map2(rhs.ddual, lambda dl, dr: real * (rhs.dreal * dl / self.dreal + dr * np.log(self.dreal)))  # pyright: ignore[reportArgumentType]
            )
        elif rhs == 0.0:
            # TODO: We should also apply this special case element-wise: for example, [4.2, -0.0] ** [3.2, 0.0] should not fail!
            return dual[B, S](np.ones(self.dreal.shape), self.ddual.zero())
        else:
            return dual[B, S](
                self.dreal**rhs,  # pyright: ignore[reportArgumentType]
                self.ddual.map(lambda dl: rhs * self.dreal ** (rhs - 1) * dl), # pyright: ignore[reportArgumentType]
            )

    def __rpow__(self, lhs: float) -> dTensor[B, S]:
        real: Tensor[S] = lhs**self.dreal  # pyright: ignore[reportAssignmentType]
        return type(self)(
            real,
            self.ddual.map(lambda dr: real * dr * np.log(lhs)),  # pyright: ignore[reportArgumentType]
        )

    @overload
    def __matmul__[N: int](
        self: dTensor[B, tuple[N]], rhs: dTensor[B, tuple[N]] | Vec[N]
    ) -> dTensor[B, tuple[N]]: ...
    @overload
    def __matmul__[N: int, M: int](
        self: dTensor[B, tuple[N]], rhs: dMat[B, N, M] | Mat[N, M]
    ) -> dTensor[B, tuple[M]]: ...
    @overload
    def __matmul__[N: int, M: int](
        self: dMat[B, N, M], rhs: dTensor[B, tuple[M]] | Vec[M]
    ) -> dTensor[B, tuple[N]]: ...
    @overload
    def __matmul__[N: int, M: int, K: int](
        self: dMat[B, N, M], rhs: dMat[B, M, K] | Mat[M, K]
    ) -> dTensor[B, tuple[N, K]]: ...
    def __matmul__(  # pyright: ignore[reportInconsistentOverload]
        self: dTensor[B, tuple[int, ...]], rhs: dTensor[B, tuple[int, ...]] | Tensor[tuple[int, ...]]
    ) -> dTensor[B, tuple[int, ...]]:
        if isinstance(rhs, dual):
            return type(self)(
                self.dreal @ rhs.dreal,
                self.ddual.map2alt(
                    rhs.ddual,
                    lambda dl, dr: dl @ rhs.dreal + self.dreal @ dr,
                    lambda dl: dl @ rhs.dreal,
                    lambda dr: self.dreal @ dr,
                )
            )
        return type(self)(self.dreal @ rhs, self.ddual.map(lambda dl: dl @ rhs))

    @property
    def mT[N: int, M: int](self: dMat[B, N, M], /) -> dTensor[B, tuple[M, N]]:  # TODO: Type this for higher-dimensional objects
        """Transpose `self` as a matrix."""
        return type(self)(self.dreal.mT, self.ddual.map(lambda dl: dl.mT))  # pyright: ignore[reportReturnType]

    @classmethod
    def _from_real_and_derivative(
        cls,
        real: Tensor[S],
        derivative: Tensor[S],
        dual: DualParts[B, S],
    ) -> dTensor[B, S]:
        return cls(real, dual.map(lambda d: d * derivative))  # pyright: ignore[reportArgumentType]

    def exp(self: dTensor[B, S], /) -> dTensor[B, S]:
        real = np.exp(self.dreal)
        return self._from_real_and_derivative(real, real, self.ddual)  # pyright: ignore[reportArgumentType]

    def sin(self: dTensor[B, S], /) -> dTensor[B, S]:
        return self._from_real_and_derivative(np.sin(self.dreal), np.cos(self.dreal), self.ddual)  # pyright: ignore[reportArgumentType]

    def cos(self: dTensor[B, S], /) -> dTensor[B, S]:
        return self._from_real_and_derivative(np.cos(self.dreal), -np.sin(self.dreal), self.ddual)  # pyright: ignore[reportArgumentType]

    def tan(self: dTensor[B, S], /) -> dTensor[B, S]:
        real = cast(Tensor[S], np.tan(self.dreal))
        diff = cast(Tensor[S], real ** 2 + 1.0)
        return self._from_real_and_derivative(real, diff, self.ddual)

    def sinh(self: dTensor[B, S], /) -> dTensor[B, S]:
        real = cast(Tensor[S], np.sinh(self.dreal))
        diff = cast(Tensor[S], np.cosh(self.dreal))
        return self._from_real_and_derivative(real, diff, self.ddual)

    def cosh(self: dTensor[B, S], /) -> dTensor[B, S]:
        real = cast(Tensor[S], np.cosh(self.dreal))
        diff = cast(Tensor[S], np.sinh(self.dreal))
        return self._from_real_and_derivative(real, diff, self.ddual)

    def tanh[X: np.floating](self: "dTensor[B, S]", /) -> "dTensor[B, S]":
        real = cast(Tensor[S], np.tanh(self.dreal))
        diff = cast(Tensor[S], 1.0 - np.tanh(self.dreal) ** 2)
        return self._from_real_and_derivative(real, diff, self.ddual)

    def matinv[N: int](self: dMat[B, N], /) -> dMat[B, N]:
        """Matrix inverse."""
        real = np.linalg.inv(self.dreal)
        return dual[B, tuple[N, N]](real, self.ddual.map(lambda d: -real @ d @ real))  # pyright: ignore[reportArgumentType]

    def _matpow_positive[N: int](self: dMat[B, N], n: int, /) -> dMat[B, N]:
        # TODO: Make this recursive algorithm into an iterative one
        if n == 1:
            return self
        n, bit = divmod(n, 2)
        y = self._matpow_positive(n)
        y @= y
        if bit != 0:
            y @= self
        return y

    def matpow[N: int](self: dMat[B, N], n: int, /) -> dMat[B, N]:
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
    def __iter__[N: int](self: "dTensor[B, tuple[N]]", /) -> Iterator[dScalar[B]]: ...
    @overload
    def __iter__[Z: Shape](self: dTensor[B, tuple[int, *Z]], /) -> Iterator[dTensor[B, Z]]: ...
    def __iter__[Z: Shape](self: dTensor[B, tuple[int, *Z]], /) -> Iterator[dTensor[B, Z]]:  # pyright: ignore[reportInconsistentOverload]
        for i, x in enumerate(self.dreal):
            yield dual(x, self.ddual.map(lambda d: d[i]))  # pyright: ignore[reportReturnType, reportAny]

    @overload  # element access (vector)
    def __getitem__[N: int](self: dVec[B, N], key: N, /) -> dScalar[B]: ...
    @overload  # element access (any shape)
    def __getitem__[Z: tuple[int, ...]](self: dTensor[B, Z], key: Z, /) -> dScalar[B]: ...
    @overload  # slicing (vector)
    def __getitem__(self: dVec[B], key: "slice[int | None, int | None, int | None]", /) -> dVec[B]: ...
    @overload  # mask (vector)
    def __getitem__(self: dVec[B], key: Vec[int, np.bool_], /) -> dVec[B]: ...
    @overload  # mask (matrix)
    def __getitem__(self: dMat[B], key: Mat[int, int, np.bool_], /) -> dMat[B]: ...
    def __getitem__(self, key: Any) -> float | dTensor[B, tuple[int, ...]]:  # pyright: ignore[reportExplicitAny, reportAny]
        return dual[B, tuple[int, ...]](self.dreal[key], self.ddual.map(lambda x: x[key]))  # pyright: ignore[reportArgumentType]

    @overload
    def as_tuple(self: dVec[B, Literal[0]], /) -> tuple[()]: ...
    @overload
    def as_tuple(self: dVec[B, Literal[1]], /) -> tuple[dScalar[B]]: ...
    @overload
    def as_tuple(self: dVec[B, Literal[2]], /) -> tuple[dScalar[B], dScalar[B]]: ...
    @overload
    def as_tuple(self: dVec[B, Literal[3]], /) -> tuple[dScalar[B], dScalar[B], dScalar[B]]: ...
    @overload
    def as_tuple(self: dVec[B], /) -> tuple[dScalar[B], ...]: ...
    def as_tuple(self: dVec[B], /) -> tuple[dScalar[B], ...]:  # pyright: ignore[reportInconsistentOverload]
        return tuple(self)

    def sum[N: int](
        self: dVec[B, N],
        /,
        *,
        axis: Literal[0] | None = None,
        out: None = None,  # pyright: ignore[reportUnusedParameter]
    ) -> dScalar[B]:
        return dual[B, tuple[()]](
            np.sum(self.dreal, axis=axis),  # pyright: ignore[reportAny]
            self.ddual.map(lambda dx: np.sum(dx, axis=axis)),  # pyright: ignore[reportArgumentType, reportAny]
        )

    def average[N: int](self: dVec[B, N], /, *, weights: Vec[N] | bool = True) -> dScalar[B]:
        if weights is True:
            weights = self.ddual.std() ** -2
        if weights is False:
            return self.sum() / self.shape[0]
        else:
            return np.sum([x * w for x, w in zip(self, weights)]) / np.sum(weights)  # pyright: ignore[reportAny]
