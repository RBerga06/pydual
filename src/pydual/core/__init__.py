"""Dual numbers for Python."""
from dataclasses import dataclass


@dataclass(frozen=True)
class dual[T = float]:
    """Minimal implementation of a dual number."""

    real: T
    dual: T

    def __add__(self, rhs: T, /) -> "dual[T]":
        return self


def foo(x: dual[float]):
    pass
