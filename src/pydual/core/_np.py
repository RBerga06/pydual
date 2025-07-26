"""Typing utilities for numpy."""
import numpy as np

type Shape = tuple[int, ...]
type Tensor[S: Shape, T: np.generic = np.float64] = np.ndarray[S, np.dtype[T]]
type Mat[N: int, M: int = N, T: np.generic = np.float64] = Tensor[tuple[N, M], T]
type Vec[N: int, T: np.generic = np.float64] = Tensor[tuple[N], T]
type Scalar[T: np.generic = np.float64] = Tensor[tuple[()], T]

__all__ = ["Shape", "Tensor", "Vec", "Mat"]
