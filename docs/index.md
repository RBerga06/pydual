# Dual numbers and uncertainties

`pydual` provides an implementation of autodifferentiation with dual numbers
that allows to effortlessly model and propagate uncertainties through calculations.

## Example
```py
import numpy as np
from pydual.core import dual

# load/create data as a numpy array like you already do
x_best = np.array([3.14, 4.20, 5.25, 6.32, 8.11])
x_errs = np.array([0.01, 0.02, 0.01, 0.01, 0.02]) #(1)!

# store the data as a dual number
x = dual.from_data(x_best, sigma=x_errs) #(2)!

# access the real and dual parts
print(x.dreal)  # the real part of `x`: original `x_best` data (3)
print(x.ddual)  # the dual part of `x`: embeds `x_errs` data (4)

# evaluate uncertainties
print(x.ddual.std())  # evaluate the standard deviation of all random variables (5)
print(x.display())    # print `x` as an array of '<best> ± <delta>' elements (6)

# calculate array sum
print(x.sum().display()) #(7)!

# calculate array average (arithmetic mean)
print(x.average().display())
```

1. Error bars on `x_best`
2. The `x` variable is a `dual` instance and represents a vector (1D array) of dual numbers: you can think of it as an array of random variables, all normally distributed (with standard deviations dictated by `x_errs`) and stochastically independent.
3. This attribute is called `.dreal` to avoid confusion with the `.real` attribute of `complex` numbers (we currently don't support complex numbers in the real or dual parts, but we'd like to in the future) - the leading `d` stands for “dual number”.
4. This attribute is called `.ddual` (and not `.dual`) for symmetry with `.dreal`.
5. If the random variables are normally distributed, their standard deviations are the best estimators for uncertainties. Note that the result is equal to `x_errs`.
6. The `.display()` method uses `.ddual.std()` under the hood, hence uncertainties are estimated this way.
7. Note that the dual part is automatically calculated as the square root of the sum of hte squares of the elements of `x.ddual.std()`, because they are independent.

## Future directions

`pydual` aims to implement the following features:

- Complex numbers support
- Custom/arbitrary error distribution (currently it implicitely only supports normal distributions)
- Full [Array API](https://data-apis.org/array-api/latest/) support
