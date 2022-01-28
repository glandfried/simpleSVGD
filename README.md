# simpleSVGD

This package is a tiny SVGD algorithm specifically developed to operate on
distributions found in [HMCLab](https://github.com/larsgeb/HMCLab). 

By default, this package uses **radial basis functions** to compute sample
interaction and **AdaGrad** to optimize the samples.

## Installation:

We recommend using at least Python 3.7.

To get the latest release, simply use pip inside your favourite environment:
```sh
pip install simpleSVGD
```

To install the latest version directly from GitHub:

```sh
git clone git@github.com:larsgeb/simpleSVGD.git
cd simpleSVGD
pip install -e .
```
## Mini-tutorial

This package can be used with minimal development. The only thing one needs to 
supply to the algorithm is:

1. The gradient of the function to optimize, `gradient_fn(samples)`. The function itself is not needed.
2. An initial collection of samples `initial_samples`, a `numpy.array`. It helps if these are close to the target
function/distribution. 

It is essential to get the input/output shapes of the target (gradient) right. As input, it should take an arbitrary amount of samples, with the appropriate dimensionality. This means if ones wants 430 samples on a 3 dimensional function, the input/output shapes looks like this:
```python
output_gradient = gradient_fn(input_samples)

input_samples.shape = (430, 3)
output_gradient.shape = (430, 3)
```

Typically, it is useful to instantiate the samples using a Normal distribution. Using NumPy, this is done with:
```python
import numpy as np
np.random.seed(235)

mean = 0
standard_dev = 1
n_samples = 100
dimensions = 2

initial_samples = np.random.normal(mean, standard_dev, [n_samples, dimensions])
```

A good 2-dimensional test function would be the Himmelblau function:
```python
def Himmelblau(input_array: np.array) -> np.array:

    # As this is a 2-dimensional function, assert that the passed input_array
    # is correct.
    assert input_array.shape[1] == 2

    # To simplify reading this function, we do this step in between. It is not
    # the most optimal way to program this.
    x = input_array[:, 0, None]
    y = input_array[:, 1, None]

    output_array = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    # As the output should be a scalar function, assert that the
    # output is also length 1 in dim 2 of the array.
    assert output_array.shape == (input_array.shape[0], 1)

    smoothing = 100
    return output_array / smoothing
```
and its gradient:
```python
def Himmelblau_grad(input_array: np.array) -> np.array:

    # As this is a 2-dimensional function, assert that the passed input_array
    # is correct.
    assert input_array.shape[1] == 2

    # To simplify reading this function, we do this step in between. It is not
    # the most optimal way to program this.
    x = input_array[:, 0, None]
    y = input_array[:, 1, None]

    # Compute partial derivatives and combine them
    output_array_dx = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    output_array_dy = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
    output_array = np.hstack((output_array_dx, output_array_dy))

    # Check if the output shape is correct
    assert output_array.shape == input_array.shape

    smoothing = 100
    return output_array / smoothing
```

```python
initial_samples = np.random.normal(0, 3, [1000, 2])

#%matplotlib notebook

figure = plt.figure(figsize=(6, 6))
plt.xlabel("Parameter 0")
plt.ylabel("Parameter 1")
plt.title("SVGD animation on the Himmelblau function")


final_samples = simpleSVGD.update(
    initial_samples,
    Himmelblau_grad,
    n_iter=130,
    stepsize=1e-1,
    alpha=0.9,
    #animate=True,
    #background=background,
    #figure=figure,
)
```

When running this code in a notebook, one can uncomment the animation lines to produce the following:


https://user-images.githubusercontent.com/21038893/151603377-a473e7b1-f7b4-417b-a685-9c0cfa98dc15.mov



## Stein Variational Gradient Descent (SVGD) 
SVGD is a general purpose variational inference algorithm that forms a natural
counterpart of gradient descent for optimization. SVGD iteratively transports a
set of particles to match with the target distribution, by applying a form of
functional gradient descent that minimizes the KL divergence.

For more information, please visit the original implementers project website -
[SVGD](http://www.cs.utexas.edu/~qlearning/project.html?p=vgd), or their
publication Qiang Liu and Dilin Wang. [Stein Variational Gradient Descent (SVGD): A General Purpose Bayesian Inference Algorithm](http://arxiv.org/abs/1608.04471). NIPS, 2016.
