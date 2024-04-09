# GUM - Generalized Unrestricted Models, a versatile regression toolbox for Matlab

# Overview
GUM is a versatile Matlab toolbox for running Generalized Unrestricted Models (GUMs). GUMs is a very general modelling framework including Generalized Linear Models (GLMs), Generalized Additive Models (GAMs) and many other families of models. 

Key features include:
- Simple model definition using R-style formulas by combining different operators on sets of regressors, including: linear mapping of regressors (e.g. `y ~ x1 + x2`), non-linear mapping of regressors (`y ~ f(x1) + f(x2)`), splitting regressors on categorical variable (`y ~ x1|x2`), lagged regressors (`y ~ lag(x1;Lags=1:5)`), multiplication of regressors (`y ~ f(x1)(x2 + f(x3))`).
- Binary, continuous or count output variables (with Bernoulli, gaussian or Poisson observation noise, respectively)
- Single line command to generate nice intuitive plots of estimated weights
- Frequentist or Bayesian treatment of linear weights
- Nonlinear mapping of regressors can be captured using sets of basis functions or Gaussian Processes
- Approximate posterior over weights using Laplace approximation; fitting of hyperparameters (for covariance prior or basis functions) using Expectation-Maximization or gradient-based cross-validation
- Plotting design matrix & Variance Inflation Factor
- Perform model validation by comparing fitted model prediction vs observed data
- Generate tables and plots for model comparison
- Easily combine model fits (e.g. plotting population average from different neuron/subjects)

The toolbox uses object-orienting programming: a model is defined from a dataset table `T` and formula `fmla` using `M = gum(T, fmla);`. Methods allow to estimate the model, plot weights, design matrix, perform model comparison (among many other options) in a single-line code.

![alt text](https://github.com/ahyafil/gum/blob/main/tutorials/GUM%20principle.png?raw=true)

Check our [preprint](https://arxiv.org/abs/2002.00920). Tutorials will soon follow.

# Installation

- Clone or download repository
- Add the folder (with subfolders!) to your Matlab path
- Add the [Matrix Computation Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/2360-the-matrix-computation-toolbox) to your Matlab path
- Enjoy!

# Contact

Refer to the [Github forum](https://github.com/ahyafil/gum/discussions) for help on using GUM.

Please report bugs [here](https://github.com/ahyafil/gum/issues). This includes any problems with the documentation. Pull Requests for bugs are greatly appreciated.

For all other questions or comments, contact [alexandre.hyafil@gmail.com]
