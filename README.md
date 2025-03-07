# RcppDPR [![codecov](https://codecov.io/gh/mohammed321/RcppDPR/graph/badge.svg?token=0CZHARW6XF)](https://codecov.io/gh/mohammed321/RcppDPR)
This package provides an Rcpp reimplementation of the Bayesian non-parametric **D**irichlet **P**rocess **R**egression model first published in Zeng & Zhou 2017 (https://doi.org/10.1038/s41467-017-00470-2).  A full Bayesian version is implemented with Gibbs sampling, as a well a faster but less accurate variational Bayes approximation.

**D**irichlet **P**rocess **R**egression is a generalization of penalized regression methods such as ridge regression, LASSO, and elastic net .  These methods can be understood from a Bayesian perspective as setting a prior on the effect sizes of the coefficients.  Ridge regression, for example, is equivalent to a Gaussian prior, while LASSO is equivalent to a Laplace prior.  Ultimately these prior can all be parameterized as specific scale mixtures of normal distributions.  DPR generalizes this model by allowing the prior to be a scale mixture of an arbitrary number of normal distributions.  The Adaptive Gibbs sampling mode provides one strategy for choosing this number.

## Installation
To install the development version of the package, use `remote::install_github`:
```
remotes::install_github("mohammed321/RcppDPR")
```
