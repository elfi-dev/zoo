"""Gaussian Copula ABC method.

References
----------
Jingjing Li, David J. Nott, Yanan Fan, Scott A. Sisson (2016)
Extending approximate Bayesian computation methods to high dimensions
via Gaussian copula.
https://arxiv.org/abs/1504.04093v1
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.special as spc

import elfi
from elfi.distributions import estimate_densities, EmpiricalDensity, MetaGaussian

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


def _set(x):
    try:
        return set(x)
    except TypeError:
        return set([x])


def _concat_ind(x, y):
    return _set(x).union(_set(y))


def complete_informative_indices(informative_indices):
    """Complete a partial specification of the informative indices into a full one.

    Assumes that the subsets of the summary statistic that are informative for the
    bivariate marginals are given by the unions of the univariate subsets. The bivariate
    indices are the indices of the upper triangular of the correlation matrix.

    Parameters
    ----------
    informative_indices : dict
      A dictionary with values indicating the subsets of the summary statistic
      informative of each component of the parameter.

      For example: {0: {0}, 1: {1}}

    Returns
    -------
    full_indices : dict
      A dictionary with specifications for all the uni- and bivariate informative indices.
    """
    res = informative_indices.copy()
    univariate = filter(lambda p: isinstance(p, int), informative_indices)
    pairs = itertools.combinations(sorted(univariate), 2)
    for pair in pairs:
        if pair not in res:
            i, j = pair
            res[pair] = _concat_ind(res[i], res[j])

    return res


def sliced_summary(indices):
    """Construct a function that returns specific indices from a summary statistic.

    Parameters
    ----------
    indices : set, list or int
      A set of indices to keep.

    Returns
    -------
    sliced_summary
      A function which slices into an array.
    """
    indices = _set(indices)

    def summary(data):
        return data[:, sorted(indices)]
    return summary


def Distance(distance='euclidean', **kwargs):
    """Construct a factory function that produces Distance nodes.

    Parameters
    ----------
    distance : str, callable
      Specifies the distance function to use (See elfi.Distance).
    **kwargs
      Any additional arguments to elfi.Distance.

    Returns
    -------
    discrepancy_factory
      A function which takes an ELFI node and returns a Distance node.
    """
    def wrapper(sliced, index):
        return elfi.Distance(distance, sliced, name='D{}'.format(index), **kwargs)
    return wrapper


def make_distances(full_indices, summary, discrepancy_factory=None, inplace=False):
    """Construct discrepancy nodes for each informative subset of the summary statistic.

    Parameters
    ----------
    full_indices : dict
      A dictionary specifying all the informative subsets of the summary statistic.
    summary : elfi.Summary
      The summary statistic node in the inference model.
    discrepancy_factory
      A function which takes an ELFI node as an argument
      and returns a discrepancy node (e.g. elfi.Distance).
    inplace : bool
      If true, the inference model is modified in place.

    Returns
    -------
    distances
      A dictionary mapping indices to corresponding discrepancy nodes.
    """
    discrepancy_factory = discrepancy_factory or Distance()

    if not inplace:
        model_copy = summary.model.copy()
        summary_name = summary.name
        summary = model_copy[summary_name]

    res = {}
    for i, pair in enumerate(full_indices.items()):
        param, indices = pair
        sliced = elfi.Summary(sliced_summary(indices), summary, name='S{}'.format(i))
        res[param] = discrepancy_factory(sliced, i)

    return res


def make_samplers(distances, sampler_factory):
    """Construct samplers for each informative subset of the summary statistic.

    Parameters
    ----------
    distances : dict
      A dictionary with discrepancy nodes corresponding to each subset of the summary statistic.
    sampler_factory
      A function which takes a discrepancy node as an argument
      and returns an ELFI ABC sampler (e.g. elfi.Rejection).

    Returns
    -------
    samplers : dict
      A mapping from the marginals of the parameter to the corresponding sampler.
    """
    return {k: sampler_factory(dist) for (k, dist) in distances.items()}


def get_samples(marginal, samplers, parameter, n_samples, **kwargs):
    """Sample from a marginal distribution of a parameter.

    Parameters
    ----------
    marginal : int or tuple
      Specifies which marginal to sample from.
    samplers : dict
      A mapping from the marginals of the parameter to the corresponding sampler.
    parameter : str
      The name of the parameter in the inference model.
    n_samples : int
      The number of samples to produce.
    **kwargs
      Additional arguments passed for sampling.

    Returns
    -------
    samples : np.ndarray
      Samples from the specified marginal distribution.
    """
    return samplers[marginal].sample(n_samples, **kwargs).outputs[parameter][:, marginal]


def _full_cor_matrix(correlations, n):
    """Construct a full correlation matrix from pairwise correlations.

    The entries are filled with the second index changing the fastest.
    """
    I = np.eye(n)
    O = np.zeros((n, n))
    indices = itertools.combinations(range(n), 2)
    for (i, inx) in enumerate(indices):
        O[inx] = correlations[i]

    # symmetrize
    return O + O.T + I


def estimate_correlation(marginal, samplers, parameter, n_samples, **kwargs):
    """Estimate an entry in the correlation matrix.

    Parameters
    ----------
    marginal : tuple
      Specifies which marginal to sample from.
    samplers : dict
      A mapping from the marginals of the parameter to the corresponding sampler.
    parameter : str
      The name of the parameter in the inference model.
    n_samples : int
      The number of samples to produce.
    **kwargs
      Additional arguments passed for sampling.

    Returns
    -------
    correlation_coefficient : float
      The correlation coefficient corresponding to the specified bivariate marginal.
    """
    samples = get_samples(marginal, samplers=samplers, parameter=parameter,
                          n_samples=n_samples, **kwargs)
    c1, c2 = samples[:, 0], samples[:, 1]
    r1 = ss.rankdata(c1)
    r2 = ss.rankdata(c2)
    eta1 = ss.norm.ppf(r1/(n_samples + 1))
    eta2 = ss.norm.ppf(r2/(n_samples + 1))
    r, p_val = ss.pearsonr(eta1, eta2)
    return r


def estimate_correlation_matrix(samplers, parameter, n_samples, **kwargs):
    """Estimate the correlation matrix.

    Parameters
    ----------
    samplers : dict
      A mapping from the marginals of the parameter to the corresponding sampler.
    parameter : str
      The name of the parameter in the inference model.
    n_samples : int
      The number of samples to produce.
    **kwargs
      Additional arguments passed for sampling.

    Returns
    -------
    correlation_matrix : np.ndarray
      A matrix of pairwise correlations between the univariate marginals.
    """
    dim = sum((1 for k in samplers if isinstance(k, int)))
    pairs = itertools.combinations(range(dim), 2)
    n_pairs = spc.comb(dim, 2, exact=True)
    correlations = [estimate_correlation(marginal=marginal,
                                         samplers=samplers, parameter=parameter,
                                         n_samples=n_samples, **kwargs)
                    for marginal in tqdm(pairs, desc='Estimating correlations', total=n_pairs)]
    cor = _full_cor_matrix(correlations, dim)
    return cor


def estimate_marginal_density(marginal, samplers, parameter, n_samples, **kwargs):
    """Estimate a univariate marginal probability density function.

    Parameters
    ----------
    marginal : int
      Specifies which marginal to estimate from.
    samplers : dict
      A mapping from the marginals of the parameter to the corresponding sampler.
    parameter : str
      The name of the parameter in the inference model.
    n_samples : int
      The number of samples to produce.
    **kwargs
      Additional arguments passed for sampling.

    Returns
    -------
    marginal_distribution : EmpiricalDensity
      An empirical estimation of the specified marginal probability density function.
    """
    return EmpiricalDensity(get_samples(marginal, samplers=samplers,
                                        parameter=parameter, n_samples=n_samples, **kwargs))


def estimate_marginals(samplers, parameter, n_samples, **kwargs):
    """Estimate all the univariate marginal probability density functions.

    Parameters
    ----------
    samplers : dict
      A mapping from the marginals of the parameter to the corresponding sampler.
    parameter : str
      The name of the parameter in the inference model.
    n_samples : int
      The number of samples to produce.
    **kwargs
      Additional arguments passed for sampling.

    Returns
    -------
    marginals : list[EmpiricalDensity]
      A list of estimated marginal probability density functions.
    """
    univariate = [p for p in samplers if isinstance(p, int)]
    return [EmpiricalDensity(get_samples(u, samplers=samplers, parameter=parameter,
                                         n_samples=n_samples, **kwargs))
            for u in tqdm(univariate, desc='Estimating marginal densities')]


def copula_abc(informative_summaries, summary, sampler_factory, parameter,
             n_samples=100, discrepancy_factory=None, **kwargs):
    """Perform the Copula ABC estimation.

    Parameters
    ----------
    informative_indices : dict
      A dictionary with values indicating the subsets of the summary statistic
      informative of each component of the parameter.

      For example: {0: {0}, 1: {1}}
    summary : elfi.Summary
      The summary statistic node in the inference model.
    sampler_factory
      A function which takes a discrepancy node as an argument
      and returns an ELFI ABC sampler (e.g. elfi.Rejection).
    parameter : str
      The name of the parameter in the inference model.
    n_samples : int
      The number of samples to produce.
    discrepancy_factory
      A function which takes an ELFI node as an argument
      and returns a discrepancy node (e.g. elfi.Distance).
    **kwargs
      Additional arguments passed for sampling.

    Returns
    -------
    posterior : MetaGaussian
      A meta-Gaussian approximation of the posterior distribution.

    References
    ----------
    Jingjing Li, David J. Nott, Yanan Fan, Scott A. Sisson (2016)
    Extending approximate Bayesian computation methods to high dimensions
    via Gaussian copula.
    https://arxiv.org/abs/1504.04093v1
    """
    summary_name = summary.name
    model = summary.model.copy()
    summary = model.get_reference(summary_name)

    full_indices = complete_informative_indices(informative_summaries)
    distances = make_distances(full_indices=full_indices, summary=summary,
                               discrepancy_factory=discrepancy_factory)
    samplers = make_samplers(distances, sampler_factory)
    marginals = estimate_marginals(samplers=samplers, parameter=parameter,
                                   n_samples=n_samples, **kwargs)
    correlation_matrix = estimate_correlation_matrix(samplers=samplers,
                                                     parameter=parameter, n_samples=n_samples, **kwargs)

    return MetaGaussian(corr=correlation_matrix, marginals=marginals)



### Visualization utilities

# TODO: This could probably be done with np.vectorize.
def tabulate(funs, *args):
    """Compute a function on the cartesian product of the arguments.

    Parameters
    ----------
    fun
      function to compute
    *args : array_like
      points along each axis

    Returns
    -------
    (grid, result)
      A meshgrid constructed from the given points and
      the results of the function evaluations.

    Examples
    --------
    >>> arr = np.arange(1, 4)
    >>> grid, res = tabulate(lambda x: x[0] + x[1], arr, arr)
    >>> res
    array([[2, 3, 4],
           [3, 4, 5],
           [4, 5, 6]])
    """
    if isinstance(funs, list):
        return _tabulate_list(funs, *args)
    else:
        return _tabulate1(funs, *args)


def _tabulate_list(funs, *args):
    """Compute functions on the cartesian product of the arguments.

    Same as `_tabulate1`, but for multiple functions.

    Parameters
    ----------
    funs
      a list of functions to evaluate
    *args: array_like
      points along each axis

    Returns
    -------
    (grid, [results])
      A meshgrid constructed from the given points and
      a list of results corresponding to each function.

    """
    grid = np.meshgrid(*args)
    stack = np.stack(grid, axis=0)
    if len(args) == 1:
        return grid[0], [np.squeeze(np.apply_along_axis(fun, 0, stack))
                         for fun in funs]
    else:
        return grid, [np.squeeze(np.apply_along_axis(fun, 0, stack)) for fun in funs]

def _tabulate1(fun, *args):
    grid = np.meshgrid(*args)
    stack = np.stack(grid, axis=0)
    if len(args) == 1:
        return grid[0], np.squeeze(np.apply_along_axis(fun, 0, stack))
    else:
        return grid, np.squeeze(np.apply_along_axis(fun, 0, stack))


def overlay(funs, *args):
    """Overlay plots of functions.

    Parameters
    ----------
    funs : list or dict
      The functions to plot. The plotting options for each function
      can be specified by passing a dictionary with the functions as the keys.
    *args
      The points along each axis to plot.

    Examples
    --------
    Plot with default settings:
    >>> overlay([np.sin, np.cos], np.linspace(-3, 3, 100)) # doctest: +SKIP

    Plot with custom settings:
    >>> overlay({np.sin:{}, np.cos:{'linestyle': 'dashed'}}, np.linspace(-3, 3, 100)) # doctest: +SKIP

    Plot a contour plot:
    >>> import scipy.stats as ss
    >>> overlay([ss.multivariate_normal(mean=np.zeros(2)).pdf], np.linspace(-3, 3, 50), np.linspace(-3, 3, 50)) # doctest: +SKIP
    """
    if isinstance(funs, (list, tuple)):
        funs = dict(zip(funs, [{} for f in funs]))

    grid, res = tabulate(list(funs.keys()), *args)
    fig, ax = plt.subplots()

    if len(args) == 1:
        for i, f in enumerate(funs):
            ax.plot(grid, res[i], **funs.get(f, None))
    elif len(args) == 2:
        for i, f in enumerate(funs):
            ax.contour(*grid, res[i], **funs.get(f, None))
    else:
        raise ValueError("Cannot plot in {} dimensions.".format(len(args)))
