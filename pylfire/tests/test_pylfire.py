import pytest  # noqa: F401; pylint: disable=unused-variable

import numpy as np

import pylfire
from pylfire.models import arch

from elfi.methods.parameter_inference import ParameterInference
from elfi.methods.results import ParameterInferenceResult


def _create_grid(n):
    """Creates a grid for the ARCH(1) model parameters.

    Parameters
    ----------
    n: int
        Number of points in a grid for each parameter.

    Returns
    -------
    np.ndarray (rows x columns) = (number of points in the grid x number of parameters)

    """
    t1, t2 = np.linspace(-1, 1, n), np.linspace(0, 1, n)
    tt1, tt2 = np.meshgrid(t1, t2)
    return np.c_[tt1.ravel(), tt2.ravel()]


def test_pylfire():
    """Tests LFIRE and LFIREPosterior classes."""
    arch_model = arch.get_model()

    lfire_method = pylfire.LFIRE(model=arch_model, params_grid=_create_grid(2), batch_size=10)

    # check instance
    assert isinstance(lfire_method, ParameterInference) and isinstance(lfire_method, pylfire.LFIRE)

    # run inference and check results instance
    lfire_results = lfire_method.infer()
    assert isinstance(lfire_results, ParameterInferenceResult) \
        and isinstance(lfire_results, pylfire.LFIREPosterior)
