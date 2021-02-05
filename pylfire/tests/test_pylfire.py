import pytest  # noqa: F401; pylint: disable=unused-variable

import numpy as np
import tempfile

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

def test_precomputed_models():
    """Tests consistency with precomputed models. """
    arch_model = arch.get_model()
    params_grid = _create_grid(3)

    # 1: run LFIRE and save models

    lfire_method = pylfire.LFIRE(arch_model, params_grid, batch_size=10)
    lfire_results = lfire_method.infer()

    temp_file = tempfile.TemporaryFile()
    lfire_method.save_models(temp_file)
    temp_file.seek(0)  # simulate closing and opening the file

    # 2: run LFIRE with the saved models

    new_lfire_method = pylfire.LFIRE(arch_model, params_grid, batch_size=10,
                                     precomputed_models = temp_file)
    # check that precomputed models were loaded
    assert new_lfire_method.state['n_batches'] == params_grid.shape[0]
    new_results = new_lfire_method.infer()
    # check that results are consistent
    assert np.all(np.isclose(new_results.posterior_means_array, lfire_results.posterior_means_array))
    assert np.all(np.isclose(new_results.map_estimates_array, lfire_results.map_estimates_array))

    temp_file.close()
