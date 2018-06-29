import scipy
import numpy as np

from elfi.model.utils import distance_as_discrepancy


from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import multivariate_normal
from scipy.stats import norm
import scipy.stats as ss
import matplotlib.pyplot as plt
import elfi
from scipy.stats import norm
# %matplotlib inline
from elfi import adjust_posterior
from elfi.methods.parameter_inference import ParameterInference, Copula_ABC

import logging
logging.basicConfig(level=logging.INFO)
from utility import *

# from Inference_COPULA_ABC_inherit_rejection import *
import elfi


#
# def dimension_wise(simulated, observed):
#     # simulated = np.column_stack(simulated)
#     # observed = np.column_stack(observed)
#     return abs(simulated - observed)

def run_copulaABC():
    np.random.seed(20180509)
    PP = 4 # dimensions
    yobs = np.zeros((1, PP))
    yobs[0, 0] = 10

    A = np.diag(np.ones(PP))
    A[0, 0] = 100
    b = 0.1
    thetas = multivariate_normal.rvs(mean=np.zeros(PP), cov=A)
    thetas[1] = thetas[1]+b*(thetas[0]**2)-100*b

    m = elfi.new_model()
    n_sample = 500
    quantiles = 0.01

    elfi.Prior(ss.multivariate_normal, np.zeros(PP), np.eye(PP), model = m, name = 'muss')
    elfi.Simulator(simulator_multivariate, m['muss'], observed=yobs, name = 'Gauss')

    elfi.Summary(identity, m['Gauss'], name = 'identity')
    elfi.Distance(dimension_wise, m['identity'], name = 'd')

    # rej = elfi.Rejection(m['d'], output_names=['identity'], batch_size=10000, seed=20180509).sample(n_sample, quantile = quantiles)
    rej = Copula_ABC(m['d'], output_names=['identity'], batch_size=10000, seed=20180509).sample(n_sample, quantile = quantiles)

    a = 1

if __name__ == '__main__':
    run_copulaABC()