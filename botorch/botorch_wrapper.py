"""This module contains wrappers for using BoTorch in ELFI."""

import copy

import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import GP
from gpytorch.settings import fast_pred_var

from elfi.methods.bo.acquisition import AcquisitionBase
from elfi.methods.bo.gpy_regression import GPyRegression


class BoTorchModel(GPyRegression):

    def __init__(self,
                 parameter_names,
                 bounds,
                 model_constructor=None,
                 model_options=None,
                 model_optimizer=None,
                 negate=False,
                 use_fast_pred_var=True,
                 seed=None):
        """Initialize BoTorch model wrapper.

        Parameters
        ----------
        parameter_names : List[str]
            Input parameter names.
        bounds : Dict[str, Sequence[float, float]].
            Lower and upper bound for each input parameter.
        model_constructor : callable, optional
            Function that creates a model instance.
        model_options : Dict[str, Any], optional
            Model constructor parameters.
        model_optimizer : callable, optional
            Function that optimizes model instance.
        negate : bool, optional
            If True, negate target values.
        use_fast_pred_var : bool, optional
            If True, use fast predictive variance computation.
        seed : int, optional

        """
        self.parameter_names = parameter_names
        self.input_dim = len(self.parameter_names)
        self.bounds = [bounds[param] for param in parameter_names]
        self.model_constructor = model_constructor or self._make_model
        self.model_options = model_options or {}
        self.model_optimizer = model_optimizer or self._optimize_model
        self.sign = 1 if not negate else -1
        self.use_fast_pred_var = use_fast_pred_var

        self.train_x = []
        self.train_y = []
        self._gp = None

    def predict(self, x, noiseless=False):
        """Return the model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate

        Returns
        -------
        tuple
            model (mean, var) at x where
                mean : np.array
                    with shape (x.shape[0], 1)
                var : np.array
                    with shape (x.shape[0], 1)

        """
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)

        if self._gp is None:
            return (np.zeros(x.shape[0], 1), np.ones(x.shape[0], 1))

        # activate evaluation mode
        self._gp.eval()
        self._gp.likelihood.eval()

        with torch.no_grad(), fast_pred_var(self.use_fast_pred_var):
            pred = self._gp.posterior(x, observation_noise=not(noiseless))

        m = self.sign * pred.mean.detach().numpy().reshape(-1, 1)
        v = pred.variance.detach().numpy().reshape(-1, 1)
        return m, v

    def predict_mean(self, x):
        """Return the model mean at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate

        Returns
        -------
        np.array
            with shape (x.shape[0], 1)

        """
        return self.predict(x, noiseless=True)[0]

    def predictive_gradients(self, x):
        """Return the gradients of the model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate

        Returns
        -------
        tuple
            model (grad_mean, grad_var) at x where
                grad_mean : np.array
                    with shape (x.shape[0], input_dim)
                grad_var : np.array
                    with shape (x.shape[0], input_dim)

        """
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
        x.requires_grad = True

        if self._gp is None:
            return (np.zeros(x.shape[0], self.input_dim), np.zeros(x.shape[0], self.input_dim))

        # activate evaluation mode
        self._gp.eval()

        with fast_pred_var(self.use_fast_pred_var):
            post = self._gp.posterior(x)
            dmdx = torch.autograd.grad(post.mean.sum(), x, retain_graph=True)[0]
            dvdx = torch.autograd.grad(post.variance.sum(), x)[0]

        dmdx = self.sign * dmdx.numpy().reshape(-1, self.input_dim)
        dvdx = dvdx.numpy().reshape(-1, self.input_dim)
        return dmdx, dvdx

    def predictive_gradient_mean(self, x):
        """Return the gradient of the model mean at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate

        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)

        """
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
        x.requires_grad = True

        if self._gp is None:
            return np.zeros(x.shape[0], self.input_dim)

        # activate evaluation mode
        self._gp.eval()

        with fast_pred_var(self.use_fast_pred_var):
            post = self._gp.posterior(x)
            dmdx = torch.autograd.grad(post.mean.sum(), x)[0]

        return self.sign * dmdx.numpy().reshape(-1, self.input_dim)

    def update(self, x, y, optimize=True):
        """Update model with new evidence.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize model fit.

        """
        y = self.sign * y
        self.train_x.append(x)
        self.train_y.append(y)
        xt = torch.tensor(np.array(self.train_x), dtype=torch.double).reshape(-1, self.input_dim)
        yt = torch.tensor(np.array(self.train_y), dtype=torch.double).reshape(-1, 1)

        if self._gp is None:
            # initialise
            self._gp = self.model_constructor(xt, yt, self.model_options)
        else:
            # reconstruct with new data
            state_dict = self._gp.state_dict()
            self._gp = self.model_constructor(xt, yt, self.model_options, state_dict=state_dict)

        if optimize:
            self.model_optimizer(self._gp)

    def optimize(self):
        """Optimize model hyperparameters."""
        if self._gp is None:
            raise RuntimeError('Model has not been initialised.')
        self.model_optimizer(self._gp)

    def _make_model(self, x, y, options, state_dict=None):
        model = SingleTaskGP(x, y, **options)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    def _optimize_model(self, model):
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

    @property
    def n_evidence(self):
        """Return the number of observed samples."""
        return np.array(self.train_y).size

    @property
    def X(self):
        """Return input evidence."""
        return np.array(self.train_x).reshape(-1, self.input_dim)

    @property
    def Y(self):
        """Return output evidence."""
        return self.sign * np.array(self.train_y).reshape(-1, 1)

    @property
    def noise(self):
        """Return the noise."""
        if self._gp is None:
            return None
        else:
            return self._gp.likelihood.noise.detach().numpy()

    @property
    def instance(self):
        """Return the gp instance."""
        return self._gp

    def copy(self):
        """Return a copy of current instance."""
        return copy.deepcopy(self)


class BoTorchAcquisition(AcquisitionBase):

    def __init__(self,
                 model,
                 acq_class,
                 acq_options,
                 optim_params=None
                 ):
        """Initialize BoTorch acquisition method.

        Parameters
        ----------
        model : BoTorchModel
            Gaussian process regression model.
        acq_class : Type[botorch.acquisition.AcquisitionFunction]
            Acquisition function type.
        acq_options : Dict[str, Any]
            acq_class constructor parameters.
        optim_params : Dict[str, Any], optional
            Acquisition function optimisation parameters.

        """
        self.model = model
        self.input_dim = self.model.input_dim
        self.bounds = torch.tensor(np.transpose(self.model.bounds), dtype=torch.double)

        self.acq_class = acq_class
        self.acq_options = acq_options
        self.optim_params = optim_params or {}

        if not 'num_restarts' in self.optim_params:
            self.optim_params['num_restarts'] = 10

        if not 'raw_samples' in self.optim_params:
            self.optim_params['raw_samples'] = 50 * self.optim_params['num_restarts']

        self.callable_options = {}
        for option in self.acq_options:
            if callable(self.acq_options[option]):
                self.callable_options[option] = self.acq_options[option]

    def evaluate(self, x, t=None):
        """Evaluate the acquisition function value at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
        t : int
            current acquisition index (unused)

        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)

        """
        if self.model.instance is None:
            return np.zeros((x.shape[0], 1))

        x = torch.tensor(x, dtype=torch.double).reshape(-1, 1, self.input_dim)
        return self.acq_function(x).detach().numpy()

    def acquire(self, n, t=None):
        """Return the next batch of acquisition points.

        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int
            Current acquisition index (unused).

        Returns
        -------
        np.array
            with shape (n, input_dim)

        """
        if self.model.instance is None:
            raise RuntimeError('Model has not been initialised.')

        x, _ = optimize_acqf(self.acq_function, bounds=self.bounds, q=n, **self.optim_params)
        return x.numpy()

    @property
    def acq_function(self):
        for option in self.callable_options:
            self.acq_options[option] = self.callable_options[option](self.model)
        return self.acq_class(self.model.instance, **self.acq_options)
