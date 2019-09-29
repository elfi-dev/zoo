import logging
from multiprocessing import cpu_count

import numpy as np

from glmnet import LogitNet

from elfi.client import set_client
from elfi.methods.parameter_inference import ParameterInference
from elfi.methods.results import ParameterInferenceResult
from elfi.methods.utils import ModelPrior
from elfi.model.elfi_model import Summary

logger = logging.getLogger(__name__)


class LFIRE(ParameterInference):
    """Likelihood-Free Inference by Ratio Estimation (LFIRE).

    For a describtion of the LFIRE, see e.g. Owen et al. 2018.

    References
    ----------
    O. Thomas, R. Dutta, J. Corander, S. Kaski, and M. U. Gutmann,
    Likelihood-Free Inference by Ratio Estimation, arXiv preprint arXiv:1611.10242, 2018.

    """

    def __init__(self, model, params_grid, marginal=None,
                 logreg_config=None, output_names=None, parallel_cv=True, **kwargs):
        """Initializes LFIRE.

        Parameters
        ----------
        model: ElfiModel
            The elfi graph used by the algorithm.
        params_grid: np.ndarray
            A grid over which posterior values are evaluated.
        marginal: np.ndarray, optional
            Marginal data.
        output_names: list, optional
            Names of the nodes whose outputs are included in the batches.
        parallel_cv: bool, optional
            Either cross-validation or elfi can be run in parallel.
        kwargs:
            See InferenceMethod.

        """
        super(LFIRE, self).__init__(model, output_names, **kwargs)

        self.summary_names = self._get_summary_names()
        if len(self.summary_names) == 0:
            raise NotImplementedError('Your model must have at least one Summary node.')

        self.params_grid = self._resolve_params_grid(params_grid)
        self.marginal = self._resolve_marginal(marginal)
        self.observed = self._get_observed_summary_values()
        self.joint_prior = ModelPrior(self.model)
        self.logreg_config = self._resolve_logreg_config(logreg_config, parallel_cv)

        self._resolve_elfi_client(parallel_cv)

        n_batches = self.params_grid.shape[0]
        self.state['posterior'] = np.zeros(n_batches)
        for parameter_name in self.parameter_names:
            self.state[parameter_name] = np.zeros(n_batches)

    def set_objective(self):
        """Sets objective for inference."""
        self.objective['n_batches'] = self.params_grid.shape[0]
        self.objective['n_sim'] = self.params_grid.shape[0] * self.batch_size

    def extract_result(self):
        """Extracts the result from the current state.

        Returns
        -------
        LFIREPosterior

        """
        return LFIREPosterior(
            method_name='LFIRE',
            outputs=self.state,
            parameter_names=self.parameter_names
        )

    def update(self, batch, batch_index):
        """Updates the inference state with a new batch and performs LFIRE.

        Parameters
        ----------
        batch: dict
        batch_index: int

        """
        # TODO: beautify this
        super(LFIRE, self).update(batch, batch_index)

        # Parse likelihood values
        likelihood = [batch[summary_name] for summary_name in self.summary_names]
        likelihood = np.array(likelihood).T

        # Create training data
        X = np.vstack((likelihood, self.marginal))
        y = np.concatenate((np.ones(likelihood.shape[0]), -1 * np.ones(self.marginal.shape[0])))

        # Logistic regression
        m = LogitNet(**self.logreg_config)
        m.fit(X, y)

        # Likelihood value
        log_likelihood_value = m.intercept_ + np.sum(np.multiply(m.coef_, self.observed))
        likelihood_value = np.exp(log_likelihood_value)

        # Joint prior value
        parameter_values = [batch[parameter_name] for parameter_name in self.parameter_names]
        joint_prior_value = self.joint_prior.pdf(parameter_values)

        # Posterior value
        posterior_value = joint_prior_value * likelihood_value

        # Update state dictionary
        self.state['posterior'][batch_index] = posterior_value
        for parameter_name in self.parameter_names:
            self.state[parameter_name][batch_index] = batch[parameter_name]

    def prepare_new_batch(self, batch_index):
        """Prepares a new batch for elfi.

        Parameters
        ----------
        batch_index: int

        Returns
        -------
        dict

        """
        params = self.params_grid[batch_index]
        names = self.parameter_names
        batch = {p: params[i] for i, p in enumerate(names)}
        return batch

    def _resolve_params_grid(self, params_grid):
        """Resolves parameters grid.

        Parameters
        ----------
        params_grid: np.ndarray

        Returns
        -------
        np.ndarray

        """
        if isinstance(params_grid, np.ndarray) and len(params_grid.shape) == 2:
            return params_grid
        else:
            raise TypeError('params_grid must be 2d numpy array.')

    def _resolve_marginal(self, marginal):
        """Resolves marginal data.

        Parameters
        ----------
        marginal: np.ndarray

        Returns
        -------
        np.ndarray

        """
        if marginal is None:
            marginal = self._generate_marginal()
            x, y = marginal.shape
            logger.info(f'New marginal data ({x} x {y}) are generated.')
            return marginal
        elif isinstance(marginal, np.ndarray) and len(marginal.shape) == 2:
            return marginal
        else:
            raise TypeError('marginal must be 2d numpy array.')

    def _generate_marginal(self):
        """Generates marginal data.

        Returns
        -------
        np.ndarray

        """
        batch = self.model.generate(self.batch_size)
        marginal = [batch[summary_name] for summary_name in self.summary_names]
        marginal = np.array(marginal).T
        return marginal

    def _get_summary_names(self):
        """Gets the names of summary statistics.

        Returns
        -------
        list

        """
        summary_names = []
        for node in self.model.nodes:
            if isinstance(self.model[node], Summary) and not node.startswith('_'):
                summary_names.append(node)
        return summary_names

    def _get_observed_summary_values(self):
        """Gets observed values for summary statistics.

        Returns
        -------
        np.ndarray

        """
        observed_ss = [self.model[summary_name].observed for summary_name in self.summary_names]
        observed_ss = np.array(observed_ss).T
        return observed_ss

    def _resolve_logreg_config(self, logreg_config, parallel_cv):
        """Resolves logistic regression config.

        Parameters
        ----------
        logreg_config: dict
            Config dictionary for logistic regression.
        parallel_cv: bool

        Returns
        -------
        dict

        """
        if isinstance(logreg_config, dict):
            # TODO: check valid kwargs
            return logreg_config
        else:
            return self._get_default_logreg_config(parallel_cv)

    def _get_default_logreg_config(self, parallel_cv):
        """Creates logistic regression config.

        Parameters
        ----------
        parallel_cv: bool

        Returns
        -------
        dict

        """
        logreg_config = {
            'alpha': 1,
            'n_splits': 10,
            'n_jobs': cpu_count() if parallel_cv else 1
        }
        return logreg_config

    def _resolve_elfi_client(self, parallel_cv):
        """Resolves elfi client. Either elfi or cross-validation can be run in parallel.

        Parameters
        ----------
        parallel_cv: bool

        """
        if parallel_cv:
            set_client('native')


class LFIREPosterior(ParameterInferenceResult):
    """Results from LFIRE inference method."""

    def __init__(self, method_name, outputs, parameter_names, **kwargs):
        """Initializes LFIREPosterior.

        Parameters
        ----------
        method_name: str
            Name of the method.
        outputs: dict
            Dictionary with outputs from the nodes, e.g. samples.
        parameter_names: list
            Names of the parameter nodes.
        kwargs:
            See ParameterInferenceResult.

        """
        super(LFIREPosterior, self).__init__(
            method_name=method_name,
            outputs=outputs,
            parameter_names=parameter_names,
            **kwargs
        )

    def results(self):
        """Returns inference results.

        Returns
        -------
        dict

        """
        return self.outputs

    def plot(self):
        """Visualizes inference results."""
        raise NotImplementedError

    def save(self):
        """Saves inference results."""
        raise NotImplementedError
