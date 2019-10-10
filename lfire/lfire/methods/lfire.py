import logging

from multiprocessing import cpu_count
from collections import OrderedDict

import numpy as np

from glmnet import LogitNet

from elfi.client import set_client
from elfi.methods.parameter_inference import ParameterInference
from elfi.methods.results import ParameterInferenceResult
from elfi.methods.utils import ModelPrior
from elfi.model.elfi_model import Summary
from elfi.visualization import visualization as viz

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
        self.state['posterior'] = np.empty(n_batches)
        self.state['lambda'] = np.empty(n_batches)
        self.state['coef'] = np.empty((n_batches, len(self.summary_names)))
        self.state['intercept'] = np.empty(n_batches)
        for parameter_name in self.parameter_names:
            self.state[parameter_name] = np.empty(n_batches)

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
        self.state['lambda'][batch_index] = m.lambda_best_
        self.state['coef'][batch_index, :] = m.coef_
        self.state['intercept'][batch_index] = m.intercept_
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
            'n_jobs': cpu_count() if parallel_cv else 1,
            'cut_point': 0
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

        self._params_grid = self._get_params_grid()
        self._posterior = self._get_posterior()

    def __repr__(self):
        """Returns a summary of results as a string."""
        return self._parse_summary()

    @property
    def results(self):
        """Returns all inference results.

        Returns
        -------
        OrderedDict

        """
        return OrderedDict([(k, v) for k, v in self.outputs.items()])

    @property
    def dim(self):
        """Returns the number of parameters.

        Returns
        -------
        int

        """
        return len(self.parameter_names)

    @property
    def n_sim(self):
        """Returns the number of simulations.

        Returns
        -------
        int

        """
        return self.outputs['n_sim']

    @property
    def posterior_means(self):
        """Returns the posterior means for each parameter.

        Returns
        -------
        OrderedDict

        """
        vals = self._posterior.reshape(-1, 1) * self._params_grid
        pos_means = np.sum(vals, axis=0) / np.sum(vals)
        return OrderedDict([(n, pos_means[i]) for i, n in enumerate(self.parameter_names)])

    @property
    def posterior_means_array(self):
        """Returns the posterior means for each parameter.

        Returns
        -------
        np.ndarray

        """
        return np.array(list(self.posterior_means.values()))

    @property
    def map_estimates(self):
        """Returns the maximum a posterior estimates for each parameters.

        Returns
        -------
        OrderedDict

        """
        argmax = self._params_grid[np.argmax(self._posterior), :]
        return OrderedDict([(n, argmax[i]) for i, n in enumerate(self.parameter_names)])

    @property
    def map_estimates_array(self):
        """Returns the maximum a posterior estimates for each parameters.

        Returns
        -------
        np.ndarray

        """
        return np.array(list(self.map_estimates.values()))

    @property
    def marginals(self):
        """Returns the marginal posterior distributions for each parameter.

        Returns
        -------
        OrderedDict

        """
        pos_shape = self._get_number_of_unique_parameter_values()
        pos_vals = self._posterior.reshape(pos_shape) / np.sum(self._posterior)
        axis = np.arange(self.dim)
        return OrderedDict([(n, np.sum(pos_vals, tuple(axis[axis != i])))
                            for i, n in enumerate(self.parameter_names)])

    @property
    def marginals_array(self):
        """Returns the marginal posterior distributions for each parameter.

        Returns
        -------
        np.ndarray

        """
        return np.array(list(self.marginals.values()))

    def plot_marginals(self, selector=None, axes=None):
        """Visualizes the marginal posterior distributions for each parameter.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from marginals. Default to all.
        axes: one or an iterable of plt.Axes, optional

        Returns
        -------
        axes: np.ndarray of plt.Axes

        """
        # TODO: allow kwargs
        marginals = self.marginals
        unique_param_vals = self._get_unique_parameter_values()
        ncols = len(marginals.keys()) if len(marginals.keys()) > 5 else 5
        marginals = viz._limit_params(marginals, selector)
        shape = (max(1, len(marginals) // ncols), min(len(marginals), ncols))
        axes, _ = viz._create_axes(axes, shape)
        axes = axes.ravel()

        for idx, key in enumerate(marginals.keys()):
            axes[idx].plot(unique_param_vals[key], marginals[key])
            axes[idx].fill_between(unique_param_vals[key], marginals[key], alpha=0.1)
            axes[idx].set_xlabel(key)

        return axes

    def plot_pairs(self, selector=None, axes=None):
        """Visualizes pairwise relationships as a matrix with marginals on the diagonal.

        Parameters
        ----------
        selector: iterable of ints or strings, optional
            Indices or keys to use from marginals and posterior. Default to all.
        axes: one or an iterable of plt.Axes, optional

        Returns
        -------
        axes: np.ndarray of plt.Axes

        """
        # TODO: allow kwargs
        posterior_shape = self._get_number_of_unique_parameter_values()
        posterior = self._posterior.reshape(posterior_shape) / np.sum(self._posterior)
        marginals = self.marginals
        unique_param_vals = self._get_unique_parameter_values()
        marginals = viz._limit_params(marginals, selector)
        shape = (len(marginals), len(marginals))
        axes, _ = viz._create_axes(axes, shape)

        for idx_row, key_row in enumerate(marginals):
            for idx_col, key_col in enumerate(marginals):
                if idx_row == idx_col:
                    # plot 1d marginals
                    axes[idx_row, idx_col].plot(
                        unique_param_vals[key_row],
                        marginals[key_row]
                    )
                    axes[idx_row, idx_col].fill_between(
                        unique_param_vals[key_row],
                        marginals[key_row],
                        alpha=0.1
                    )
                else:
                    # plot 2d marginals
                    xx, yy = np.meshgrid(
                        unique_param_vals[key_col],
                        unique_param_vals[key_row],
                        indexing='ij'
                    )
                    axes[idx_row, idx_col].contourf(
                        *[xx, yy],
                        self._get_2d_marginal(idx_row, idx_col, posterior),
                        cmap='Blues'
                    )
            axes[idx_row, 0].set_ylabel(key_row)
            axes[-1, idx_row].set_xlabel(key_row)

        return axes

    def save(self):
        """Saves inference results."""
        raise NotImplementedError

    def summary(self):
        """Prints a verbose summary of contained results."""
        print(self._parse_summary())

    def map_estimates_summary(self):
        """Prints a representation of the maximum a posterior estimates."""
        print(self._parse_map_estimates_summary())

    def posterior_means_summary(self):
        """Prints a representation of the posterior means."""
        print(self._parse_posterior_means_summary())

    def _get_params_grid(self):
        """Returns the parameters grid over which the posterior distribution is calculated.

        Returns
        -------
        np.ndarray

        """
        return np.c_[tuple(self.outputs[n] for n in self.parameter_names)]

    def _get_number_of_unique_parameter_values(self):
        """Returns the number of unique parameter values for each parameter.

        Returns
        -------
        list

        """
        return [len(np.unique(self.outputs[n])) for n in self.parameter_names]

    def _get_unique_parameter_values(self):
        """Returns the unique parameter values for each paramater.

        Returns
        -------
        OrderedDict

        """
        return OrderedDict([(n, np.unique(self.outputs[n])) for n in self.parameter_names])

    def _get_posterior(self):
        """Returns the calculated posterior values.

        Returns
        -------
        np.ndarray

        """
        return self.outputs['posterior']

    def _get_2d_marginal(self, row, col, posterior):
        """Returns two dimensional posterior marginal distribution.

        Parameters
        ----------
        row: int
            Row number or index.
        col: int
            Column number or index.
        posterior: np.ndarray
            Reshaped posterior distribution.

        Returns
        -------
        marginal_2d: np.ndarray

        """
        axis = tuple(np.delete(np.arange(self.dim), [row, col]))
        if row < col:
            marginal_2d = np.sum(posterior, axis).T
        else:
            marginal_2d = np.sum(posterior, axis)
        return marginal_2d

    def _parse_summary(self):
        """Parses the summary string for printing.

        Returns
        -------
        str

        """
        desc = f'Method: {self.method_name}\n'
        if hasattr(self, 'n_sim'):
            desc += f'Number of simulations: {self.n_sim}\n'
        try:
            desc += self._parse_map_estimates_summary() + '\n'
        except TypeError:
            pass
        try:
            desc += self._parse_posterior_means_summary() + '\n'
        except TypeError:
            pass
        return desc

    def _parse_map_estimates_summary(self):
        """Parses the map estimates summary string for printing.

        Returns
        -------
        str

        """
        s = 'MAP estimates: '
        s += ', '.join([f'{k}: {v:.3g}' for k, v in self.map_estimates.items()])
        return s

    def _parse_posterior_means_summary(self):
        """Parses the posterior means summary string for printing.

        Returns
        -------
        str

        """
        s = 'Posterior means: '
        s += ', '.join([f'{k}: {v:.3g}' for k, v in self.posterior_means.items()])
        return s
