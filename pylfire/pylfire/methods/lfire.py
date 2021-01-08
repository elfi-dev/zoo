import logging
import pickle
import os
import json
import warnings
import copy

from multiprocessing import cpu_count
from collections import OrderedDict

import numpy as np

from pylfire.classifiers.classifier import Classifier, LogisticRegression

from elfi.client import set_client
from elfi.methods.parameter_inference import ParameterInference
from elfi.methods.results import ParameterInferenceResult
from elfi.methods.utils import ModelPrior
from elfi.model.elfi_model import Summary
from elfi.visualization import visualization as viz

logger = logging.getLogger(__name__)


class LFIRE(ParameterInference):
    """Likelihood-Free Inference by Ratio Estimation (LFIRE).

    For a describtion of the LFIRE, see e.g. Thomas et al. 2018.

    References
    ----------
    O. Thomas, R. Dutta, J. Corander, S. Kaski, and M. U. Gutmann,
    Likelihood-Free Inference by Ratio Estimation, arXiv preprint arXiv:1611.10242, 2018.

    """

    def __init__(self, model, params_grid, marginal=None, classifier=None,
                 output_names=None, seed_marginal=None, precomputed_models=None, 
                 **kwargs):
        """Initializes LFIRE.

        Parameters
        ----------
        model: ElfiModel
            The elfi graph used by the algorithm.
        params_grid: np.ndarray
            A grid over which posterior values are evaluated.
        marginal: np.ndarray, optional
            Marginal data.
        classifier: str, optional
            Classifier to be used. Default LogisticRegression.
        output_names: list, optional
            Names of the nodes whose outputs are included in the batches.
        batch_size: int, optional
            A size of training data.
        seed_marginal: int, optional
            Seed for marginal data generation.
        precomputed_models: str, optional
            Precomputed classifier parameters file.
        kwargs:
            See InferenceMethod.

        """
        super(LFIRE, self).__init__(model, output_names, **kwargs)

        # 1. parse model:
        self.summary_names = self._get_summary_names()
        if len(self.summary_names) == 0:
            raise NotImplementedError('Your model must have at least one Summary node.')
        self.joint_prior = ModelPrior(self.model)

        # 2. LFIRE setup:
        self.params_grid = self._resolve_params_grid(params_grid)
        self.classifier = self._resolve_classifier(classifier)
        self._resolve_elfi_client(self.classifier.parallel_cv)
        n_batches = self.params_grid.shape[0]

       # 3. initialise results containers:
        self.state['posterior'] = np.empty(n_batches)
        self.state['infinity'] = {parameter_name: [] for parameter_name in self.parameter_names}
        # separate precomputed model container
        self.pre={'model': [], 'prior_value': np.array([])}

        # 4. initialise or load approximate posterior model:
        if precomputed_models is None:
            self.marginal = self._resolve_marginal(marginal, seed_marginal)
            for parameter_name in self.parameter_names:
                self.state[parameter_name] = np.empty(n_batches)
            for cls_parameter in self.classifier.parameter_names:
                    self.state[cls_parameter] = []
        else:
            self.load_models(precomputed_models)


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
            outputs=copy.deepcopy(self.state),
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
        likelihood = np.column_stack(likelihood)

        # Create training data
        X = np.vstack((likelihood, self.marginal))
        y = np.concatenate((np.ones(likelihood.shape[0]), -1 * np.ones(self.marginal.shape[0])))

        # Classification
        self.classifier.fit(X, y)

        # Likelihood value
        log_likelihood_value = self.classifier.predict_log_likelihood_ratio(self.observed)
        likelihood_value = np.exp(log_likelihood_value)

        # Joint prior value
        parameter_values = [batch[parameter_name] for parameter_name in self.parameter_names]
        joint_prior_value = self.joint_prior.pdf(parameter_values)

        # Posterior value
        posterior_value = joint_prior_value * likelihood_value

        # Check if posterior value is non-finite
        if np.isinf(posterior_value):
            params = self.params_grid[batch_index]
            warnings.warn(f'Posterior value is not finite for parameters \
                          {self.parameter_names} = {params} and thus will be replaced with zero!',
                          RuntimeWarning)
            posterior_value = 0
            for i, parameter_name in enumerate(self.parameter_names):
                self.state['infinity'][parameter_name] += [params[i]]

        # Update state dictionary
        self.state['posterior'][batch_index] = posterior_value
        for parameter_name in self.parameter_names:
            self.state[parameter_name][batch_index] = batch[parameter_name]
        cls_params = self.classifier.attributes['parameters']
        for cls_parameter in self.classifier.parameter_names:
            self.state[cls_parameter].append(cls_params[cls_parameter])

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

    def infer(self, *args, observed=None, **kwargs):
        """Set the objective and start the iterate loop until the inference is finished.
        See the other arguments from the `set_objective` method.

        Parameters
        ----------
        observed: dict, optional
            Observed data with node names as keys.
        bar : bool, optional
            Flag to remove (False) or keep (True) the progress bar from/in output.

        Returns
        -------
        LFIREPosterior

        """
        # 1. extract observed sum stats
        if observed is not None:
            self.model.observed = observed
        self.observed = self._get_observed_summary_values()

        # 2. evaluate posterior
        if self.state['n_batches'] == 0:
            post=super(LFIRE, self).infer(*args, **kwargs)
            self._prepare_posterior_evaluation()
        else:
            post=self._evaluate_posterior()
        return post

    def save_models(self, filename):
        """Save parameter grid and classifier parameters.

        Parameters
        ----------
        filename: str

        """
        p={}
        for parameter_name in self.parameter_names:
            p[parameter_name] = self.state[parameter_name]
        for cls_parameter in self.classifier.parameter_names:
            p[cls_parameter] = self.state[cls_parameter]
        np.savez(filename,**p)

    def load_models(self, filename):
        """Load parameter grid and classifier parameters.

        Parameters
        ----------
        filename: str

        """
        # 1. load saved data:
        p = np.load(filename)
        for variable in p.files:
            self.state[variable] = p[variable]
        # 2. check that parameter values match expectation:
        for index, parameter_name in enumerate(self.parameter_names):
            if parameter_name not in self.state:
                raise KeyError('Model parameter {} '.format(parameter_name)
                               + 'not found in saved data')
            if np.all(self.params_grid[:,index] != self.state[parameter_name]):
                raise ValueError('Parameter values in saved data do not match '
                                 + 'the input parameter grid.')
        # 3. check classifier parameters:
        for cls_parameter in self.classifier.parameter_names:
            if cls_parameter not in self.state:
                raise KeyError('Classifier parameter {} '.format(cls_parameter)
                               + 'not found in saved data.')
        # 4. make posterior model:
        self.state['n_batches'] = self.params_grid.shape[0]
        self._prepare_posterior_evaluation()

    def _prepare_posterior_evaluation(self):
        """Precompute prior probabilities and initialise classifiers."""
        # compute prior probabilities
        self.pre['prior_value'] = self.joint_prior.pdf(self.params_grid)
        # initialise classifiers
        self.pre['model'] = []
        for n in range(self.state['n_batches']):
            params={param: self.state[param][n] for param in self.classifier.parameter_names}
            model = self.classifier.load_model(params)
            self.pre['model'].append(model)

    def _evaluate_posterior(self):
        """Evaluates posterior probabilities.

        Returns
        -------
        LFIREPosterior

        """
        # TODO: add option to calculate prior probabilities and set up classifiers on the loop here
        for ii in range(self.state['n_batches']):
            # load precomputed model
            model = self.pre['model'][ii]
            # evaluate likelihood ratio
            ratio = self.classifier.predict_likelihood_ratio(self.observed, model = model)
            # calculate posterior value
            self.state['posterior'][ii] = self.pre['prior_value'][ii] * ratio

        return self.extract_result()

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

    def _resolve_marginal(self, marginal, seed_marginal=None):
        """Resolves marginal data.

        Parameters
        ----------
        marginal: np.ndarray
        seed_marginal: int, optional

        Returns
        -------
        np.ndarray

        """
        if marginal is None:
            marginal = self._generate_marginal(seed_marginal)
            x, y = marginal.shape
            logger.info(f'New marginal data ({x} x {y}) are generated.')
            return marginal
        elif isinstance(marginal, np.ndarray) and len(marginal.shape) == 2:
            return marginal
        else:
            raise TypeError('marginal must be 2d numpy array.')

    def _generate_marginal(self, seed_marginal=None):
        """Generates marginal data.

        Parameters
        ----------
        seed_marginal: int, optional

        Returns
        -------
        np.ndarray

        """
        if seed_marginal is None:
            batch = self.model.generate(self.batch_size)
        else:
            batch = self.model.generate(self.batch_size, seed=seed_marginal)
        marginal = [batch[summary_name] for summary_name in self.summary_names]
        marginal = np.column_stack(marginal)
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
        observed_ss = np.column_stack(observed_ss)
        return observed_ss


    def _resolve_classifier(self, classifier):
        """Resolves classifier."""
        if classifier is None:
            return LogisticRegression()
        elif isinstance(classifier, Classifier):
            return classifier
        else:
            raise ValueError('classifier must be an instance of Classifier.')

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
        pos_means = np.sum(vals, axis=0) / np.sum(self._posterior)
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

    @property
    def infinity(self):
        """Returns True if any posterior value is non-finite.

        Returns
        -------
        bool

        """
        s = np.sum([len(v) for v in self.outputs['infinity'].values()])
        return True if s > 0 else False

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

    def save(self, fname, path=None):
        """Saves inference results in json or pickle file formats.

        Parameters
        ----------
        fname: str
            A filename that will be saved. The type is inferred from
            extension ('json', 'pkl' or 'p').
        path: str, optional
            An absolute path to the folder, where inference results will be saved.
            Default is the current working directory

        """
        kind = os.path.splitext(fname)[1][1:]
        if kind not in ('p', 'pkl', 'json'):
            raise OSError("Wrong file type format. Please use 'json', 'pkl' or 'p'.")

        if path is None:
            # get the absolute path
            path = os.path.abspath(os.getcwd())

        self._initialize_directory(path)
        if kind in ('p', 'pkl'):
            with open(path + '/' + fname, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        elif kind == 'json':
            with open(path + '/' + fname, 'w') as f:
                json.dump(self._parse_summary_dict(), f)

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

    def _parse_summary_dict(self):
        """Parses the summary dictionary for saving.

        Returns
        -------
        str

        """
        return OrderedDict([('n_sim', self.n_sim),
                            ('map_estimates', self.map_estimates),
                            ('posterior_means', self.posterior_means)])

    def _initialize_directory(self, path):
        """Creates a given directory if not exists.

        Parameters
        ----------
        path: str

        """
        try:
            os.mkdir(path)
        except OSError:
            pass
