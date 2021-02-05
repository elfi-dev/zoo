import numpy as np
from multiprocessing import cpu_count
from glmnet import LogitNet
from GPy.models import GPClassification
from GPy.kern import RBF


class Classifier:
    """A base class for a ratio estimation classifier."""

    def __init__(self):
        """Initializes a given classifier."""
        raise NotImplementedError


    def fit(self, X, y, index = 0):
        """Fits a selected classification model.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.
        y: np.ndarray (n_samples, )
            Target values, must be binary.
        index: int, optional
            Model index, used to store the fitted model.

        """
        raise NotImplementedError

    def get(self, index = 0):
        """Returns stored model parameters.

        Parameters
        ----------
        index : int
            Model index.

        Returns
        -------
        dict

        """
        raise NotImplementedError

    def set(self, params_dict, index = 0):
        """Loads model.

        Parameters
        ----------
        params_dict : dict
            Model parameters.
        index : int, optional
            Model index, used to store the model.
        """
        raise NotImplementedError

    def predict_log_likelihood_ratio(self, X, index = 0):
        """Predicts a log-likelihood ratio.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.
        index : int, optional
            Model index, indicates which model is used in prediction.

        Returns
        -------
        np.ndarray

        """
        raise NotImplementedError

    def predict_likelihood_ratio(self, X, index = 0):
        """Predicts a likelihood ratio.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.
        index : int, optional
            Model index, indicates which model is used in prediction.

        Returns
        -------
        np.ndarray

        """
        return np.exp(self.predict_log_likelihood_ratio(X, index=index))

    @property
    def parallel_cv(self):
        """Returns does classifier run cross-validation in parallel."""
        return False


class LogisticRegression(Classifier):
    """Logistic regression classifier for ratio estimation."""

    def __init__(self, config=None, parallel_cv=True, class_min=0.005):
        """Initializes logistic regression classifier."""
        self.config = self._resolve_config(config, parallel_cv)
        self.model = LogitNet(**self.config)
        self.class_min = class_min
        self.parameter_names=['lambda', 'intercept', 'coef']
        self.store = {}

    def fit(self, X, y, index = 0):
        """Fits logistic regression classifier."""
        self.model.fit(X, y)
        # selected lambda:
        lambda_best = np.atleast_1d(self.model.lambda_best_)
        # fitted linear model parameters:
        p_0 = np.atleast_1d(self.model.intercept_)
        p_1 = np.squeeze(self.model.coef_)
        # store as array:
        self.store[index] = np.concatenate((lambda_best, p_0, p_1))

    def get(self, index = 0):
        """Returns stored model parameters."""
        params={}
        params['lambda'] = self.store[index][0]
        params['intercept'] = self.store[index][1]
        params['coef'] = self.store[index][2:]
        return params

    def set(self, params_dict, index = 0):
        """Loads model."""
        params = [np.atleast_1d(params_dict[param]) for param in self.parameter_names]
        self.store[index] = np.concatenate(params)

    def predict_log_likelihood_ratio(self, X, index = 0):
        """Predicts the log-likelihood ratio."""
        params = self.store[index][1:]
        log_ratio = params[0] + np.sum(np.multiply(params[1:], X))
        return np.maximum(log_ratio, np.log(self.class_min/(1-self.class_min)))

    @property
    def parallel_cv(self):
        """Returns does classifier run cross-validation in parallel."""
        return self.config['n_jobs'] > 1

    def _get_default_config(self, parallel_cv):
        """Returns a default config for the logistic regression."""
        return {
            'alpha': 1,
            'n_splits': 10,
            'n_jobs': cpu_count() if parallel_cv else 1,
            'cut_point': 0
        }

    def _resolve_config(self, config, parallel_cv):
        """Resolves a config for logistic regression."""
        if not isinstance(config, dict):
            config = self._get_default_config(parallel_cv)
        return config


class GPClassifier(Classifier):
    """Gaussian process classifier for ratio estimation."""

    def __init__(self, kernel=None, mean_function=None, class_min=0.005):
        """Initializes the Gaussian process classifier."""
        self.kernel = kernel
        self.mean_function = mean_function
        self.class_min = class_min
        self.parameter_names=['X','Y','param_array']
        self.store = {}

    def fit(self, X, y, index = 0):
        """Fits the Gaussian process classifier."""
        model = self._initialize_model(X, y)
        model.optimize()
        self.store[index] = model

    def get(self, index = 0):
        """Returns stored model parameters."""
        params={}
        params['X'] = self.store[index].X
        params['Y'] = self.store[index].Y
        params['param_array'] = self.store[index].param_array
        return params

    def set(self, params_dict, index = 0):
        """Loads model."""
        self.store[index] = self._load_model(params_dict)

    def predict_log_likelihood_ratio(self, X, index = 0):
        """Predicts the log-likelihood ratio."""
        model = self.store[index]
        class_probs = np.maximum(model.predict(X)[0], self.class_min)
        return np.log(class_probs / (1 - class_probs))

    def _get_default_kernel(self, input_dim):
        """Returns the default kernel."""
        return RBF(input_dim, ARD=True)

    def _initialize_model(self, X, y):
        """Initializes the Gaussian process classifier."""
        kernel = self.kernel.copy() if self.kernel else self._get_default_kernel(X.shape[1])
        mean_function = self.mean_function.copy() if self.mean_function else self.mean_function
        return GPClassification(X, y.reshape(-1, 1), kernel=kernel, mean_function=mean_function)

    def _load_model(self, params):
        """Loads the Gaussian process classifier."""
        model = self._initialize_model(params['X'], params['Y'])
        model[:] = params['param_array']
        model.parameters_changed()
        return model
