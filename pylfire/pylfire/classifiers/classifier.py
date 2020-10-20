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

    def fit(self, X, y):
        """Fits a selected classification model.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.
        y: np.ndarray (n_samples, )
            Target values, must be binary.

        """
        raise NotImplementedError

    def predict_log_likelihood_ratio(self, X):
        """Predicts a log-likelihood ratio.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.

        Returns
        -------
        np.ndarray

        """
        raise NotImplementedError

    def predict_likelihood_ratio(self, X):
        """Predicts a likelihood ratio.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.

        Returns
        -------
        np.ndarray

        """
        return np.exp(self.predict_log_likelihood_ratio(X))

    @property
    def attributes(self):
        """Returns attributes dictionary."""
        raise NotImplementedError

    @property
    def parallel_cv(self):
        """Returns does classifier run cross-validation in parallel."""
        return False


class LogisticRegression(Classifier):
    """Logistic regression classifier for ratio estimation."""

    def __init__(self, config=None, parallel_cv=True, class_min=0):
        """Initializes logistic regression classifier."""
        self.config = self._resolve_config(config, parallel_cv)
        self.model = LogitNet(**self.config)
        self.class_min = class_min

    def fit(self, X, y):
        """Fits logistic regression classifier."""
        self._update_data_attributes(X, y)
        self.model.fit(X, y)

    def predict_log_likelihood_ratio(self, X):
        """Predicts the log-likelihood ratio."""
        class_probs = np.maximum(self.model.predict_proba(X)[:, 1], self.class_min)
        return np.log(class_probs / (1 - class_probs))

    @property
    def attributes(self):
        """Returns attributes dictionary."""
        return {
            'parameters': {
                'lambda': self.model.lambda_best_.tolist(),
                'intercept': [self.model.intercept_],
                'coefficients': self.model.coef_.ravel().tolist()
            }
        }

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

    def _update_data_attributes(self, X, y):
        """Updates the data attributes."""
        self.X, self.y = X, y


class GPClassifier(Classifier):
    """Gaussian process classifier for ratio estimation."""

    def __init__(self, kernel=None, mean_function=None, class_min=0):
        """Initializes the Gaussian process classifier."""
        self.model = None
        self.kernel = kernel
        self.mean_function = mean_function
        self.class_min = class_min

    def fit(self, X, y):
        """Fits the Gaussian process classifier."""
        self.model = self._initialize_model(X, y)
        self.model.optimize()

    def predict_log_likelihood_ratio(self, X):
        """Predicts the log-likelihood ratio."""
        class_probs = np.maximum(self.model.predict(X)[0], self.class_min)
        return np.log(class_probs / (1 - class_probs))

    @property
    def attributes(self):
        """Returns attributes dictionary."""
        return {
            'X': self.model.X,
            'Y': self.model.Y,
            'parameters': self.model.param_array.tolist()
        }

    def _get_default_kernel(self, input_dim):
        """Returns the default kernel."""
        return RBF(input_dim, ARD=True)

    def _initialize_model(self, X, y):
        """Initializes the Gaussian process classifier."""
        kernel = self.kernel.copy() if self.kernel else self._get_default_kernel(X.shape[1])
        mean_function = self.mean_function.copy() if self.mean_function else self.mean_function
        return GPClassification(X, y.reshape(-1, 1), kernel=kernel, mean_function=mean_function)
