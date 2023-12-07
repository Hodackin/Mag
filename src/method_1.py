import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import \
    mean_absolute_error, \
    mean_absolute_percentage_error

from src.SGTM import GTM
from src.GRNN_ import GRNN


class SGTM_GRNN(BaseEstimator):

    def __init__(self, name="SGTM_GRNN", sigma: float = .2):
        self.name = name
        self.sigma = sigma

        self.ns_1 = None
        self.ns_2 = None
        self.grnn = None

        self.is_fitted_ = False


    def fit(self, X, y):
        """
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
               The training input samples.
            y : array-like, shape (n_samples,) or (n_samples, n_outputs)
               The target values.
            Returns
            -------
            self : object
               Returns self.
       """
        X, y = check_X_y(X, y, accept_sparse=True)
        X_y = np.concatenate((X, y[:, np.newaxis]), axis=1)

        hl_size = X.shape[1]

        self.ns_1 = GTM(numb_of_steps=hl_size)
        self.ns_1.fit(X_y, y)
        y_new = self.ns_1.predict(X_y)

        self.ns_2 = GTM(numb_of_steps=hl_size)
        self.ns_2.fit(X, y_new)

        train_global_y = self.ns_2.predict(X)
        train_local_y = y - train_global_y

        self.grnn = GRNN(sigma=self.sigma)
        self.grnn.fit(X, train_local_y)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                The training input samples.
            Returns
            -------
            y : ndarray, shape (n_samples,)
                Returns an array of predicted values.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        pred_global_y = self.ns_2.predict(X)
        pred_local_y = self.grnn.predict(X)

        return pred_global_y + pred_local_y

    def _separate_return_of_pred(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        pred_global_y = self.ns_2.predict(X)
        pred_local_y = self.grnn.predict(X)
        return pred_global_y, pred_local_y

    def score(self, X, y):
        """
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
            y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values
            Returns
            -------
            score : float
                estimated of similarity between true and pred values
                by choosen metric.
        """
        y_pred = self.predict(X)
        # return mean_squared_error(y, y_pred, squared=False)
        return mean_absolute_percentage_error(y, y_pred)
