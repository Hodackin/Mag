import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_absolute_percentage_error


class GRNN(BaseEstimator, RegressorMixin):

    def __init__(self, name="GRNN", sigma: float = 0.1, threshold=1e-7):
        self.name = name
        self.sigma = sigma
        self.threshold = threshold

        self.X_ = None
        self.Y_ = None
        self.is_fitted_ = False

    def _rbf_kernel(self, X):
        return np.exp(
            -np.linalg.norm(self.X_ - X, axis=1) ** 2 / (2 * self.sigma ** 2))

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

        self.X_ = X
        self.y_ = y

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

        def one_row_pred(row):
            K = self._rbf_kernel(row)
            ksum = K.sum()

            # if ksum < self.threshold:
            #     ksum = self.threshold
            ksum = np.nan_to_num(ksum)
            # print(ksum)
            return np.nan_to_num(np.multiply(K, self.y_).sum() / ksum)

        return np.apply_along_axis(one_row_pred, axis=1, arr=X)

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
            score: float
                estimated of similarity between true and pred values
                by choosen metric.
        """
        y_pred = self.predict(X)
        return mean_absolute_percentage_error(y, y_pred)

