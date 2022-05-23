from sklearnEASE.wrapper import Wrapper
import pandas as pd
import numpy as np

class EASE(Wrapper):
    def __init__(self, algorithm = 'ease', model = 'collective', add_bias=False):
        """Constructor""" # Initialize wrapper
        Wrapper.__init__(self,algorithm=algorithm, model=model, add_bias=add_bias)

    def _fit(self, train_data, X_side, l2, l2_side, alpha, normalize_model):
        model = self.compute_model(X=train_data, side=X_side, l2=l2, l2_side=l2_side, alpha=alpha, normalize_model=normalize_model, algorithm=self.algorithm, model=self.model)
        return model

    @staticmethod
    def compute_EASE(X, l2 = 5e2):
        ''' Compute a closed-form OLS SLIM-like item-based model. (H. Steck @ WWW 2019) '''
        G = X.T @ X + l2 * np.identity((X.shape[1]))
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        B[np.diag_indices(B.shape[0])] = .0
        return B

    @staticmethod
    def compute_EDLAE(X, l2 = 5e2):
        ''' Compute a closed-form OLS SLIM-like item-based model. (H. Steck @ WWW 2019) '''
        xtx = X.T @ X
        Alpha = np.diag(np.diag(xtx))
        G = xtx + l2 * Alpha
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        B[np.diag_indices(B.shape[0])] = .0
        return B

    def compute_algorithm(self, X, l2=5e2, algorithm='ease', normalize_model=False):

        if algorithm == 'ease':
            model = self.compute_EASE(X.values, l2=l2)
        elif algorithm == 'edlae':
            model = self.compute_EDLAE(X.values, l2=l2)

        model = pd.DataFrame(model, index=X.columns, columns=X.columns)

        if normalize_model:
            observed_values_std = X.stack().std()
            predicted_values_std = (X @ model).stack().std()
            model = model*observed_values_std/predicted_values_std
        return model

    def compute_model(self, X, side=None, l2 = 5e2, l2_side = 5e2, alpha=1, normalize_model=False, algorithm='ease', model='collective'):

        if model == 'collective':
            Z = pd.concat([X, alpha*side])
            computation = self.compute_algorithm(Z, l2 = l2, algorithm=algorithm, normalize_model=normalize_model)

        elif model == 'additive':
            B = self.compute_algorithm(X, l2 = l2, algorithm=algorithm, normalize_model=normalize_model)
            B_side = self.compute_algorithm(side, l2 = l2_side, algorithm=algorithm, normalize_model=normalize_model)
            computation = alpha*B + (1 - alpha)*B_side

        elif model == 'noside':
            computation = self.compute_algorithm(X, l2 = l2, algorithm=algorithm, normalize_model=normalize_model)

        return computation
