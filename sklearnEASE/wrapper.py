import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smurff
import os
import seaborn as sns
from sklearn.base import BaseEstimator
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse import coo_matrix

from sklearnEASE.metrics import corr_metric, add_bias

class Wrapper(BaseEstimator):
    """Scikit learn wrapper for matrix completion models."""

    def __init__(self,algorithm,model,add_bias):

        self.algorithm = algorithm
        self.model = model
        self.add_bias = add_bias

    def fit(self, X_train, side=None, l2 = 5e2, l2_side = 5e2, alpha=1, normalize_model=False):

        if self.add_bias:
            X_train = add_bias(X_train)
            if isinstance(side,pd.DataFrame):
                side = add_bias(side)

        self.B = self._fit(X_train, side, l2, l2_side, alpha, normalize_model)
        return self

    def transform(self, X_test):

        if self.add_bias:
            X_train = add_bias(X_test)

        Xhat = X_test @ self.B

        return Xhat + Xhat.T

    def score(self, X, X_test, S_test, name):
        ''''Produce training, testing, and validation scoring metrics (rmse, corr(pearson,), frobenius, relative error)'''

        if self.add_bias:
            X = add_bias(X)
            X_test = add_bias(X_test)
            S_test = add_bias(S_test,val=0)

        bool_mask = S_test == 1
        Xhat = self.transform(X_test)

        predicted = Xhat[bool_mask].stack()
        measured = X[bool_mask].stack()

        error = predicted - measured

        frob = np.linalg.norm(Xhat - X, 'fro')
        rel_frob = frob/(np.linalg.norm(X, 'fro'))
        rmse = np.sqrt(np.mean((error**2)))
        rel_rmse = rmse/(np.sqrt(np.mean((measured**2))))
        spearman = predicted.corr(measured, method='spearman')
        pearson = predicted.corr(measured, method='pearson')

        return {'{}_frob'.format(name):frob, '{}_rel_frob'.format(name):rel_frob, '{}_rmse'.format(name):rmse,
        '{}_rel_rmse'.format(name):rel_rmse, '{}_spearman'.format(name):spearman, '{}_pearson'.format(name):pearson}


    def _makePlots(self, train_dict, X_train, X_test, saveplot=True, complete_matrix=None):

        pred_avg = np.array(train_dict['pred_avg'])
        pred_std = np.array(train_dict['pred_std'])
        test_corr = train_dict['test_corr']
        rmse_avg = train_dict['rmse_avg']
        pred_coord = train_dict['pred_coord']
        train_avg = np.array(train_dict['train_avg'])
        train_std = np.array(train_dict['train_std'])
        train_coord = train_dict['train_coord']
        shape = X_test.shape
        n_train_examples = len(X_train.data)//2
        test_ratio = len(X_test.data)/(shape[0]**2)

        if complete_matrix != None:
            M = complete_matrix.tocsc()
        else:
            M = (X_train + X_test)

        linkage_ = linkage(M.toarray(), method='ward')
        dendrogram_ = dendrogram(linkage_, no_plot=True)
        clust_index = dendrogram_['leaves']
        M_clust = M[:,clust_index][clust_index,:].toarray()

        test_sparse = to_sparse(pred_avg, pred_coord, shape)
        test_std_sparse = to_sparse(pred_std, pred_coord, shape)
        train_sparse = to_sparse(train_avg, train_coord, shape)
        train_std_sparse = to_sparse(train_std, train_coord, shape)

        pred_clust = (test_sparse + train_sparse)[:,clust_index][clust_index,:].toarray()
        std_clust = (test_std_sparse + train_std_sparse)[:,clust_index][clust_index,:].toarray()

        fig, ax = plt.subplots(3, 3, figsize=(20, 20))

        sns.heatmap(M_clust,robust=True,ax=ax[0,0], square=True,
                    yticklabels=False, xticklabels=False)
        ax[0, 0].set_title('True Matrix')

        sns.heatmap(pred_clust,robust=True,ax=ax[0,1], square=True,
                    yticklabels=False, xticklabels=False)
        ax[0, 1].set_title('Predicted Matrix')

        sns.heatmap(std_clust,robust=True,ax=ax[0,2], cmap='viridis',
                    yticklabels=False, xticklabels=False, square=True)
        ax[0, 2].set_title('Uncertainty (Stdev.)')

        ax[1, 0].scatter(X_test.data, pred_avg, edgecolors=(0, 0, 0))
        ax[1, 0].plot([X_test.data.min(), X_test.data.max()], [pred_avg.min(), pred_avg.max()], 'k--',
                      lw=4)
        ax[1, 0].set_xlabel('Measured')
        ax[1, 0].set_ylabel('Predicted')
        ax[1, 0].set_title('Measured vs Avg. Prediction')

        ax[1, 1].scatter(pred_std, pred_avg, edgecolors=(0, 0, 0))
        ax[1, 1].set_xlabel('Standard Deviation')
        ax[1, 1].set_ylabel('Predicted')
        ax[1, 1].set_title('Stdev. vs Prediction')

        x_ax = np.arange(len(X_test.data))
        sort_vals = np.argsort(X_test.data)
        ax[1, 2].plot(x_ax, X_test.data[sort_vals], linewidth=4, label="measured")
        ax[1, 2].plot(x_ax, pred_avg[sort_vals], 'rx', alpha=0.5, label='predicted')
        ax[1, 2].set_title('Sorted and overlayed measured and predicted values')
        ax[1, 2].legend()

        ax[2, 0].plot(x_ax, X_test.data[sort_vals], linewidth=4, label="measured")
        ax[2, 0].plot(x_ax, pred_std[sort_vals], 'r', label='stdev.')
        ax[2, 0].set_title('Sorted and overlayed measured stdev values')
        ax[2, 0].legend()

        ax[2, 1].plot(x_ax, X_test.data[sort_vals], label="actual")
        ax[2, 1].fill_between(x_ax, pred_avg[sort_vals] - pred_std[sort_vals],
                              pred_avg[sort_vals] + pred_std[sort_vals],
                              alpha=0.5, label="std")
        ax[2, 1].set_title('predicted stdev. relative to predicted value')
        ax[2, 1].legend()

        ax[2, 2].plot(x_ax, X_test.data[sort_vals], label="actual")
        ax[2, 2].fill_between(x_ax, X_test.data[sort_vals] - pred_std[sort_vals],
                              X_test.data[sort_vals] + pred_std[sort_vals],
                              alpha=0.5, label='std')
        ax[2, 2].set_title('predicted stdev. relative to measured value')
        ax[2, 2].legend()

        fig.tight_layout()
        fig.suptitle('{} - NSAMPLES: {} NUM_LATENT: {} SIDE_NOISE: {} NUM_TRAIN {} BURNIN: {} TEST_RATIO {:.5f}\n Metrics - Corr: {:.5f} - Test RMSE: {:.5f}'. \
                     format(self.trainSession.getSaveName().split('.')[0],
                            self.num_samples, self.num_latent, self.side_noise,
                            n_train_examples, self.burnin, test_ratio,
                            test_corr, rmse_avg))

        fig.subplots_adjust(top=0.90)
        if saveplot:
            fig.savefig('figure_log.png')
            print("Saved figure to current working directory..")

        return self
