import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smurff
import os
import seaborn as sns
from sklearn.base import BaseEstimator
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse import coo_matrix
import copy

from sklearnBPMF.data.utils import verify_ndarray, verify_pdframe, add_bias
from sklearnBPMF.core.metrics import reciprocal_rank, average_precision, average_recall, discounted_gain, normalized_gain, score_completion

class Wrapper(BaseEstimator):
    """Scikit learn wrapper for matrix completion models."""

    def __init__(self,algorithm,model,add_bias,k_metrics=True,k=20,
                 l2=5e2,l2_side=5e2,alpha=1,normalize_model=False):

        self.algorithm = algorithm
        self.model = model
        self.add_bias = add_bias
        self.k_metrics = k_metrics
        self.k = k
        self.l2 = l2
        self.l2_side = l2_side
        self.alpha = alpha
        self.normalize_model = normalize_model

    def fit(self,X_train,X_side=None,X_test=None,y=None,M=None):

        #Store training and evaluation data
        self.X_train = X_train
        self.X_test = X_test
        self.y = y
        self.M = M

        # Produce masks
        self.S_train = verify_pdframe((X_train != 0)*1)
        self.S_test = verify_pdframe((X_test != 0)*1)

        X,side = self.format_data(X_train,y=y,side=X_side)

        self.B = self._fit(X, side, self.l2, self.l2_side, self.alpha, self.normalize_model)

        # Store performance metrics
        self.store_metrics(self.X_train)

    def transform(self, X_test):

        X_test = verify_ndarray(X_test)

        if self.add_bias:
            X_test = add_bias(X_test)

            X_pred = (X_test @ self.B)[:,1:]
        else:
            X_pred = X_test @ self.B

        Xhat = X_pred + X_pred.T

        return pd.DataFrame(Xhat)

    def format_data(self,X,y=None,side=None):

        if y is None:
            y = copy.copy(X)
            self.y = y

        X,y,side = verify_ndarray(X,y,side)

        # Format bias
        if self.add_bias:
            X = add_bias(X)
            if isinstance(side,np.ndarray):
                side = add_bias(side)

        return X,side

    def predict(self,S='test'):

        if S == 'test':
            S = self.S_test
        elif S == 'train':
            S = self.S_train

        pred = (self.Xhat * S.replace(0,np.nan).values).stack()
        avg = list(pred.values.astype(np.float32))
        coord = list(pred.index.values)

        return avg, coord

    def store_metrics(self,X):

        self.Xhat = self.transform(X)

        test_scores = score_completion(self.M,self.Xhat,self.S_test,'test',k_metrics=self.k_metrics,k=self.k)
        train_scores = score_completion(self.M,self.Xhat,self.S_train,'train',k_metrics=self.k_metrics,k=self.k)
        cond_number = np.linalg.cond(verify_ndarray(X) + np.eye(X.shape[0]))
        cond_number_mask = np.linalg.cond(self.S_train + np.eye(X.shape[0]))

        pred_avg, pred_coord = self.predict(S='test')
        train_avg, train_coord = self.predict(S='train')

        #Assign current training metrics
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.train_dict = dict(pred_avg = pred_avg,
                               pred_coord = pred_coord,
                               train_avg = train_avg,
                               train_coord = train_coord,
                               cond_number = cond_number,
                               cond_number_mask = cond_number_mask,
                               **test_scores,
                               **train_scores)

    def score(self, X, Xhat, S_test, name):
        ''''Produce training, testing, and validation scoring metrics (rmse, corr(pearson,), frobenius, relative error)'''

        if self.add_bias:
            Xhat = Xhat.iloc[:-1,:-1]

        bool_mask = S_test == 1
        X_true = X * S_test

        predicted = Xhat[bool_mask].stack()
        measured = X[bool_mask].stack()

        error = predicted - measured

        frob = np.linalg.norm(Xhat - X, 'fro')
        rel_frob = frob/(np.linalg.norm(X, 'fro'))
        rmse = np.sqrt(np.mean((error**2)))
        rel_rmse = rmse/(np.sqrt(np.mean((measured**2))))
        spearman = predicted.corr(measured, method='spearman')
        pearson = predicted.corr(measured, method='pearson')

        reciprocal_r = reciprocal_rank(Xhat, X_true, 50)[0]
        mean_precision = average_precision(Xhat, X_true, 50)[0]
        mean_recall = average_recall(Xhat, X_true, 50)[0]
        ndcg = normalized_gain(Xhat, X_true, 50)[0]

        return {'{}_frob'.format(name):frob, '{}_rel_frob'.format(name):rel_frob, '{}_rmse'.format(name):rmse,
        '{}_rel_rmse'.format(name):rel_rmse, '{}_spearman'.format(name):spearman, '{}_pearson'.format(name):pearson,
        '{}_reciprocal_r'.format(name):reciprocal_r, '{}_mean_precision'.format(name):mean_precision,
        '{}_mean_recall'.format(name):mean_recall, '{}_ndcg'.format(name):ndcg}


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
