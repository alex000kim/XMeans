"""
Implementation of XMeans algorithm based on
Pelleg, Dan, and Andrew W. Moore. "X-means: Extending K-means with Efficient Estimation of the Number of Clusters."
ICML. Vol. 1. 2000.
https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf
"""

import numpy as np
from sklearn.cluster import KMeans

EPS = np.finfo(float).eps


def loglikelihood(R, R_n, variance, M, K):
    """
    See Pelleg's and Moore's for more details.
    :param R: (int) size of cluster
    :param R_n: (int) size of cluster/subcluster
    :param variance: (float) maximum likelihood estimate of variance under spherical Gaussian assumption
    :param M: (float) number of features (dimensionality of the data)
    :param K: (float) number of clusters for which loglikelihood is calculated
    :return: (float) loglikelihood value
    """
    if 0 <= variance <= EPS:
        res = 0
    else:
        res = R_n * (np.log(R_n) - np.log(R) - 0.5 * (np.log(2 * np.pi) + M * np.log(variance) + 1)) + 0.5 * K
        if res == np.inf:
            res = 0
    return res


def get_additonal_k_split(K, X, clst_labels, clst_centers, n_features, K_sub, k_means_args):
    bic_before_split = np.zeros(K)
    bic_after_split = np.zeros(K)
    clst_n_params = n_features + 1
    add_k = 0
    for clst_index in range(K):
        clst_points = X[clst_labels == clst_index]
        clst_size = clst_points.shape[0]
        if clst_size <= K_sub:
            # skip this cluster if it is too small
            # i.e. cannot be split into more clusters
            continue
        clst_variance = np.sum((clst_points - clst_centers[clst_index]) ** 2) / float(clst_size - 1)
        bic_before_split[clst_index] = loglikelihood(clst_size, clst_size, clst_variance, n_features,
                                                     1) - clst_n_params / 2.0 * np.log(clst_size)
        kmeans_subclst = KMeans(n_clusters=K_sub, **k_means_args).fit(clst_points)
        subclst_labels = kmeans_subclst.labels_
        subclst_centers = kmeans_subclst.cluster_centers_
        log_likelihood = 0
        for subclst_index in range(K_sub):
            subclst_points = clst_points[subclst_labels == subclst_index]
            subclst_size = subclst_points.shape[0]
            if subclst_size <= K_sub:
                # skip this subclst_size if it is too small
                # i.e. won't be splittable into more clusters on the next iteration
                continue
            subclst_variance = np.sum((subclst_points - subclst_centers[subclst_index]) ** 2) / float(
                subclst_size - K_sub)
            log_likelihood = log_likelihood + loglikelihood(clst_size, subclst_size, subclst_variance, n_features,
                                                            K_sub)
        subclst_n_params = K_sub * clst_n_params
        bic_after_split[clst_index] = log_likelihood - subclst_n_params / 2.0 * np.log(clst_size)
        # Count number of additional clusters that need to be created based on BIC comparison
        if bic_before_split[clst_index] < bic_after_split[clst_index]:
            add_k += 1
    return add_k


class XMeans(KMeans):
    def __init__(self, kmax=50, max_iter=1000, **k_means_args):
        """

        :param kmax: maximum number of clusters that XMeans can divide the data in
        :param max_iter: maximum number of iterations for the `while` loop (hard limit)
        :param k_means_args: all other parameters supported by sklearn's KMeans algo (except `n_clusters`)
        """
        if 'n_clusters' in k_means_args:
            raise Exception("`n_clusters` is not an accepted parameter for XMeans algorithm")
        if kmax < 1:
            raise Exception("`kmax` cannot be less than 1")
        self.KMax = kmax
        self.max_iter = max_iter
        self.k_means_args = k_means_args

    def fit(self, X, y=None):
        K = 1
        K_sub = 2
        K_old = K
        n_features = np.size(X, axis=1)
        stop_splitting = False
        iter_num = 0
        while not stop_splitting and iter_num < self.max_iter:
            K_old = K
            kmeans = KMeans(n_clusters=K, **self.k_means_args).fit(X)
            clst_labels = kmeans.labels_
            clst_centers = kmeans.cluster_centers_
            # Iterate through all clusters and determine if further split is necessary
            add_k = get_additonal_k_split(K, X, clst_labels, clst_centers, n_features, K_sub, self.k_means_args)
            K += add_k
            # stop splitting clusters when BIC stopped increasing or if max number of clusters in reached
            stop_splitting = K_old == K or K >= self.KMax
            iter_num = iter_num + 1
        # Run vanilla KMeans with the number of clusters determined above
        kmeans = KMeans(n_clusters=K_old, **self.k_means_args).fit(X)
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        self.n_clusters = K_old


