import pytest
from sklearn.datasets.samples_generator import make_blobs
from xmeans import XMeans


N = 7 # number of clusters to create

@pytest.fixture()
def setup_data():
    X, labels_true = make_blobs(n_samples=700, centers=N, n_features=20, random_state=0)
    xm_clust = XMeans()
    xm_clust.fit(X)
    yield xm_clust, labels_true

def test_num_clusters_detected(setup_data):
    # Ensure the number of created clusters matches the number of detected
    xm_clust, labels_true = setup_data
    N_clust_pred = len(set(xm_clust.labels_))
    assert N == N_clust_pred

def test_cluster_allocation(setup_data):
    # Ensure there one-to-one mapping between cluster labels
    xm_clust, labels_true = setup_data
    labels_pred = xm_clust.labels_
    label_map = {}
    for l_true, l_pred in zip(labels_true, labels_pred):
        if l_true not in label_map:
            label_map[l_true] = [l_pred]
        else:
            label_map[l_true].append(l_pred)
    for key, value in label_map.items():
        assert len(set(value)) == 1
