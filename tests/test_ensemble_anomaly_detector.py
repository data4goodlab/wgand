from pyod.models.iforest import IForest
from sklearn.ensemble import RandomForestRegressor
from wgand import EnsembleAnomalyDetector
from wgand.utils import load_tissue_graph, get_metrics
from karateclub.node_embedding.neighbourhood import RandNE
from wgand.utils import get_metrics, get_disease_info_df, get_tissue_mapping_df
import networkx as nx
import numpy as np

'''
Functions to test the EnsembleAnomalyDetector class.
'''
def generate_random_weighted_graph():
    """
    Generate a random weighted graph.
    """
    G = nx.erdos_renyi_graph(100, 0.1)
    for edge in G.edges():
        G.edges[edge]["weight"] = np.random.uniform(0, 1)
    return G

def test_ensemble_anomaly_detector():
    """
    Test the EnsembleAnomalyDetector class.
    """
    G = generate_random_weighted_graph()
    model = EnsembleAnomalyDetector(
        weight_clf=RandomForestRegressor(),
        meta_clf=IForest(),
        embedding_model=RandNE(),
    )
    model.fit(G)
    scores = model.predict(list(G.nodes()))
    assert scores.shape[0] == G.number_of_nodes()
    