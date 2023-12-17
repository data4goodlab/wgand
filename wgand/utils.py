import networkx as nx
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def fit_embedding_model(g, emb_model):
    """
    Fit embedding model on a graph.
    Parameters
    ----------
    g : networkx.Graph
        Graph.
    emb_model : karateclub.node_embedding
        Embedding model.
    Returns
    -------
    emb_model : karateclub.node_embedding
        Fitted embedding model.
    """
    for v, u in g.edges:
        g[v][u].pop("weight")

    emb_model.fit(g)
    return emb_model
    
def get_metrics(y, probs):
    """
    Get evaluation metrics.
    Parameters
    ----------
    y : list
        List of labels.
    probs : list
        List of probabilities.
    Returns
    -------
    metrics_dict : dict
        Dictionary of evaluation metrics.
    """
    metrics_dict = {}
    metrics_dict["auc"] = roc_auc_score(y, probs)
    metrics_dict["p@1"] = precision_at_k(y, probs,1)
    metrics_dict["p@3"] = precision_at_k(y, probs,3)
    metrics_dict["p@10"] = precision_at_k(y, probs,10)
    metrics_dict["p@20"] = precision_at_k(y, probs,20)
    metrics_dict["p@anom"] = precision_at_k(y, probs,np.sum(y))
    metrics_dict["anom"] = np.sum(y)
    return metrics_dict


def get_disease_info_df(csv_path):
    """
    Get disease information DataFrame.
    Parameters
    ----------
    csv_path : str
        Path to csv file.
    Returns
    -------
    disease_info : pandas.DataFrame
        DataFrame of disease information.
    """
    disease_info = pd.read_csv(csv_path)
    disease_info["Disease_name"] = disease_info["Disease_name"].str.strip("{")
    disease_info["Disease_name"] = disease_info["Disease_name"].str.strip("}")
    disease_info["Disease_name"] = disease_info["Disease_name"].str.strip("?")
    disease_info["Tissue"] = disease_info["Tissue"].str.replace("-", " ")
    return disease_info

def get_tissue_mapping_df(csv_path,disease_info):
    """
    Get tissue mapping DataFrame that maps between the graph files and the disease info file.
    Parameters
    ----------
    csv_path : str
        Path to csv file.
    disease_info : pandas.DataFrame
        DataFrame of disease information.
    Returns
    -------
    tissue_mapping : pandas.DataFrame
        DataFrame of tissue mapping.
    """
    tissue_mapping = pd.read_csv(csv_path)
    tissue_mapping = tissue_mapping.dropna()
    tissue_mapping = tissue_mapping.merge(pd.DataFrame(disease_info.drop_duplicates(subset=["Tissue","Gene_ID"]).groupby("Tissue").size()), left_on="tissue_name_disease_file", right_on="Tissue", how="left")
    tissue_mapping = tissue_mapping.rename(columns={0:"disease_nodes_num"})
    tissue_mapping = tissue_mapping[tissue_mapping["disease_nodes_num"]>20]
    tissue_mapping = tissue_mapping[~tissue_mapping.tissue_name_network_file.isin(["Breast Mammary Tissue", "Minor Salivary Gland"])]
    return tissue_mapping

def load_tissue_graph_by_matching(tissue_path, tissue_name, disease_info):
    """
    Load tissue graph of the most simillar tissue by name and add disease name to each node.
    Parameters
    ----------
    tissue_path : str
        Path to tissue graph
    tissue_name : str
        Name of tissue to load
    disease_info : pandas.DataFrame
        DataFrame of disease information
    Returns
    -------
    g : networkx.Graph
        Graph of tissue with disease name added to each node
    """
    g = nx.readwrite.gml.read_gml(tissue_path/f"{tissue_name.replace(' ','_')}.gml")
    tissue_alt_name = disease_info.loc[disease_info["Tissue"].apply(lambda x: len(set(x.lower().split()) & set(tissue_name.lower().split()))).idxmax()]["Tissue"]
    tissue_df = disease_info[disease_info["Tissue"]==tissue_alt_name]
    if len(tissue_df)==0:
        raise KeyError(f"{tissue_name} not found")
    gene_disease = dict(zip(tissue_df["Gene_ID"], tissue_df["Disease_name"]))
    for v in g.nodes:
        if v in gene_disease:
            g.nodes[v]["disease_name"] = gene_disease[v]
    return g

def load_tissue_graph(tissue_path, tissue_name, tissue_disease_file_name, disease_info):
    """
    Load tissue graph and add disease name to each node.
    Parameters
    ----------
    tissue_path : str
        Path to tissue graph
    tissue_name : str
        Name of tissue to load
    tissue_disease_file_name : str
        Name of disease file to load
    disease_info : pandas.DataFrame
        DataFrame of disease information
    Returns
    -------
    g : networkx.Graph
        Graph of tissue with disease name added to each node
    """
    g = nx.readwrite.gml.read_gml(tissue_path/f"{tissue_name.replace(' ','_')}.gml")
    tissue_df = disease_info[disease_info["Tissue"]==tissue_disease_file_name]
    if len(tissue_df)==0:
        raise KeyError(f"{tissue_name} not found")
    gene_disease = dict(zip(tissue_df["Gene_ID"], tissue_df["Disease_name"]))
    for v in g.nodes:
        if v in gene_disease:
            g.nodes[v]["disease_name"] = gene_disease[v]
    return g
        
def load_chaperones_graph(graph_path, disease_tissue,tissue_disease_file_name, disease_info):
    """
    Load chaperones graph and add disease name to each node.
    Parameters
    ----------
    graph_path : str
        Path to chaperones graph
    disease_tissue : str
        Name of tissue to load
    tissue_disease_file_name : str  
        Name of disease file to load
    disease_info : pandas.DataFrame 
        DataFrame of disease information
    Returns 
    ------- 
    g : networkx.Graph
        Graph of chaperones with disease name added to each node
    """
    g = nx.readwrite.gml.read_gml(graph_path)

    tissue_df = disease_info[disease_info["Tissue"]==disease_tissue]
    if len(tissue_df)==0:
        raise KeyError(f"{tissue_disease_file_name} not found")
    gene_disease = set(tissue_df["Chaperone  name"])
    for v in g.nodes:
        if v in gene_disease:
            g.nodes[v]["disease_name"] = 1
    return g
            
    
def get_gdf(g, features=None):
    """
    Convert graph to graph data frame.
    Parameters
    ----------
    g : networkx.Graph
        Graph.
    features : list, optional
        List of features.
    Returns
    -------
    gdf : pandas.DataFrame
        Graph data frame.
    """
    g_num = nx.convert_node_labels_to_integers(g, first_label=0, ordering='sorted', label_attribute="node_name")
    gdf = nx.convert_matrix.to_pandas_edgelist(g_num)
    gdf["source"] = gdf["source"].astype(int)
    gdf["target"] = gdf["target"].astype(int)
    gdf = gdf.rename(columns={"weight":"interaction"})
    if features is not None:
        gdf["features"] = features
    return gdf


def eval_weight_predictor(gdf, weight_clf, cv):
    """
    Evaluate weight predictor on graph.
    Parameters
    ----------
    gdf : pandas.DataFrame
        Graph data frame.
    weight_clf : sklearn.base.BaseEstimator
        Weight predictor.
    cv : sklearn.model_selection._split.BaseCrossValidator
        Cross-validation.
    Returns
    -------
    score : float
        Mean cross-validated score.
    """
    X_train = gdf.copy()
    y_train = X_train.pop("interaction")
    scores = cross_val_score(weight_clf, list(X_train["features"]) , y_train, scoring="neg_mean_squared_error", cv=cv)
    return scores.mean()
    
def get_weight_predictor(gdf, weight_clf):
    """
    Fit weight predictor on graph.
    Parameters
    ----------
    gdf : pandas.DataFrame
        Graph data frame.
    weight_clf : sklearn.base.BaseEstimator
        Weight predictor.
    Returns
    -------
    weight_clf : sklearn.base.BaseEstimator
        Weight predictor.
    """
    X_train = gdf.copy()
    y_train = X_train.pop("interaction")
    weight_clf.fit(list(X_train["features"]), y_train)
    setattr(weight_clf, "is_fitted_", True)
    return weight_clf
    


def precision_at_k(y_true, y_pred, k, is_sorted=False):
    """
    Computes precision at k.
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_pred : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    is_sorted : bool, default=False
        Whether y_pred is sorted.
    Returns
    -------
    precision @ k : float
    """
    if not is_sorted:
        sorted_probas = sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)
    else:
        sorted_probas = list(zip(y_pred, y_true))
    return sum([line[1] for line in sorted_probas[:k]]) / k 

def precision_all_k(y_true, y_pred):
    """
    Computes precision at all k.
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_pred : array-like, shape = [n_samples]
        Predicted scores.
    Returns
    -------
    precision @ all k : list of dict
        List of dict with keys "P" and "K".
    """
    total = 0
    res = []
    sorted_probas = sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)
    for k, line in enumerate(sorted_probas):
        total += line[1]
        res.append({"P": total/(k+1), "K": k+1})
    return res

def sigmoid(x):
    """
    Sigmoid function for normalizing the prediction to a probability between 0 and 1.
    """
    return 1/(1 + np.exp(-x))
