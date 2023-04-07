import networkx as nx
from pathlib import Path
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import cross_val_score
import numpy as np



def load_tissue_graph_by_matching(tissue_path, tissue_name, disease_info):
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
    g = nx.readwrite.gml.read_gml(tissue_path/f"{tissue_name.replace(' ','_')}.gml")
    # tissue_alt_name = tissue_mapping[tissue_mapping["tissue_name_network_file"]==tissue_name]["tissue_name_disease_file"].iloc[0]
    tissue_df = disease_info[disease_info["Tissue"]==tissue_disease_file_name]
    if len(tissue_df)==0:
        raise KeyError(f"{tissue_name} not found")
    gene_disease = dict(zip(tissue_df["Gene_ID"], tissue_df["Disease_name"]))
    for v in g.nodes:
        if v in gene_disease:
            g.nodes[v]["disease_name"] = gene_disease[v]
    return g
        
def load_chaperones_graph(graph_path, disease_tissue,tissue_disease_file_name, disease_info):
    g = nx.readwrite.gml.read_gml(graph_path)

    tissue_df = disease_info[disease_info["Tissue"]==disease_tissue]
    if len(tissue_df)==0:
        raise KeyError(f"{tissue_disease_file_name} not found")
    gene_disease = set(tissue_df["Chaperone  name"])
    for v in g.nodes:
        if v in gene_disease:
            g.nodes[v]["disease_name"] = 1
    return g
            
    
# df["features"]

def get_gdf(g, features=None):
    g_num = nx.convert_node_labels_to_integers(g, first_label=0, ordering='sorted', label_attribute="node_name")
    gdf = nx.convert_matrix.to_pandas_edgelist(g_num)
    gdf["source"] = gdf["source"].astype(int)
    gdf["target"] = gdf["target"].astype(int)
    gdf = gdf.rename(columns={"weight":"interaction"})
    if features is not None:
        gdf["features"] = features
    return gdf


def eval_weight_predictor(gdf, weight_clf, cv=10):
    X_train = gdf.copy()
    y_train = X_train.pop("interaction")
    # n_estimators=100,n_jobs=-1,max_features=10
    # weight_clf = classifier(**clf_params)
    # cores = cross_val_score(clf, pd.concat([df_agg["diff"], df_agg["gen"]], axis=1), df_agg["disease"],scoring="roc_auc", cv=10)
    scores = cross_val_score(weight_clf, list(X_train["features"]) , y_train, scoring="neg_mean_squared_error", cv=cv)
    return scores.mean()
    
def get_weight_predictor(gdf, weight_clf):

    X_train = gdf.copy()
    y_train = X_train.pop("interaction")
    # n_estimators=100,n_jobs=-1,max_features=10
    # weight_clf = classifier(**clf_params)
    weight_clf.fit(list(X_train["features"]), y_train)
    return weight_clf
    


def precision_at_k(y_true, y_pred, k, is_sorted=False):
    if not is_sorted:
        sorted_probas = sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)
    else:
        sorted_probas = list(zip(y_pred, y_true))
    return sum([line[1] for line in sorted_probas[:k]]) / k 

def precision_all_k(y_true, y_pred):
    total = 0
    res = []
    sorted_probas = sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)
    for k, line in enumerate(sorted_probas):
        total += line[1]
        res.append({"P": total/(k+1), "K": k+1})
    return res

def sigmoid(x):
    return 1/(1 + np.exp(-x))
