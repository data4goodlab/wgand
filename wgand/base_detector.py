import networkx as nx
import numpy as np
from wgand.utils import get_gdf, get_weight_predictor, eval_weight_predictor, sigmoid
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import pyod.models.combination as comb 

class BaseDetector(object):
    
    def __init__(self, g, weight_clf, meta_clf, embedding_model=None, feature_selection=False, n_components=0):
        self.g = g
        self.g_num = nx.convert_node_labels_to_integers(g, first_label=0, ordering='sorted', label_attribute="node_name")
        self.embedding_model = embedding_model
        self.gdf = get_gdf(self.g)
        if embedding_model:
            self.gdf["features"] = self.gdf.progress_apply(lambda x: np.concatenate([embedding_model.get_embedding()[int(x["source"])],embedding_model.get_embedding()[int(x["target"])]]), axis=1)
        self.weight_clf = weight_clf
        self.node_clf = meta_clf
        self.eval = {}
        self.feature_selection = feature_selection 
        self.n_components = n_components 
        # self._is_fitted = False
        
    def set_gdf_features(self, feature):
        self.gdf["features"] = feature
        
    # def train_weight_classifier(self):
        
    #     self.weight_clf = get_weight_predictor(self.gdf, self.weight_clf)

    def fit(self, X=None):
        """
        Fit the node outlier detector
        Parameters
        ----------
        X : list
            List of nodes to score on
        """
        try:
            self.check_is_fitted()
        except NotFittedError:
            self.weight_clf = get_weight_predictor(self.gdf, self.weight_clf)
            
    def check_is_fitted(self):
        check_is_fitted(self.weight_clf)
        return True


    def eval_weight_estimator(self, cv=10):
        self.eval["R2"] =  self.weight_clf.score(list(self.gdf["features"]) , self.gdf["interaction"])

        self.eval["MSE"] = eval_weight_predictor(self.gdf,  self.weight_clf, cv)
            
        self.eval["Weight Clf"] =  self.weight_clf.__name__
            
    def get_node_features(self):
        self.gdf["score"] = self.weight_clf.predict(list(self.gdf["features"]))
        # self.gdf["score"] = cross_val_predict(self.weight_clf, X=list(self.gdf["features"]), y=list(self.gdf["interaction"]),cv=2)
        self.gdf["diff"] = self.gdf["interaction"] - self.gdf["score"]
        self.gdf["abs_diff"] = np.abs(self.gdf["interaction"] - self.gdf["score"])
        
        node_df = self.gdf[["source","diff", "abs_diff"]]
        node_df2 = self.gdf[["target","diff", "abs_diff"]]
        node_df = node_df.rename(columns={"source":"node"})
        node_df2 = node_df2.rename(columns={"target":"node"})
        node_df = node_df.append(node_df2)
        df_agg = node_df.groupby("node").agg({"diff":["mean", "std", "median","sum", "sem"], "abs_diff":["mean", "std", "median","sum", "sem"]})
        df_agg["id"] =  df_agg.index
        # for f in ["mean", "std", "median","sum", "sem"]:
        #     df_agg["diff2", f] = 1/df_agg["diff"][f]
        df_agg["node_name"] = df_agg["id"].apply(lambda x: self.g_num.nodes[x]['node_name'])

        # df_agg = df_agg.sort_values([("diff", "mean")], ascending=False)
        df_agg = df_agg.fillna(0)
        df_agg.columns = (df_agg.columns.get_level_values(0) +"_" + df_agg.columns.get_level_values(1)).str.strip("_")
        return df_agg
    
    def get_node_training_data(self, nodes=None, return_node_names=False):

    
        df_agg = self.get_node_features()
        # df_agg = df_agg.sample(frac=1, random_state=2)
        if nodes is not None:
            df_agg = df_agg[df_agg["node_name"].isin(nodes)]
            df_agg.node_name = pd.Categorical(df_agg.node_name, categories=nodes, ordered=True)
            df_agg = df_agg.sort_values("node_name")
        else: 
            df_agg = df_agg.sort_values("node_name")

        if type(self.feature_selection) is list:
            X = df_agg[self.feature_selection].values
        else:
            X = df_agg.drop(columns=["id", "node_name"]).values
            
        if self.n_components:
            pca_tran = PCA(n_components=self.n_components, random_state=2).fit(X)
            X_pca = pca_tran.transform(X)
            X = np.concatenate([X,  X_pca], axis=1)    
            
        if hasattr(self.feature_selection, "get_support"):
            # summarize scores
            X = self.feature_selection.transform(X)
        
        if return_node_names:
            return X, df_agg["node_name"].values
        
        return X
            
    def predict_feature_score(self, nodes):
        """
        Predict the feature score for a list of nodes
        Parameters
        ----------
        nodes : list
            List of nodes to score
        Returns
        -------
        scores : np.array
            Array of scores for the nodes
        """
        check_is_fitted(self.weight_clf)
        X = self.get_node_training_data(nodes)

        scores = sigmoid(X)

        return comb.average(scores).reshape(-1, 1)