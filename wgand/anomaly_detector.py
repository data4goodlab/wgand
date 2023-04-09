import numpy as np
from wgand.base_detector import BaseDetector
from wgand.pca_anomaly_detector import PcaAnomalyDetector
from wgand.utils import sigmoid
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
import pyod.models.combination as comb 


class AnomalyDetector(PcaAnomalyDetector):
    """
    Node outlier detector
    """
    
    def __init__(self, g, weight_clf, meta_clf, embedding_model=None, feature_selection=False, n_components=0):
        super(AnomalyDetector, self).__init__(g, weight_clf, meta_clf, embedding_model, feature_selection, n_components)
        # self.pca_tran = PCA(n_components=2, random_state=2)
        
    def fit(self, X):
        """
        Fit the node outlier detector
        Parameters
        ----------
        X : list
            List of nodes to score on
        """
        super().fit(X)
            
        probs_features = self.predict_feature_score(X)

        X = self.get_node_training_data(X)


        probs_pca = super().predict_proba(X, False) 

        X_pca = self.pca_tran.transform(X)

        X = np.concatenate([X, X_pca, probs_pca, probs_features], axis=1)
        self.node_clf.fit(X)
    

    def predict_proba(self, nodes):
        """
        Predict the anomaly score for a list of nodes
        Parameters
        ----------
        nodes : list
            List of nodes to score
        Returns
        -------
        scores : np.array
            Array of scores for the nodes
        """
        return self.predict_node_proba(nodes)
    
    def predict(self, nodes):
        """
        Predict the anomaly score for a list of nodes
        Parameters
        ----------
        nodes : list
            List of nodes to score
        Returns
        -------
        scores : np.array
            Array of scores for the nodes
        """
        return self.predict_node_proba(nodes)[:,1]>0.5

    def predict_node_proba(self, nodes):
        """
        Predict the anomaly score for a list of nodes
        Parameters
        ----------
        nodes : list
            List of nodes to score
        Returns
        -------
        scores : np.array
            Array of scores for the nodes
        """
        self.check_is_fitted()
        X = self.get_node_training_data(nodes)

        X_pca = self.pca_tran.transform(X)
        probs_proba = sigmoid(X_pca[:,1]).reshape(-1, 1) 

        probs_features = self.predict_feature_score(nodes)

        X = np.concatenate([X, X_pca, probs_proba, probs_features], axis=1)
        scores = [self.node_clf.predict_proba(X)[:,1:]]
        scores.append(probs_proba)

        scores.append(probs_features)

        scores = np.concatenate(scores, axis=1)

        return comb.maximization(scores)


    def check_is_fitted(self):
        check_is_fitted(self.node_clf)
        return super().check_is_fitted()