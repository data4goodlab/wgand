import numpy as np
from wgand.base_detector import BaseDetector
from wgand.utils import sigmoid
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
import pyod.models.combination as comb 


class PcaAnomalyDetector(BaseDetector):
    """
    Node outlier detector
    """
    
    def __init__(self, g, weight_clf, meta_clf=None, embedding_model=None, feature_selection=False, n_components=0):
        super(PcaAnomalyDetector, self).__init__(g, weight_clf, meta_clf, embedding_model, feature_selection, n_components)
        self.pca_tran = PCA(n_components=2, random_state=2)
        
    def fit(self, X=None):
        """
        Fit the node outlier detector
        Parameters
        ----------
        X : list
            List of nodes to score on
        """
        super(PcaAnomalyDetector, self).fit(X)
            
        try:
            check_is_fitted(self.pca_tran)
        except NotFittedError:
            all_data = self.get_node_training_data()
            self.pca_tran.fit(all_data)


    def predict_score(self, X, transform=True):
        """
        Predict the anomaly score for a list of nodes
        Parameters
        ----------
        X : list
            List of nodes to score
        Returns
        -------
        scores : np.array
            Array
        """
        try:
            check_is_fitted(self.weight_clf)
        except NotFittedError:
            self.fit(X)
 
 
        if transform:
            X = self.get_node_training_data(X)
        X_pca = self.pca_tran.transform(X)
    
        # scaler = MinMaxScaler().fit(X_pca[:,1:])
        # probs = scaler.transform(X_pca[:,1:]).ravel().clip(0, 1).reshape(-1, 1)
        return X_pca[:,1].reshape(-1, 1)

  

    def predict_proba(self, nodes, transform=True):
        """
        Predict the probability of a node being outlier.
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
        scores = self.predict_score(nodes, transform)  
        return sigmoid(scores)



    def predict(self, nodes):
        """
        Predict the label of a node being outlier.
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
        scores = self.predict_proba(nodes)
        return (scores > 0.5).astype(int)
    
    def check_is_fitted(self):
        check_is_fitted(self.pca_tran)
        return super().check_is_fitted()