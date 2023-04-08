import numpy as np
from wgand.base_estimator import BaseEstimator
from wgand.utils import sigmoid
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
import pyod.models.combination as comb 


class AnomalyDetector(BaseEstimator):
    """
    Node outlier detector
    """
    
    def __init__(self, g, weight_clf, meta_clf, embedding_model=None, feature_selection=False, n_components=0):
        super(AnomalyDetector, self).__init__(g, weight_clf, meta_clf, embedding_model, feature_selection, n_components)
        self.pca_tran = PCA(n_components=2, random_state=2)
        
    def fit(self, X):
        """
        Fit the node outlier detector
        Parameters
        ----------
        X : list
            List of nodes to score on
        """
        try:
            check_is_fitted(self.weight_clf)
        except NotFittedError:
            self.train_weight_classifier()
            
        probs2 = self.predict_feature_score(X)

        X = self.get_node_training_data(X)


        self.pca_tran.fit(X)

        X_pca = self.pca_tran.transform(X)
    
        scaler = MinMaxScaler().fit(X_pca[:,1:])
        probs1 = scaler.transform(X_pca[:,1:]).ravel().clip(0, 1).reshape(-1, 1)
        probs1 = sigmoid(X_pca[:,1]).reshape(-1, 1) 

        

        X = np.concatenate([X, X_pca, probs1, probs2], axis=1)
        self.node_clf.fit(X)

    
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
        check_is_fitted(self.node_clf)
        X = self.get_node_training_data(nodes)

        X_pca = self.pca_tran.transform(X)
    
        scaler = MinMaxScaler().fit(X_pca[:,1:])
        probs1 = scaler.transform(X_pca[:,1:]).ravel().clip(0, 1).reshape(-1, 1)
        probs1 = sigmoid(X_pca[:,1]).reshape(-1, 1) 

        probs2 = self.predict_feature_score(nodes)

        X = np.concatenate([X, X_pca, probs1, probs2], axis=1)
        scores = [self.node_clf.predict_proba(X)[:,1:]]
        scores.append(probs1)

        scores.append(probs2)

        scores = np.concatenate(scores, axis=1)

        return comb.maximization(scores)
