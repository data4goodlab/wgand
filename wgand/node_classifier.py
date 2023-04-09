from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, classification_report
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from wgand.utils import precision_at_k
from wgand.base_detector import BaseDetector
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


class NodeClassifier(BaseDetector):
    
    def __init__(self, g, weight_clf, meta_clf, embedding_model=None, feature_selection=False, n_components=0):
        super(NodeClassifier, self).__init__(g, weight_clf, meta_clf, embedding_model, feature_selection, n_components)
        

    def fit(self, X, y):
        """
        Fit the node classifier
        Parameters
        ----------
        X : list
            List of nodes to train on
        y : list
            List of labels for the nodes
        """
        try:
            check_is_fitted(self.weight_clf)
        except NotFittedError:
            self.train_weight_classifier()
        # Todo test order
        X = self.get_node_training_data(X, False)


        if sum(y)>9:

            self.eval["Node Clf"] = self.node_clf.__name__
            self.node_clf.fit(X, y)
            
        else:
            raise ValueError("Not enough positive samples")
        
    
    def get_out_of_fold_predictions(self, X, y, cv=None):
        check_is_fitted(self.weight_clf)
        if cv is None:
            cv = StratifiedKFold(n_splits=10)
            
        X, node_names = self.get_node_training_data(X, True)
            
        if sum(y)>9:

            scores = cross_val_predict(self.node_clf, X=X, y=y,cv=cv, method='predict_proba')
            # idx = np.concatenate([test for _, test in cv.split(X, y)])
            return node_names, scores[:, 1]

        else:
            print("Not enough positive examples")
            return None
    
    def predict_node_proba(self, nodes):
        check_is_fitted(self.node_clf)
        X = self.get_node_training_data(nodes)
        return self.node_clf.predict_proba(X)   
            
    def eval_node_meta_classifier(self, X, y, cv = False, train_test=False, return_train_score=False):
          
        X, _  = self.get_node_training_data(X)

        if train_test:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
                                   
            self.node_clf.fit(X_train,y_train)
            print(classification_report(y_test, self.node_clf.predict(X_test)))           
            print('precision_at_5', precision_at_k(y_test,self.node_clf.predict_proba(X_test)[:, 1], k=5))
            print('precision_at_anom', precision_at_k(y_test,self.node_clf.predict_proba(X_test)[:, 1], k=y_test.sum()))
                                   


        scoring = {'auc': 'roc_auc',
                   "precision": "precision",
                   "recall": "recall",
                   "f1": "f1",
                   'precision_at_1': make_scorer(precision_at_k, k=1, needs_proba=True),
                   'precision_at_3': make_scorer(precision_at_k, k=3, needs_proba=True),
                   'precision_at_10': make_scorer(precision_at_k, k=10, needs_proba=True),
                   'precision_at_20': make_scorer(precision_at_k, k=20, needs_proba=True),
                   # 'precision_at_anom': make_scorer(precision_at_k, k=df_agg["disease"].sum()//10, needs_proba=True)
                  }
        if sum(y)>9:

            if cv:
                    # print(cross_val_score(self.node_clf, df_agg.drop(columns=["id", "disease"]), df_agg["disease"],scoring="precision", cv=5))
                    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
                    if cv is True:
                        cv = StratifiedKFold(n_splits=10, random_state=2, shuffle=True)
                    scores = cross_validate(self.node_clf, X=X, y=y,scoring=scoring, cv=cv, return_train_score=return_train_score,error_score="raise")
                    
                    # print(scores)
                    for k,v in scores.items():
                        if k.startswith("train") or k.startswith("test"):
                            self.eval[k] = v.mean()
                       
            self.eval["Node Clf"] = self.node_clf.__name__
            self.node_clf.fit(X,y)
            
        else:
            print("Not enough positive examples")
            
    

