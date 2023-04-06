import networkx as nx
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, RFE, RFECV
from sklearn.feature_selection import f_classif
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_validate, cross_val_predict
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from NetAnom.utils import get_gdf, get_weight_predictor, precision_at_k, eval_weight_predictor
from sklearn.decomposition import PCA


class NodeClassifier(object):
    
    def __init__(self, g, embedding_model=None):
        self.g = g
        self.g_num = nx.convert_node_labels_to_integers(g, first_label=0, ordering='sorted', label_attribute="gen")
        self.embedding_model = embedding_model
        self.gdf = get_gdf(self.g)
        if embedding_model:
            self.gdf["features"] = self.gdf.progress_apply(lambda x: np.concatenate([embedding_model.get_embedding()[int(x["source"])],embedding_model.get_embedding()[int(x["target"])]]), axis=1)
        self.weight_clf = None
        self.node_clf = None
        self.eval = {}
        
    def set_gdf_features(self, feature):
        self.gdf["features"] = feature
        
    def train_weight_classifier(self, clf, clf_params, clf_eval = False):
        
        self.weight_clf = get_weight_predictor(self.gdf, clf, clf_params)

        if clf_eval:
            self.eval["R2"] =  self.weight_clf.score(list(self.gdf["features"]) , self.gdf["interaction"])

            self.eval["MSE"] = eval_weight_predictor(self.gdf, clf, clf_params)
            
        self.eval["Weight Clf"] = clf.__name__
            
    def get_node_features(self):
        self.gdf["score"] = self.weight_clf.predict(list(self.gdf["features"]))
        # self.gdf["score"] = cross_val_predict(self.weight_clf, X=list(self.gdf["features"]), y=list(self.gdf["interaction"]),cv=2)
        self.gdf["diff"] = self.gdf["interaction"] - self.gdf["score"]
        self.gdf["abs_diff"] = np.abs(self.gdf["interaction"] - self.gdf["score"])
        
        gen_df = self.gdf[["source","diff", "abs_diff"]]
        gen_df2 = self.gdf[["target","diff", "abs_diff"]]
        gen_df = gen_df.rename(columns={"source":"gen"})
        gen_df2 = gen_df2.rename(columns={"target":"gen"})
        gen_df = gen_df.append(gen_df2)
        df_agg = gen_df.groupby("gen").agg({"diff":["mean", "std", "median","sum", "sem"], "abs_diff":["mean", "std", "median","sum", "sem"]})
        df_agg["id"] =  df_agg.index
        # for f in ["mean", "std", "median","sum", "sem"]:
        #     df_agg["diff2", f] = 1/df_agg["diff"][f]
        df_agg["gen_name"] = df_agg["id"].apply(lambda x: self.g_num.nodes[x]['gen'])
        df_agg["disease"] = df_agg["id"].apply(lambda x: self.g_num.nodes[x])
        df_agg["disease"] = df_agg["disease"].apply(lambda x: 1 if "disease_name" in x else 0)
        df_agg = df_agg.sort_values([("diff", "mean")], ascending=False)
        df_agg = df_agg.fillna(0)
        df_agg.columns = (df_agg.columns.get_level_values(0) +"_" + df_agg.columns.get_level_values(1)).str.strip("_")
        return df_agg
    
    def get_node_training_data(self, feature_selection, pca, return_gene_names=False):

    
        df_agg = self.get_node_features()
        # df_agg = df_agg.sample(frac=1, random_state=2)
        df_agg = df_agg.sort_values("gen_name")
        y = df_agg["disease"].values

        if type(feature_selection) is list:
            X = df_agg[feature_selection].values
        else:
            X = df_agg.drop(columns=["disease", "id", "gen_name"]).values
            
        if pca:
            pca_tran = PCA(n_components=pca, random_state=2).fit(X)
            X_pca = pca_tran.transform(X)
            X = np.concatenate([X,  X_pca], axis=1)    
            
        if hasattr(feature_selection, "get_support"):
            fit = feature_selection.fit(X, y)
            # summarize scores
            X = fit.transform(X)
        
        if return_gene_names:
            return X, y, df_agg["gen_name"]
        
        return X, y
            
    def train_node_meta_classifier(self, clf, clf_params, cv = False, train_test=False, return_train_score=False, calibrator=False, calib_params=None, feature_selection=False, pca=0):
        
        
        if calibrator:
            self.node_clf = calibrator(clf(**clf_params), **calib_params)
        else:
            self.node_clf = clf(**clf_params)
        

        X, y = self.get_node_training_data(feature_selection, pca)



            
        # rus = RandomUnderSampler(random_state=42, sampling_strategy="majority")
        # X, y  = rus.fit_resample(df_agg.drop(columns=["id", "disease"]), df_agg["disease"])
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
                       
            self.eval["Node Clf"] = clf.__name__
            self.node_clf.fit(X,y)
            
        else:
            print("Not enough positive examples")
            
    # CalibratedClassifierCV
    
    def get_node_probas(self, clf, clf_params, calibrator=None, calib_params=None, cv=None, feature_selection=None, pca=0):
        if cv is None:
            cv = StratifiedKFold(n_splits=10)
            
        if calibrator is not None:
            self.node_clf = calibrator(clf(**clf_params), **calib_params)
        else:
            self.node_clf = clf(**clf_params)
        X, y, gene_names = self.get_node_training_data(feature_selection, pca, True)
            
        if sum(y)>9:

            scores = cross_val_predict(self.node_clf, X=X, y=y,cv=cv, method='predict_proba')
            # idx = np.concatenate([test for _, test in cv.split(X, y)])
            return gene_names, scores[:, 1]

        else:
            print("Not enough positive examples")
            return None

        
    def get_node_probas2(self):
        df_agg = self.get_node_features()
        df_agg = df_agg.sample(frac=1, random_state=2)
        X, y = df_agg.drop(columns=["disease", "id", "gen_name"]).values, df_agg["disease"].values

        if sum(y)>9:

            scores = self.node_clf.predict_proba(X)
            # idx = np.concatenate([test for _, test in cv.split(X, y)])
            return df_agg["gen_name"], scores[:, 1]

        else:
            print("Not enough positive examples")
            return None