# WGAND - Weighte Graph Anomalus Node Detection

This repository contains the official implementaion of the paper "Weighted Graph Anomaly Node Detection" by Dima Kagan, Micahel Fire, Juma Jubran, and Esti Yeger-Lotem.
The code is implentaed as a python library with the the paper experiments executed inside jupyter notebooks.

## Installation
```bash
$ git clone https://github.com/data4goodlab/wgand.git
$ cd wgand
$ pip install -r requirements.txt
$ python setup.py install
```
To run the notebook also install: 
```bash
$ pip install xgboost lightgbm
```
## Usage
WGAND (Ensemble)
```python
from pyod.models.iforest import IForest
from sklearn.ensemble import RandomForestRegressor
from wgand.anomaly_detector import EnsembleAnomalyDetector
from wgand.utils import load_tissue_graph, get_metrics
from karateclub.node_embedding.neighbourhood import RandNE
from wgand.utils import get_metrics, get_disease_info_df, get_tissue_mapping_df

disease_info = get_disease_info_df("data/Disease_Info.csv")
tissue_mapping = get_tissue_mapping_df("data/tissue_mapping_new.csv",disease_info)

tissue_name = tissue_mapping.iloc[0]["tissue_name_network_file"]
tissue_name_disease_file = tissue_mapping.iloc[0]["tissue_name_disease_file"]

try:
    g = load_tissue_graph(tissue_path, tissue_name, tissue_name_disease_file, disease_info)

    nodes = list(g.nodes)
    y = [1 if "disease_name" in g.nodes[n] else 0 for n in nodes ]
except KeyError:
    print(f"{tissue_name} not found in disease info")

nad = EnsembleAnomalyDetector(g, RandomForestRegressor(n_jobs=-1,n_estimators=500,random_state=2), IForest(n_jobs=-1,random_state=2), embedding_model=RandNE())
nad.fit(nodes)

probs = nad.predict_node_proba(nodes)
print(get_metrics(y, probs))
```

WGAND (Feature's Mean)
```python
from sklearn.ensemble import RandomForestRegressor
from wgand.base_detector import BaseDetector as WgandDetector
from wgand.utils import load_tissue_graph, get_metrics
from karateclub.node_embedding.neighbourhood import RandNE
from wgand.utils import get_metrics, get_disease_info_df, get_tissue_mapping_df

disease_info = get_disease_info_df("data/Disease_Info.csv")
tissue_mapping = get_tissue_mapping_df("data/tissue_mapping_new.csv",disease_info)

tissue_name = tissue_mapping.iloc[0]["tissue_name_network_file"]
tissue_name_disease_file = tissue_mapping.iloc[0]["tissue_name_disease_file"]

try:
    g = load_tissue_graph(tissue_path, tissue_name, tissue_name_disease_file, disease_info)

    nodes = list(g.nodes)
    y = [1 if "disease_name" in g.nodes[n] else 0 for n in nodes ]
except KeyError:
    print(f"{tissue_name} not found in disease info")

nad = WgandDetector(g, RandomForestRegressor(n_jobs=-1,n_estimators=500,random_state=2), embedding_model=RandNE())
nad.fit(nodes)
  

probs = nad.predict_feature_score(nodes)
print(get_metrics(y, probs))
```
## Citation

## License
GNU General Public License v3.0

