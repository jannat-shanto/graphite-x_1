import optuna
import os
import json
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

warnings.filterwarnings("ignore")

def main(args):
    print("🚀 Starting Hyperparameter Optimization with Optuna...", flush=True)

   
    eventname_edgefeats = json.load(open(args.eventname_edgefeats_path, "r"))
    nodetype_nodefeats = json.load(open(args.nodetype_nodefeats_path, "r"))

    print(f"Loading datasets...", flush=True)
    
    
    train_dataset = load_dataset(
        benign_data_path=os.path.join(args.dataset_path, "train/benign"),
        malware_data_path=os.path.join(args.dataset_path, "train/malware"),
        dim_node=len(nodetype_nodefeats),
        dim_edge=len(eventname_edgefeats) + 1
    )

   
    n_gram_val = args.N[0] if isinstance(args.N, list) else args.N
    
    print("Generating Embeddings (This happens once)...", flush=True)
    temp_graphite = Graphite_Ngram(N=n_gram_val)
    
    temp_graphite.nodetype_nodefeats = nodetype_nodefeats
    temp_graphite.eventname_edgefeats = eventname_edgefeats
    temp_graphite.fit_count_vectorizer(train_dataset)
    
   
    X_raw = []
    y_raw = []
    
    cnt = 0
    for data in train_dataset:
        emb = temp_graphite.generate_graph_embedding(data).tolist()
        X_raw.append(emb)
        label = 1 if "malware" in data.name.lower() else 0
        y_raw.append(label)
        cnt += 1
        if cnt % 100 == 0:
            print(f"Processed {cnt} graphs...", flush=True)

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)

  
    print("Applying SMOTE before optimization...", flush=True)
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X_raw, y_raw)
    print(f"Optimization Dataset Shape: {X.shape}", flush=True)

    
    def objective(trial):
        
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 500, 2000)
        xgb_lr = trial.suggest_float('xgb_lr', 0.01, 0.1, log=True)
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 12)
        xgb_subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
        
        lgbm_n_estimators = trial.suggest_int('lgbm_n_estimators', 500, 2000)
        lgbm_lr = trial.suggest_float('lgbm_lr', 0.01, 0.1, log=True)
        lgbm_num_leaves = trial.suggest_int('lgbm_num_leaves', 20, 100)
        
       
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 500, 1500)
        rf_max_depth = trial.suggest_int('rf_max_depth', 10, 50)
        
        
        w_xgb = trial.suggest_float('w_xgb', 1.0, 3.0)
        w_lgbm = trial.suggest_float('w_lgbm', 1.0, 3.0)
        w_rf = trial.suggest_float('w_rf', 0.5, 2.0)

       
        clf_xgb = XGBClassifier(
            n_estimators=xgb_n_estimators, learning_rate=xgb_lr, max_depth=xgb_max_depth,
            subsample=xgb_subsample, colsample_bytree=0.7, n_jobs=-1, random_state=42, verbosity=0
        )
        
        clf_lgbm = LGBMClassifier(
            n_estimators=lgbm_n_estimators, learning_rate=lgbm_lr, num_leaves=lgbm_num_leaves,
            n_jobs=-1, random_state=42, verbose=-1
        )
        
        clf_rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth, n_jobs=-1, random_state=42
        )
        
       
        model = VotingClassifier(
            estimators=[('xgb', clf_xgb), ('lgbm', clf_lgbm), ('rf', clf_rf)],
            voting='soft',
            weights=[w_xgb, w_lgbm, w_rf]
        )
        
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    print("\n Optuna is searching for the best hyperparameters... (This will take time)")
    study.optimize(objective, n_trials=30) 

   
    print("\n" + "="*50)
    print(" Optimization Finished!")
    print(f"Best Trial Accuracy: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("Saved best parameters to 'best_params.json'")

if __name__ == "__main__":
    args = param_parser()
    main(args)