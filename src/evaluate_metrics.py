import os
import json
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

warnings.filterwarnings("ignore")

def main(args):
    print("Starting Comprehensive Model Evaluation...", flush=True)

    # 1. Load Feature Maps
    eventname_edgefeats = json.load(open(args.eventname_edgefeats_path, "r"))
    nodetype_nodefeats = json.load(open(args.nodetype_nodefeats_path, "r"))

    # 2. Load Train Datasets
    print(f"\nLoading Train datasets from: {args.dataset_path}/train", flush=True)
    train_dataset = load_dataset(
        benign_data_path=os.path.join(args.dataset_path, "train/benign"),
        malware_data_path=os.path.join(args.dataset_path, "train/malware"),
        dim_node=len(nodetype_nodefeats),
        dim_edge=len(eventname_edgefeats) + 1
    )

    # Load Test Datasets
    print(f"\nLoading Test datasets from: {args.dataset_path}/test", flush=True)
    test_dataset = load_dataset(
        benign_data_path=os.path.join(args.dataset_path, "test/benign"),
        malware_data_path=os.path.join(args.dataset_path, "test/malware"),
        dim_node=len(nodetype_nodefeats),
        dim_edge=len(eventname_edgefeats) + 1
    )

    # 3. Extract Features (X) and Labels (y)
    print("\nFitting TF-IDF and Extracting N-gram features...", flush=True)
    graphite_ngram = Graphite_Ngram(
        N=args.N[0] if isinstance(args.N, list) else args.N, 
        pool=args.pool[0] if isinstance(args.pool, list) else args.pool
    )
    
    # [FIX]: We MUST call .fit() first so that it initializes 'nodetype_nodefeats', 
    # 'eventname_edgefeats', and most importantly fits the internal TfidfVectorizer.
    graphite_ngram.fit(
        train_dataset=train_dataset,
        nodetype_nodefeats=nodetype_nodefeats,
        eventname_edgefeats=eventname_edgefeats
    )
    
    # Train Data Extraction
    print("Extracting training embeddings...", flush=True)
    X_train = []
    for data in train_dataset:
        emb = graphite_ngram.generate_graph_embedding(data)
        X_train.append(emb.tolist() if hasattr(emb, 'tolist') else emb)
    X_train = np.array(X_train)
    y_train = np.array([1 if "malware" in data.name.lower() else 0 for data in train_dataset])

    # Test Data Extraction
    print("Extracting testing embeddings...", flush=True)
    X_test = []
    for data in test_dataset:
        emb = graphite_ngram.generate_graph_embedding(data)
        X_test.append(emb.tolist() if hasattr(emb, 'tolist') else emb)
    X_test = np.array(X_test)
    y_test = np.array([1 if "malware" in data.name.lower() else 0 for data in test_dataset])

    # 4. K-Fold Cross Validation (For Mean and Max Metrics)
    print("\n" + "="*50)
    print("Running 5-Fold Stratified Cross-Validation (Train Data)")
    print("="*50)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc_scores = []
    cv_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Apply SMOTE only on training fold
        smote = SMOTE(random_state=42)
        X_fold_train_res, y_fold_train_res = smote.fit_resample(X_fold_train, y_fold_train)

        # Initialize Base Estimators with Best Params
        xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
        lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
        rf = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=42, n_jobs=-1)

        # Voting Classifier
        model = VotingClassifier(
            estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
            voting='soft',
            weights=[2.0, 1.5, 1.0] 
        )

        # Train & Predict
        model.fit(X_fold_train_res, y_fold_train_res)
        y_pred = model.predict(X_fold_val)

        # Calculate metrics
        acc = accuracy_score(y_fold_val, y_pred)
        f1 = f1_score(y_fold_val, y_pred, average='macro')
        
        cv_acc_scores.append(acc)
        cv_f1_scores.append(f1)
        print(f"Fold {fold+1} -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    # 5. Final Evaluation on Unseen Test Data (For Test Acc and Test F1)
    print("\n" + "="*50)
    print("Running Final Evaluation on Unseen Test Dataset")
    print("="*50)
    
    # Train on FULL Train Data with SMOTE
    smote_final = SMOTE(random_state=42)
    X_train_full_res, y_train_full_res = smote_final.fit_resample(X_train, y_train)
    
    final_model = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        voting='soft',
        weights=[2.0, 1.5, 1.0]
    )
    final_model.fit(X_train_full_res, y_train_full_res)
    
    # Predict on Test Data
    y_test_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    # 6. Print Final Report
    print("\n" + "★"*50)
    print("🏆 GRAPHITE-X FINAL METRICS REPORT")
    print("★"*50)
    print(f"Mean CV Accuracy : {np.mean(cv_acc_scores)*100:.2f}%")
    print(f"Max CV Accuracy  : {np.max(cv_acc_scores)*100:.2f}%")
    print(f"Mean CV F1-Score : {np.mean(cv_f1_scores):.4f}")
    print(f"Max CV F1-Score  : {np.max(cv_f1_scores):.4f}")
    print("-" * 50)
    print(f"Test Accuracy    : {test_acc*100:.2f}%")
    print(f"Test F1-Score    : {test_f1:.4f}")
    print("★"*50 + "\n")

if __name__ == "__main__":
    args = param_parser()
    main(args)