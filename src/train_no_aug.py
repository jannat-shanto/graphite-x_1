import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

def main(args):
    print("Running Graphite Training WITHOUT Augmentation (No SMOTE)...", flush=True)

    
    eventname_edgefeats = json.load(open(args.eventname_edgefeats_path, "r"))
    nodetype_nodefeats = json.load(open(args.nodetype_nodefeats_path, "r"))

    
    print("\nLoading train datasets...", flush=True)
    train_dataset = load_dataset(
        benign_data_path=os.path.join(args.dataset_path, "train/benign"),
        malware_data_path=os.path.join(args.dataset_path, "train/malware"),
        dim_node=len(nodetype_nodefeats),
        dim_edge=len(eventname_edgefeats) + 1
    )

    print("\nLoading test datasets...", flush=True)
    test_dataset = load_dataset(
        benign_data_path=os.path.join(args.dataset_path, "test/benign"),
        malware_data_path=os.path.join(args.dataset_path, "test/malware"),
        dim_node=len(nodetype_nodefeats),
        dim_edge=len(eventname_edgefeats) + 1
    )

   
    n_gram_val = args.N[0] if isinstance(args.N, list) else args.N
    pool_val = args.pool[0] if isinstance(args.pool, list) else args.pool

    graphite_ngram = Graphite_Ngram(
        N=n_gram_val,
        pool=pool_val,
        n_estimators=args.estimators,
        learning_rate=args.lr
    )

    
    graphite_ngram.nodetype_nodefeats = nodetype_nodefeats
    graphite_ngram.eventname_edgefeats = eventname_edgefeats
    graphite_ngram.fit_count_vectorizer(train_dataset)

   
    print("\nGenerating train embeddings...", flush=True)
    X_train, y_train = [], []
    for data in train_dataset:
        emb = graphite_ngram.generate_graph_embedding(data).tolist()
        X_train.append(emb)
        y_train.append(1 if "malware" in data.name.lower() else 0)

    
    print(f"\nTraining Ensemble on Original Dataset Size: {len(y_train)} | Malware: {sum(y_train)}", flush=True)
    
    
    graphite_ngram.base_model.fit(X=np.array(X_train), y=np.array(y_train))
    print(" Model fitted perfectly without augmentation!", flush=True)

    
    print("\nRunning evaluation on test set...", flush=True)
    preds, truths = [], []
    for idx, test_data in enumerate(test_dataset):
        emb_list = graphite_ngram.generate_graph_embedding(test_data).tolist()
        pred = graphite_ngram.base_model.predict([emb_list]).item()
        truth = 1 if "malware" in test_data.name.lower() else 0
        
        preds.append(pred)
        truths.append(truth)

    
    test_acc = accuracy_score(y_true=truths, y_pred=preds)
    test_f1 = f1_score(y_true=truths, y_pred=preds)

    print("\n" + "="*50, flush=True)
    print(" Final Results (NO Augmentation):", flush=True)
    print(f"Test-Acc: {test_acc:.4f} ({test_acc*100:.2f}%)", flush=True)
    print(f"Test-F1 : {test_f1:.4f}", flush=True)
    print("="*50, flush=True)

if __name__ == "__main__":
    args = param_parser()
    main(args)