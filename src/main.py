from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

import os
import json
import shap
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns # <--- [NEW] Added seaborn for custom matrix labels
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def main(args):
    # 1. Load Feature Maps
    print(f"Loading feature maps...", flush=True)
    eventname_edgefeats = json.load( open( args.eventname_edgefeats_path, "r") )
    nodetype_nodefeats = json.load( open( args.nodetype_nodefeats_path, "r") )
    
    MODEL_FILE = "saved_graphite_model.pkl"

    # 2. Check if model is already saved
    if os.path.exists(MODEL_FILE):
        print(f"\nSaved model found! Loading from '{MODEL_FILE}'...", flush=True)
        graphite_ngram = joblib.load(MODEL_FILE)
        print("Model loaded successfully! Skipping training phase.\n", flush=True)
    else:
        print(f"\n No saved model found. Training from scratch...", flush=True)
        
        # Load Train Datasets
        print(f"Loading train datasets from: {args.dataset_path}/train", flush=True)
        train_dataset = load_dataset( 
            benign_data_path = os.path.join( args.dataset_path, "train/benign"),  
            malware_data_path = os.path.join( args.dataset_path, "train/malware"), 
            dim_node = len(nodetype_nodefeats),  
            dim_edge= len(eventname_edgefeats) + 1 
        ) 

        n_gram_val = args.N[0] if isinstance(args.N, list) else args.N
        pool_val = args.pool[0] if isinstance(args.pool, list) else args.pool

        print(f"Initializing Graphite-X with N={n_gram_val}...", flush=True)
        graphite_ngram = Graphite_Ngram( 
            N = n_gram_val, 
            pool= pool_val,
            n_estimators=args.estimators,
            learning_rate=args.lr 
        )

        # Training
        graphite_ngram.fit( 
            train_dataset = train_dataset,  
            nodetype_nodefeats = nodetype_nodefeats,  
            eventname_edgefeats= eventname_edgefeats 
        )
        
        print(f"\nSaving trained model to '{MODEL_FILE}'...", flush=True)
        joblib.dump(graphite_ngram, MODEL_FILE)
        print("Model saved successfully!\n", flush=True)

    # 3. Load Test Datasets
    print(f"Loading test datasets from: {args.dataset_path}/test", flush=True)
    test_dataset = load_dataset( 
        benign_data_path =  os.path.join( args.dataset_path,"test/benign"), 
        malware_data_path =  os.path.join( args.dataset_path,"test/malware"), 
        dim_node = len(nodetype_nodefeats), 
        dim_edge= len(eventname_edgefeats) + 1 
    )

    # 4. Testing & Collecting Data
    print("Running evaluation on test set...", flush=True)
    preds, truths, script_names = [], [], [] 
    test_embeddings = [] 

    for idx, test_data in enumerate(test_dataset):
        emb = graphite_ngram.generate_graph_embedding(test_data)
        emb_list = emb.tolist()
        test_embeddings.append(emb_list)
        
        pred = graphite_ngram.base_model.predict([emb_list]).item()
        truth  = [ 1 if "malware" in test_data.name.lower() else 0 ][0]
        
        if idx % 50 == 0:
            print(f"Predicted: {pred} | Truth: {truth} --- {test_data.name}", flush=True)
            
        preds.append(pred)
        truths.append(truth)
        script_names.append(test_data.name) 

    # 5. Results & Metrics
    test_acc = accuracy_score(y_true = truths, y_pred = preds)
    test_f1 = f1_score(y_true = truths, y_pred = preds)

    print("\n" + "="*50, flush=True)
    print(f"Final Results (Graphite-X):", flush=True)
    print(f"Test-Acc: {test_acc:.4f} ({test_acc*100:.2f}%)", flush=True)
    print(f"Test-F1 : {test_f1:.4f}", flush=True)
    print("="*50, flush=True)

    # 6. Save Prediction Analysis for Correlation (CSV)
    print("\nSaving prediction analysis to CSV...", flush=True)
    df = pd.DataFrame({
        "Script_Name": script_names,
        "True_Label": truths,
        "Predicted_Label": preds
    })
    df['True_Class'] = df['True_Label'].map({0: 'Benign', 1: 'Malware'})
    df['Predicted_Class'] = df['Predicted_Label'].map({0: 'Benign', 1: 'Malware'})
    
    df['Result_Type'] = 'Correct'
    df.loc[(df['True_Label'] == 1) & (df['Predicted_Label'] == 0), 'Result_Type'] = 'False Negative (Missed Malware)'
    df.loc[(df['True_Label'] == 0) & (df['Predicted_Label'] == 1), 'Result_Type'] = 'False Positive (False Alarm)'
    
    df.to_csv("prediction_analysis.csv", index=False)
    print(">> Saved 'prediction_analysis.csv'")

    # --- [NEW] 7. Generate Custom Labeled Confusion Matrix Plot ---
    print("Generating Confusion Matrix Plot...", flush=True)
    cm = confusion_matrix(truths, preds)
    
    # Create custom labels with actual values
    group_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, 
                xticklabels=["Benign", "Malware"], yticklabels=["Benign", "Malware"],
                annot_kws={"size": 14})
    
    plt.title("Confusion Matrix for Graphite-X", pad=15, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(">> Saved 'confusion_matrix.png' with proper labels!")

    # 8. Explainable AI (SHAP) Logic
    print("\nGenerating Explainable AI (XAI) Plots...", flush=True)
    try:
        feature_names = graphite_ngram.get_feature_names()
        xgb_model = graphite_ngram.base_model.estimators_[0]
        
        X_test_matrix = np.array(test_embeddings)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_matrix)

        plt.figure(figsize=(14, 10))
        plt.title("Top 20 Features Driving Malware Detection (SHAP)", pad=20)
        shap.summary_plot(shap_values, X_test_matrix, feature_names=feature_names, plot_type="bar", max_display=20, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary_bar.png", bbox_inches='tight', dpi=300)
        plt.close() 

        plt.figure(figsize=(14, 10))
        plt.title("Feature Impact Direction (SHAP Beeswarm)", pad=20)
        shap.summary_plot(shap_values, X_test_matrix, feature_names=feature_names, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig("shap_beeswarm.png", bbox_inches='tight', dpi=300)
        plt.close() 
        
        print("XAI Analysis Complete. Check the high-quality .png files.")

    except Exception as e:
        print(f"XAI Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    args = param_parser()
    main(args)