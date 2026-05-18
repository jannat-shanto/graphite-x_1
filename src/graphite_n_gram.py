import warnings
warnings.filterwarnings("ignore") 

from typing import List 
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.ensemble import VotingClassifier, RandomForestClassifier 
from xgboost import XGBClassifier                            
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE  # For Augmentation

class Graphite_Ngram:
   r""" An implementation of Graphite N-gram (Optimized with Optuna). """

   def __init__(self, 
                N : int = 4, 
                pool : str = "sum",
                n_estimators : int = 1500, 
                learning_rate : float = 0.02): 
      
      self.N = N
      pool_choices = {"sum": torch.sum, "mean": torch.mean, "max": torch.max}
      self.pool = pool_choices[ pool ]
      
      # --- AGGRESSIVE FEATURE EXTRACTION ---
      self.count_vectorizer = TfidfVectorizer( 
                                               ngram_range = (1, 5), 
                                               max_df = 0.95,        
                                               min_df = 1,           
                                               max_features = 15000, 
                                               sublinear_tf=True     
                                             )

      # --- Level 1: The Models (Optimized with Trial 0 Params) ---
      
      # 1. XGBoost (Updated)
      clf_xgb = XGBClassifier(
         n_estimators= 1176,          
         learning_rate= 0.0125,       
         max_depth= 7,                
         subsample= 0.73,             
         colsample_bytree= 0.7,
         objective= 'binary:logistic',
         n_jobs= -1, 
         random_state= 2024,     
         eval_metric='logloss',
         verbosity=0
      )

      # 2. LightGBM (Updated)
      clf_lgbm = LGBMClassifier(
         n_estimators= 1824,          
         learning_rate= 0.0529,       
         num_leaves= 21,              
         max_depth= -1,
         subsample= 0.7,
         colsample_bytree= 0.7,
         n_jobs= -1, 
         random_state= 2024,     
         verbose= -1
      )

      # 3. Random Forest (Updated)
      clf_rf = RandomForestClassifier(
          n_estimators= 1441,         
          max_depth= 33,              
          n_jobs=-1, 
          random_state=2024      
      )

      # --- Level 2: Voting with Weights (Updated) ---
      self.base_model = VotingClassifier(
          estimators=[
              ('xgb', clf_xgb), 
              ('lgbm', clf_lgbm), 
              ('rf', clf_rf)
          ],
          voting='soft',
          weights=[2.57, 2.18, 1.33], 
          n_jobs=1 
      )
      
      return


   def _get_thread_sorted_event_sequence(self, data : Data, thread_node_idx : int ) -> List[str]:
      edge_src_node_indices = data.edge_index[0]
      edge_tar_node_indices = data.edge_index[1]
      outgoing_edges_from_thread = torch.nonzero( edge_src_node_indices == thread_node_idx ).flatten() 
      incoming_edges_to_thread = torch.nonzero( edge_tar_node_indices == thread_node_idx ).flatten() 
      
      edge_feats_of_outgoing_edges_from_thread = data.edge_attr[ outgoing_edges_from_thread ]
      edge_feats_of_incoming_edges_to_thread = data.edge_attr[ incoming_edges_to_thread ]
      edge_feats_of_all_edges_of_thread = torch.cat([edge_feats_of_incoming_edges_to_thread, edge_feats_of_outgoing_edges_from_thread], dim = 0) 
      
      sort_by_timestamp = torch.argsort( edge_feats_of_all_edges_of_thread[:, -1], descending=False ) 
      edge_feats_of_all_edges_of_thread__sorted = edge_feats_of_all_edges_of_thread[ sort_by_timestamp ]

      eventname_indices = torch.nonzero( edge_feats_of_all_edges_of_thread__sorted[:,:-1], as_tuple=False)[:, -1]
      thread_sorted_event_sequence = [ self.eventname_edgefeats[i] for i in eventname_indices ]
      
      return thread_sorted_event_sequence


   def _get_thread_neighboring_nodetypes(self, data : Data, thread_node_idx : int ) -> torch.tensor:
      edge_src_node_indices = data.edge_index[0]
      edge_tar_node_indices = data.edge_index[1]
      outgoing_edges_from_thread = torch.nonzero( edge_src_node_indices == thread_node_idx ).flatten()
      incoming_edges_to_thread = torch.nonzero( edge_tar_node_indices == thread_node_idx ).flatten()
      
      dst_of_outgoing_edges = data.edge_index[1, outgoing_edges_from_thread]
      src_of_incoming_edges = data.edge_index[0, incoming_edges_to_thread]
      
      all_interacting_neighbors = torch.cat([dst_of_outgoing_edges, src_of_incoming_edges])
      
      nodetype_featvects = data.x
      
      if len(all_interacting_neighbors) > 0:
          thread_neighboring_nodetypes = torch.sum( nodetype_featvects[ all_interacting_neighbors ], dim = 0 ).view(1,-1)
      else:
          thread_neighboring_nodetypes = torch.zeros((1, nodetype_featvects.size(1)))

      return thread_neighboring_nodetypes


   def fit_count_vectorizer(self, train_dataset : List[Data]) -> None:
      thread_nodetype = torch.tensor([1 if _type.lower() == "thread" else 0 for _type in self.nodetype_nodefeats])
      all_thread_level_event_sequences = []

      cnt = 1
      for train_data in train_dataset:            
         nodetype_featvects = train_data.x
         thread_node_indices = torch.nonzero( torch.all( torch.eq( nodetype_featvects, thread_nodetype ), dim=1 ), as_tuple=False).flatten()

         for thread_node_idx in thread_node_indices.tolist():
            thread_sorted_event_sequence = self._get_thread_sorted_event_sequence( data = train_data, thread_node_idx = thread_node_idx)
            all_thread_level_event_sequences.append( thread_sorted_event_sequence )
         
         if cnt % 50 == 0: 
             print(f"{cnt} / {len(train_dataset)}: Processed sequences...", flush = True)
         cnt += 1
   
      all_thread_level_event_sequences_for_fitting = [ ' '.join(thread_event_seq) for thread_event_seq in all_thread_level_event_sequences if len(thread_event_seq) >= 1 ] 
      
      print("Fitting Aggressive TF-IDF Vectorizer...", flush=True)
      self.count_vectorizer.fit( all_thread_level_event_sequences_for_fitting )
      print(f"fitted vectorizer. Vocab size: {len(self.count_vectorizer.vocabulary_)}", flush = True)
      return 


   def generate_graph_embedding(self, data : Data) -> torch.tensor:
      thread_nodetype = torch.tensor([1 if _type.lower() == "thread" else 0 for _type in self.nodetype_nodefeats])
      nodetype_featvects = data.x
      thread_node_indices = torch.nonzero( torch.all(torch.eq( nodetype_featvects, thread_nodetype ), dim=1), as_tuple=False).flatten()

      all_thread_node_embeddings = torch.tensor([]) 

      for thread_node_idx in thread_node_indices.tolist():
         thread_sorted_event_sequence = self._get_thread_sorted_event_sequence( data = data, thread_node_idx = thread_node_idx )
         thread_sorted_event_sequence_for_transform = " ".join(thread_sorted_event_sequence)
         
         try:
             thread_Ngram_count_vector = self.count_vectorizer.transform( [ thread_sorted_event_sequence_for_transform ] ).toarray()
             thread_Ngram_count_tensor = torch.Tensor( thread_Ngram_count_vector ).view(1,-1)
         except ValueError:
             thread_Ngram_count_tensor = torch.zeros((1, len(self.count_vectorizer.get_feature_names_out()))).view(1,-1)

         thread_neighboring_nodetypes_tensor = self._get_thread_neighboring_nodetypes( data = data, thread_node_idx = thread_node_idx )
         thread_node_embedding = torch.cat( [thread_neighboring_nodetypes_tensor, thread_Ngram_count_tensor], dim = 1)
         all_thread_node_embeddings = torch.cat( ( all_thread_node_embeddings, thread_node_embedding ) , dim = 0 )

      if all_thread_node_embeddings.size(0) == 0:
          feature_dim = len(self.nodetype_nodefeats) + len(self.count_vectorizer.get_feature_names_out())
          return torch.zeros(feature_dim)

      graph_embedding = self.pool(all_thread_node_embeddings, dim = 0)            
      return graph_embedding


   def fit(self, train_dataset : List[Data], nodetype_nodefeats : List[str], eventname_edgefeats : List[str]) -> None:
      self.nodetype_nodefeats, self.eventname_edgefeats = nodetype_nodefeats, eventname_edgefeats
      self.fit_count_vectorizer( train_dataset )

      train_data_dict = dict()
      cnt = 1
      print("Generating graph embeddings...", flush=True)
      for train_data in train_dataset:
         if cnt % 50 == 0:
            print(f"{cnt} / {len(train_dataset)} processed", flush = True)
         train_data_graph_embedding = self.generate_graph_embedding( train_data )
         train_data_dict[ train_data.name ] = train_data_graph_embedding.tolist()
         cnt+=1

      X = list( train_data_dict.values() )
      y = [ 1 if "malware" in data_name.lower() else 0 for data_name in train_data_dict.keys() ]
      
      # --- SMOTE Augmentation Logic ---
      print(f"Original Dataset Size: {len(y)} | Malware: {sum(y)}", flush=True)
      print("Applying SMOTE Augmentation...", flush=True)
      
      try:
          smote = SMOTE(random_state=42)
          X_resampled, y_resampled = smote.fit_resample(np.array(X), np.array(y))
          print(f"Augmented Dataset Size: {len(y_resampled)} | Malware: {sum(y_resampled)}", flush=True)
      except Exception as e:
          print(f"SMOTE Failed (likely due to small data), using original. Error: {e}")
          X_resampled, y_resampled = np.array(X), np.array(y)
      
      print("Fitting Weighted Ensemble (XGB + LGBM + RF) on Augmented Data...", flush=True)
      
      self.base_model.fit(X = X_resampled, y = y_resampled)
      
      print(f"DONE! Models fitted.", flush = True)
      return   

   def predict(self, test_data : Data):
      test_data_graph_embedding = self.generate_graph_embedding( test_data )
      embedding_list = test_data_graph_embedding.tolist()
      return self.base_model.predict( [embedding_list] ).item()

   def get_feature_names(self):
       # Combine neighbor node types + N-gram features
       ngram_features = self.count_vectorizer.get_feature_names_out().tolist()
       all_features = self.nodetype_nodefeats + ngram_features
       return all_features