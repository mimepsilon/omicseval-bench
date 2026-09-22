import pandas as pd
import torch
import numpy as np 
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

class ContextOmicsDataset(torch.utils.data.Dataset):
    def __init__(self, expressions: torch.Tensor, contexts: torch.Tensor, targets: torch.Tensor):
        self.expressions = expressions
        self.contexts = contexts
        self.targets = targets

    def __len__(self) -> int:
        return self.expressions.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.expressions[idx], self.contexts[idx], self.targets[idx]

class HomologyContextLoader:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        
    import numpy as np  # Ensure numpy is imported at the top of data_loader.py

    def _encode_categorical_context(self, context_cols: List[str]) -> torch.Tensor:
        """Converts structural text contexts into integer class tokens smoothly."""
        encoded_list = []
        for col in context_cols:
            encoded_list.append(self.df[col].astype('category').cat.codes.values)
    
    # Merge into a single continuous numpy array first to optimize memory allocation
        encoded_numpy = np.stack(encoded_list, axis=0)
        return torch.tensor(encoded_numpy, dtype=torch.long).T


    def split_by_context_ood(self, holdout_context_val: str, context_col: str, 
                             feature_cols: List[str], target_col: str, 
                             all_context_cols: List[str]) -> Tuple[ContextOmicsDataset, ContextOmicsDataset]:
        test_mask = self.df[context_col] == holdout_context_val
        train_mask = ~test_mask
        
        X_train_raw = self.df.loc[train_mask, feature_cols].values
        X_test_raw = self.df.loc[test_mask, feature_cols].values
        
        # Prevent data leakage across biological context partitions
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        contexts_encoded = self._encode_categorical_context(all_context_cols)
        
        y_train = torch.tensor(self.df.loc[train_mask, target_col].values, dtype=torch.float32)
        y_test = torch.tensor(self.df.loc[test_mask, target_col].values, dtype=torch.float32)
        
        return (
            ContextOmicsDataset(torch.tensor(X_train_scaled, dtype=torch.float32), contexts_encoded[train_mask], y_train),
            ContextOmicsDataset(torch.tensor(X_test_scaled, dtype=torch.float32), contexts_encoded[test_mask], y_test)
        )

