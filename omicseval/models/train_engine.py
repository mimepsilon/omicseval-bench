import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omicseval.data.data_loader import ContextOmicsDataset
from omicseval.metrics.stats_engine import compute_concordance_index

class OODTrainEngine:
    """Orchestrates multi-modal training and out-of-distribution model validation loops."""
    
    # FIX: Accept any PyTorch nn.Module instead of importing a specific class name
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, data_loader: DataLoader) -> float:
        """Runs a single training epoch across multi-modal batches."""
        self.model.train()
        total_loss = 0.0
        
        for omics_feats, context_tokens, targets in data_loader:
            omics_feats = omics_feats.to(self.device)
            context_tokens = context_tokens.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            predictions = self.model(omics_feats, context_tokens)
            loss = self.criterion(predictions, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * omics_feats.size(0)
            
        return total_loss / len(data_loader.dataset)

    def evaluate_ood(self, test_dataset: ContextOmicsDataset) -> dict:
        """Evaluates the model on completely unseen biological contexts."""
        self.model.eval()
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        with torch.no_grad():
            for omics_feats, context_tokens, targets in test_loader:
                omics_feats = omics_feats.to(self.device)
                context_tokens = context_tokens.to(self.device)
                
                predictions = self.model(omics_feats, context_tokens).squeeze(1).cpu()
                
                c_index = compute_concordance_index(targets, predictions)
                mae = nn.functional.l1_loss(predictions, targets).item()
                
                return {
                    "OOD_Concordance_Index": c_index,
                    "OOD_MAE": mae,
                    "Predictions": predictions,
                    "Ground_Truth": targets
                }


