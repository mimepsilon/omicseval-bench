import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from omicseval.data.data_loader import HomologyContextLoader
from omicseval.models.nn_architecture import TransformerOmicsNet  # <-- CHANGED IMPORT
from omicseval.models.train_engine import OODTrainEngine
from omicseval.metrics.stats_engine import compute_concordance_index

def main():
    print("🚀 Running Comparative Multi-Modal OOD Evaluation Pipeline...")
    
    genes = [f"gene_{i}" for i in range(1, 11)]
    contexts = ["tissue_type", "batch_id"]
    
    loader = HomologyContextLoader("real_multiomics.csv")
    train_set, test_set = loader.split_by_context_ood(
        holdout_context_val="Brain",
        context_col="tissue_type",
        feature_cols=genes,
        target_col="target_age",
        all_context_cols=contexts
    )
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    
    # Instantiate the new Transformer model
    model = TransformerOmicsNet(
        num_omics_features=len(genes),
        context_cardinalities=[4, 3], 
        d_model=32,
        nhead=4
    )
    
    trainer = OODTrainEngine(model=model, lr=1e-3) # Lower learning rate often better for transformers
    
    print("\n🏋️ [Model 1/2] Optimizing Transformer-Hybrid Sequence Net...")
    for epoch in range(1, 11):
        loss = trainer.train_epoch(train_loader)
        
    nn_metrics = trainer.evaluate_ood(test_set)
    
    print("🌲 [Model 2/2] Training Baseline Random Forest Regressor...")
    X_train_flat = np.hstack([train_set.expressions.numpy(), train_set.contexts.numpy()])
    y_train_flat = train_set.targets.numpy()
    X_test_flat = np.hstack([test_set.expressions.numpy(), test_set.contexts.numpy()])
    y_test_flat = test_set.targets.numpy()
    
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_flat, y_train_flat)
    
    rf_preds = torch.tensor(rf_model.predict(X_test_flat), dtype=torch.float32)
    rf_true = torch.tensor(y_test_flat, dtype=torch.float32)
    
    rf_c_index = compute_concordance_index(rf_true, rf_preds)
    rf_mae = torch.nn.functional.l1_loss(rf_preds, rf_true).item()
    
    print("\n📊 [OUT-OF-DISTRIBUTION BENCHMARK LEADERBOARD]")
    print("-" * 65)
    print(f"{'Model Architecture':<30} | {'OOD C-Index (↑)':<15} | {'OOD MAE (↓)':<12}")
    print("-" * 65)
    print(f"{'Transformer Omics Net (Ours)':<30} | {nn_metrics['OOD_Concordance_Index']:<15.4f} | {nn_metrics['OOD_MAE']:<12.4f}")
    print(f"{'Baseline Random Forest Regressor':<30} | {rf_c_index:<15.4f} | {rf_mae:<12.4f}")
    print("-" * 65)

if __name__ == "__main__":
    main()

