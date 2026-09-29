import os
from omicseval.data.data_loader import HomologyContextLoader
from omicseval.models.scientist_agent import LongevityDiscoveryAgent

def main():
    print("🚀 Initializing ScientistTwo Autonomous Longevity Workbench...")
    
    data_file = "real_multiomics.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError("❌ Please execute 'python fetch_gex_data.py' first.")
    
    # Explicitly list the real human genes tracked in the upgraded data layer
    metabolic_mtor = ["MTOR", "RPTOR", "RICTOR", "AKT1", "PIK3CA"]
    dna_repair     = ["CDKN2A", "TP53", "SIRT1", "SIRT6", "PARP1", "ATM"]
    housekeeping   = ["ACTB", "GAPDH", "RPL13A"]
    
    all_features = metabolic_mtor + dna_repair + housekeeping
    contexts = ["tissue_type", "batch_id"]
    
    loader = HomologyContextLoader(data_file)
    train_set, test_set = loader.split_by_context_ood(
        holdout_context_val="Brain",
        context_col="tissue_type",
        feature_cols=all_features, # Pulls your real gene strings cleanly
        target_col="target_age",
        all_context_cols=contexts
    )
    
    print(f"📦 Active Human Cohort Split Complete (Held out OOD: 'Brain' Tissue)")
    agent = LongevityDiscoveryAgent(train_set=train_set, test_set=test_set)
    agent.run_autonomous_discovery_cycle(total_iterations=3)

if __name__ == "__main__":
    main()

