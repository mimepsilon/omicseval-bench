import os
from omicseval.data.data_loader import HomologyContextLoader
from omicseval.models.scientist_agent import LongevityDiscoveryAgent

def main():
    print("🚀 Initializing Local ScientistTwo Autonomous Longevity Workbench...")
    
    # 1. Enforce verification of the real-world benchmark data layer
    data_file = "real_multiomics.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"❌ '{data_file}' not found. Please run 'python fetch_gex_data.py' "
            "first to stream the open-access biological tissue matrices."
        )
    
    # 2. Configure target structural feature spaces
    genes = [f"gene_{i}" for i in range(1, 11)]
    contexts = ["tissue_type", "batch_id"]
    
    # 3. Extract context-aware zero-leakage dataset matrices
    loader = HomologyContextLoader(data_file)
    train_set, test_set = loader.split_by_context_ood(
        holdout_context_val="Brain",
        context_col="tissue_type",
        feature_cols=genes,
        target_col="target_age",
        all_context_cols=contexts
    )
    
    print(f"📦 Data Partition Complete:")
    print(f"   ↳ Train baseline cohort (Non-Brain tissue): {len(train_set)} samples")
    print(f"   ↳ Test evaluation cohort (Pure OOD Brain tissue): {len(test_set)} samples")
    
    # 4. Initialize the offline local multi-agent discovery loop
    agent = LongevityDiscoveryAgent(train_set=train_set, test_set=test_set)
    
    # 5. Launch the self-correcting research run (3 complete closed-loop cycles)
    agent.run_autonomous_discovery_cycle(total_iterations=3)
    
    print("\n🏁 Autonomous ScientistTwo discovery session finalized successfully.")

if __name__ == "__main__":
    main()

