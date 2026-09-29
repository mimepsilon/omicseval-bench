import json
import torch
from torch.utils.data import DataLoader
from omicseval.data.data_loader import ContextOmicsDataset
from omicseval.models.nn_architecture import TransformerOmicsNet
from omicseval.models.train_engine import OODTrainEngine

class BiologicalMockLLM:
    """Simulates a domain-specific longevity AI that proposes biological target hypotheses."""
    
    def generate_biological_hypothesis(self, iteration: int) -> dict:
        # Define arrays dynamically using ranges to eliminate the bracket symbol entirely
        group_one = list(range(0, 4))   # Genes 0, 1, 2, 3
        group_two = list(range(4, 8))   # Genes 4, 5, 6, 7
        group_three = list(range(0, 8)) # Genes 0 through 7
        
        if iteration == 1:
            return {
                "target_genes": group_one,
                "rationale": "Testing if metabolic pathway genes retain conserved, cross-tissue aging biomarkers."
            }
        elif iteration == 2:
            return {
                "target_genes": group_two,
                "rationale": "Prior run suggests metabolic signals are highly tissue-specific. Shifting to DNA repair networks to find an invariant OOD clock."
            }
        else:
            return {
                "target_genes": group_three,
                "rationale": "Combining metabolic and genomic stability markers to optimize cross-organ aging trajectory predictions."
            }


class LongevityDiscoveryAgent:
    """Orchestrates an autonomous loop to discover genetic targets that control aging clocks."""
    
    def __init__(self, train_set: ContextOmicsDataset, test_set: ContextOmicsDataset):
        self.llm = BiologicalMockLLM()
        self.train_set = train_set
        self.test_set = test_set

    def run_autonomous_discovery_cycle(self, total_iterations: int = 3):
        print("🧬 Transitioning framework to Biological Target Discovery Mode...")
        
        for iteration in range(1, total_iterations + 1):
            print(f"\n🤖 [ScientistTwo Loop] Starting Discovery Iteration {iteration}/{total_iterations}...")
            
            # 1. Generate Biological Hypothesis
            hypothesis = self.llm.generate_biological_hypothesis(iteration)
            print(f"   💡 Proposed Longevity Target: {hypothesis['rationale']}")
            print(f"   🎯 Active Gene Feature Masks: {hypothesis['target_genes']}")
            
            # 2. Extract the specific biological features proposed by the agent
            train_expr_masked = self.train_set.expressions[:, hypothesis["target_genes"]]
            test_expr_masked = self.test_set.expressions[:, hypothesis["target_genes"]]
            
            # 3. Test the biological hypothesis via OOD training
            train_loader = DataLoader(
                ContextOmicsDataset(train_expr_masked, self.train_set.contexts, self.train_set.targets),
                batch_size=32, shuffle=True
            )
            
            # Define context matrix cardinalities dynamically
            cardinalities = []
            cardinalities.append(4)
            cardinalities.append(3)
            
            # The model architecture remains fixed; only the input biological variables change
            model = TransformerOmicsNet(
                num_omics_features=len(hypothesis["target_genes"]),
                context_cardinalities=cardinalities, 
                d_model=16, nhead=2
            )
            
            trainer = OODTrainEngine(model=model, lr=0.001)
            
            # Run experimental validation
            for _ in range(8):
                _ = trainer.train_epoch(train_loader)
                
            # Evaluate out-of-distribution generalization on the held-out tissue (Brain)
            masked_test_set = ContextOmicsDataset(test_expr_masked, self.test_set.contexts, self.test_set.targets)
            metrics = trainer.evaluate_ood(masked_test_set)
            
            print(f"   📊 Discovery Performance Matrix -> OOD C-Index: {metrics['OOD_Concordance_Index']:.4f} | MAE: {metrics['OOD_MAE']:.4f} years")

