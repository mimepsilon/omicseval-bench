import torch
from torch.utils.data import DataLoader
from omicseval.data.data_loader import ContextOmicsDataset
from omicseval.models.nn_architecture import TransformerOmicsNet
from omicseval.models.train_engine import OODTrainEngine

class BiologicalMockLLM:
    """Simulates a domain-specific longevity AI that proposes biological target hypotheses."""
    
    def generate_biological_hypothesis(self, iteration: int) -> dict:
        # Instead of positional numbers, we slice indices based on real-world pathway boundaries
        if iteration == 1:
            return {
                "target_indices": list(range(0, 5)), # Maps directly to MTOR pathway metrics
                "rationale": "Evaluating if highly active metabolic networks (mTOR/AKT) contain structural biological aging signatures conserved across organs."
            }
        elif iteration == 2:
            return {
                "target_indices": list(range(5, 11)), # Maps directly to DNA Repair and Senescence
                "rationale": "Prior loop proves metabolic features fail under OOD tissue shifts. Testing if cell-cycle checkpoint and genomic stability markers (p53/Sirtuins) maintain an invariant clock."
            }
        else:
            return {
                "target_indices": list(range(0, 14)), # Combines all networks
                "rationale": "Integrating metabolic flux variables with genomic degradation markers to stabilize cross-context trajectory predictions."
            }

class LongevityDiscoveryAgent:
    def __init__(self, train_set: ContextOmicsDataset, test_set: ContextOmicsDataset):
        self.llm = BiologicalMockLLM()
        self.train_set = train_set
        self.test_set = test_set

    def run_autonomous_discovery_cycle(self, total_iterations: int = 3):
        print("🧬 Deploying pipeline to Human Genetic Pathway Verification Matrix...")
        
        for iteration in range(1, total_iterations + 1):
            print(f"\n🤖 [ScientistTwo Loop] Starting Discovery Iteration {iteration}/{total_iterations}...")
            
            hypothesis = self.llm.generate_biological_hypothesis(iteration)
            print(f"   💡 Proposed Longevity Target: {hypothesis['rationale']}")
            
            # Splitting tensors based on actual biological pathway mappings
            train_expr_masked = self.train_set.expressions[:, hypothesis["target_indices"]]
            test_expr_masked = self.test_set.expressions[:, hypothesis["target_indices"]]
            
            train_loader = DataLoader(
                ContextOmicsDataset(train_expr_masked, self.train_set.contexts, self.train_set.targets),
                batch_size=32, shuffle=True
            )
            
            cardinalities = [4, 3] # Fixed tissue/batch constraints
            
            model = TransformerOmicsNet(
                num_omics_features=len(hypothesis["target_indices"]),
                context_cardinalities=cardinalities, 
                d_model=16, nhead=2
            )
            
            trainer = OODTrainEngine(model=model, lr=0.001)
            
            for _ in range(8):
                _ = trainer.train_epoch(train_loader)
                
            masked_test_set = ContextOmicsDataset(test_expr_masked, self.test_set.contexts, self.test_set.targets)
            metrics = trainer.evaluate_ood(masked_test_set)
            
            print(f"   📊 Performance Result -> OOD C-Index: {metrics['OOD_Concordance_Index']:.4f} | MAE: {metrics['OOD_MAE']:.4f} years")

