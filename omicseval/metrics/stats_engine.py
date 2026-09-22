import torch

def compute_ece(labels: torch.Tensor, probabilities: torch.Tensor, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) for binary targets."""
    if labels.shape != probabilities.shape:
        raise ValueError("Tensors must have identical shapes.")
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].float().mean().item()
            avg_confidence_in_bin = probabilities[in_bin].mean().item()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def compute_concordance_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates the pairwise Concordance Index (C-index) for regression."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Tensors must have identical shapes.")
        
    n = y_true.size(0)
    if n < 2:
        return 0.0

    y_true_matrix = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    y_pred_matrix = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)

    valid_pairs = y_true_matrix > 0
    concordant = (y_pred_matrix > 0) & valid_pairs
    ties = (y_pred_matrix == 0) & valid_pairs

    num_valid = valid_pairs.sum().item()
    if num_valid == 0:
        return 0.0

    return (concordant.sum().item() + 0.5 * ties.sum().item()) / num_valid

