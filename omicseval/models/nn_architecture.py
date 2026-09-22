import torch
import torch.nn as nn
from typing import List

class TransformerOmicsNet(nn.Module):
    """
    A Transformer-hybrid architecture that treats individual genes and 
    categorical contexts as distinct 'tokens' in a sequence, applying 
    Self-Attention to learn dynamic, context-dependent feature weightings.
    """
    def __init__(self, 
                 num_omics_features: int, 
                 context_cardinalities: List[int], 
                 d_model: int = 32, 
                 nhead: int = 4, 
                 num_layers: int = 1):
        super().__init__()
        
        self.d_model = d_model
        self.num_genes = num_omics_features
        
        # 1. Linear projection to map each single gene value to a token embedding space
        self.gene_projection = nn.Linear(1, d_model)
        
        # 2. Embedding layers for each categorical context variable
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=d_model)
            for cardinality in context_cardinalities
        ])
        
        # 3. Standard PyTorch Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            dropout=0.1, 
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Downstream prediction head
        total_tokens = num_omics_features + len(context_cardinalities)
        self.regressor = nn.Sequential(
            nn.Linear(total_tokens * d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, omics_features: torch.Tensor, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            omics_features: Float tensor of shape [batch_size, num_omics_features]
            context_indices: Long tensor of shape [batch_size, num_context_variables]
        """
        batch_size = omics_features.size(0)
        tokens = []
        
        # Project each individual gene into its own token slot [batch_size, 1, d_model]
        for i in range(self.num_genes):
            gene_val = omics_features[:, i].unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1]
            tokens.append(self.gene_projection(gene_val)) # [batch_size, 1, d_model]
            
        # Add context embeddings as additional tokens [batch_size, 1, d_model]
        for i, emb_layer in enumerate(self.context_embeddings):
            indices = context_indices[:, i]
            tokens.append(emb_layer(indices).unsqueeze(1))
            
        # Stack tokens into a true sequence matrix: [batch_size, sequence_length, d_model]
        sequence = torch.cat(tokens, dim=1)
        
        # Pass sequence through the self-attention mechanism
        attended_sequence = self.transformer(sequence)
        
        # Flatten the outputs and pass through regression layers
        flat_features = attended_sequence.contiguous().view(batch_size, -1)
        return self.regressor(flat_features)


