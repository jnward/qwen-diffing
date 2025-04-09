import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from einops import einsum, rearrange

"""
Dimension key:

B: batch_size
D: d_model (model dimension)
F: dict_size (feature/dictionary size)
Z: d_in (hidden dimension - double model dimension for crosscoder: 2*D)
"""

class BatchTopKCrosscoder(nn.Module):
    def __init__(
        self, 
        d_model: int,
        dict_size: int, 
        k: int,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k
        
        self.W_encoder_ZF = nn.Parameter(torch.empty(2 * d_model, dict_size))
        self.b_encoder_F = nn.Parameter(torch.zeros(dict_size))
        
        self.W_decoder_FZ = nn.Parameter(torch.empty(dict_size, 2 * d_model))
        self.b_decoder_Z = nn.Parameter(torch.zeros(2 * d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        initial_weight_DF = torch.empty(self.d_model, self.dict_size)
        nn.init.kaiming_uniform_(initial_weight_DF)
        
        with torch.no_grad():
            # Initialize encoder weight: stack the same initial weights for both parts
            # of the input (Z = 2*D)
            self.W_encoder_ZF[:self.d_model, :] = initial_weight_DF.clone()
            self.W_encoder_ZF[self.d_model:, :] = initial_weight_DF.clone()
            
            # Initialize decoder weight as transpose of encoder
            self.W_decoder_FZ.data = self.W_encoder_ZF.T.data.clone()
    
    def get_latent_activations(self, x_BZ: torch.Tensor) -> torch.Tensor:
        activations_BF = einsum(x_BZ, self.W_encoder_ZF, "b z, z f -> b f") + self.b_encoder_F
        return F.relu(activations_BF)
    
    def apply_batchtopk(self, activations_BF: torch.Tensor) -> torch.Tensor:
        # activations_BF: [batch_size, dict_size]
        batch_size = activations_BF.shape[0]
        
        decoder_norms_F = torch.norm(self.W_decoder_FZ, dim=1)
        
        # Calculate value scores
        value_scores_BF = einsum(activations_BF, decoder_norms_F, "b f, f -> b f")
        
        # Flatten and find top-k
        flat_scores_bF = rearrange(value_scores_BF, "b f -> (b f)")
        
        # Find top k*batch_size activations
        topk_indices = torch.topk(flat_scores_bF, k=self.k * batch_size, dim=0).indices
        
        # Create sparse mask and apply
        mask_bF = torch.zeros_like(flat_scores_bF)
        mask_bF[topk_indices] = 1.0
        mask_BF = rearrange(mask_bF, "(b f) -> b f", b=batch_size)
        
        # Apply mask to get sparse activations
        sparse_activations_BF = activations_BF * mask_BF
        
        return sparse_activations_BF
    
    def decode(self, sparse_activations_BF: torch.Tensor) -> torch.Tensor:
        return einsum(sparse_activations_BF, self.W_decoder_FZ, "b f, f d -> b d") + self.b_decoder_Z
    
    def forward(self, x_BZ: torch.Tensor) -> Dict[str, torch.Tensor]:
        activations_BF = self.get_latent_activations(x_BZ)
        sparse_activations_BF = self.apply_batchtopk(activations_BF)
        recon_BZ = self.decode(sparse_activations_BF)
        
        return {
            "recon": recon_BZ,
            "sparse_activations": sparse_activations_BF,
            "activations": activations_BF,
        }