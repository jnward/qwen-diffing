import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
from datasets import load_dataset, Features, Value
from data_utils import get_activations, cached_activation_generator
from crosscoder import BatchTopKCrosscoder
from nnterp import load_model as nnterp_load_model
from nnsight.modeling.language import LanguageModel

def load_crosscoder_model(model_path: str) -> BatchTopKCrosscoder:
    """Load the trained crosscoder model"""
    # Add PosixPath to safe globals for loading
    from pathlib import PosixPath
    import torch.serialization
    torch.serialization.add_safe_globals([PosixPath])
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Extract model parameters
    d_model = checkpoint['W_encoder_ZF'].shape[0] // 2  # Divide by 2 since it's concatenated
    dict_size = checkpoint['W_encoder_ZF'].shape[1]
    k = checkpoint.get('k', 40)  # Default k if not specified
    
    # Create model and load state
    model = BatchTopKCrosscoder(d_model=d_model, dict_size=dict_size, k=k)
    model.load_state_dict(checkpoint)
    
    return model

def analyze_decoder_norm_diffs(decoder_A: torch.Tensor, decoder_B: torch.Tensor) -> np.ndarray:
    """
    Analyze decoder norm differences to understand feature exclusivity.
    Returns values between -1 (base model only) and 1 (R1 model only).
    
    Args:
        decoder_A: Decoder weights for base model [dict_size, d_model]
        decoder_B: Decoder weights for R1 model [dict_size, d_model]
        
    Returns:
        Array of norm differences, normalized to [-1, 1]
    """
    # Calculate norms across the d_model dimension
    norm_A = torch.norm(decoder_A, dim=1)  # Shape: [dict_size]
    norm_B = torch.norm(decoder_B, dim=1)  # Shape: [dict_size]
    
    # Calculate normalized difference
    # (B-A)/(B+A) will be:
    # 1 when B >> A (R1 only)
    # -1 when A >> B (base only)
    # 0 when A ≈ B (shared)
    norm_diff = (norm_B - norm_A) / (norm_B + norm_A + 1e-10)
    print(f"Norm diff shape (Tensor): {norm_diff.shape}")
    
    return norm_diff.cpu().numpy()

def plot_feature_analysis(results: Dict, output_dir: Path):
    """Create visualizations of the analysis results"""
    # Set the style for all plots
    plt.style.use('seaborn')
    
    # Plot decoder norm differences with improved styling
    plt.figure(figsize=(12, 8))
    sns.histplot(data=results['norm_diffs'])
    plt.title('Distribution of Features Between Base and R1 Models', fontsize=16, pad=20)
    plt.xlabel('Decoder norm diff (1 = R1-only, -1 = Base-only)', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    
    # Add gridlines
    plt.grid(True, alpha=0.3)
    
    # Set background color
    plt.gca().set_facecolor('#F0F2F6')  # Light blue background
    plt.gcf().set_facecolor('white')
    
    # Add a vertical line at x=0
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI for better quality
    plt.savefig(output_dir / 'feature_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Setup
    model_path = "crosscoder-layer14.pt"
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load crosscoder model
    crosscoder = load_crosscoder_model(model_path)
    print("Loaded crosscoder model")
    
    # Get decoder weights
    decoder_weights = crosscoder.W_decoder_FZ.data  # Shape: [dict_size, 2*d_model]
    print(f"Decoder weights shape: {decoder_weights.shape}")
    d_model = decoder_weights.shape[1] // 2
    
    # Split decoder weights for base and R1 models
    decoder_A = decoder_weights[:, :d_model]  # Shape: [dict_size, d_model]
    decoder_B = decoder_weights[:, d_model:]  # Shape: [dict_size, d_model]
    
    # Calculate decoder norm differences
    norm_diffs = analyze_decoder_norm_diffs(decoder_A, decoder_B)
    
    # Compile results
    results = { 'norm_diffs': norm_diffs }
    
    # Plot results
    plot_feature_analysis(results, output_dir)
    
    # Save numerical results
    results_json = {
        'norm_diff_stats': {
            'mean': float(np.mean(norm_diffs)),
            'median': float(np.median(norm_diffs)),
            'std': float(np.std(norm_diffs)),
            'r1_exclusive_count': int(np.sum(norm_diffs > 0.95)),  # Features mostly in R1
            'base_exclusive_count': int(np.sum(norm_diffs < -0.95)),  # Features mostly in base
            'shared_count': int(np.sum(np.abs(norm_diffs) <= 0.95))  # Shared features
        }
    }
    
    # Save results
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

if __name__ == "__main__":
    main() 