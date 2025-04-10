# %%
import torch
import plotly.express as px
from crosscoder import BatchTopKCrosscoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import plotly.express as px
import pandas as pd
import numpy as np
import heapq
from tqdm import tqdm
device = "cuda"

torch.manual_seed(42)
torch.set_grad_enabled(False)

# %%
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", device_map=device, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
r1_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map=device, torch_dtype=torch.bfloat16)
# %%
# dataset = load_dataset(
#     "a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
#     "am_0.5M",
#     split="train",
#     streaming=False,
# )
dataset = load_dataset(
    "ServiceNow-AI/R1-Distill-SFT",
    "v1",
    split="train",
    streaming=False,
)
dataset = dataset.shuffle(seed=42)

# n_features = 16384
n_features = 49152
k = 100
layer_num = 14

crosscoder = BatchTopKCrosscoder(
    d_model=r1_model.config.hidden_size,
    dict_size=n_features,
    k=k,
)
crosscoder.to(device)


# %%
state_dict = torch.load("crosscoder-layer14_49152_100_fullshuffle_aux.pt")
crosscoder.load_state_dict(state_dict)

# %%
crosscoder

# %%
from data_utils import cached_activation_generator, token_iter

# batch_size = 512
# my_data_generator = cached_activation_generator(
#     base_model=base_model,
#     finetune_model=r1_model,
#     tokenizer=tokenizer,
#     dataset=dataset,
#     layer_num=layer_num,
#     activation_batch_size=batch_size,
#     generator_batch_size=64,
#     acts_per_run=100_000,
#     ctx_len=1024,
#     skip_first_n_tokens=1, # skip BOS
#     return_tokens=True,
# )

my_data_generator = token_iter(
    tokenizer=tokenizer,
    dataset=dataset,
    batch_size=1,
    ctx_len=1024,
)

example = next(iter(my_data_generator))
tokenizer.decode(example[0])
# %%
base_decoder = crosscoder.W_decoder_FZ[:, :1536]
r1_decoder = crosscoder.W_decoder_FZ[:, 1536:]

# %%
base_decoder.norm(dim=1)
# %%
r1_decoder.norm(dim=1)
# %%
norm_diff = r1_decoder.norm(dim=1) - base_decoder.norm(dim=1)
fig = px.histogram(
    norm_diff.cpu().detach().numpy(),
    marginal="box",
    nbins=200,
    labels={'value': 'Feature Type', 'count': 'Number of Features'},
    title='Distribution of Features Between Base and R1 Models',
    color_discrete_sequence=['#636EFA']
)
fig.update_layout(
    xaxis_title='Decoder norm diff (1 = R1-only, -1 = Base-only)',
    yaxis_title='Number of Features',
    # bargap=0.1,
    showlegend=False
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.show()

# %%
cosine_sim = torch.nn.functional.cosine_similarity(base_decoder, r1_decoder, dim=1)
fig = px.histogram(
    cosine_sim.cpu().detach().numpy(),
    nbins=200,
    marginal="box",
    labels={'value': 'Cosine Similarity', 'count': 'Number of Features'},
    title='Distribution of Decoder Cosine Similarity Between Base and R1 Models',
)
fig.update_layout(
    xaxis_title='Cosine Similarity',
    yaxis_title='Number of Features',
    # bargap=0.1,
    showlegend=False
)
fig.show()

# %%
fig = px.scatter(
    x=norm_diff.cpu().detach().numpy(),
    y=cosine_sim.cpu().detach().numpy(),
    labels={'x': 'Decoder norm diff', 'y': 'Cosine similarity'},
    title='Decoder Cosine Similarity vs Norm Difference',
    opacity=0.5
)
fig.update_layout(
    xaxis_title='Decoder norm diff (1 = R1-only, -1 = Base-only)',
    yaxis_title='Cosine similarity',
    showlegend=False
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.show()
# %%
r1_model
# %%
def generate_feature_df(
    base_model,
    r1_model,
    crosscoder: BatchTopKCrosscoder,
    tokenizer,
    dataset,
    layer_num,
    batch_size=16,
    ctx_len=1024,
):
    my_data_generator = token_iter(
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        ctx_len=ctx_len,
    )
    for batch in my_data_generator:
        batch = batch.to(device)
        base_out = base_model.forward(batch, output_hidden_states=True)
        r1_out = r1_model.forward(batch, output_hidden_states=True)
        base_hidden = base_out.hidden_states[14][:, 1:]  # remove BOS
        r1_hidden = r1_out.hidden_states[14][:, 1:]  # remove BOS
        concat_hidden = torch.cat([base_hidden, r1_hidden], dim=-1)
        flat_hidden = concat_hidden.view(-1, concat_hidden.shape[-1]).float()
        print(concat_hidden.shape)
        print(batch.shape)
        features = crosscoder.get_latent_activations(flat_hidden)
        features = features.view(batch.shape[0], -1, features.shape[-1])
        print(features.shape)
        for i in range(batch.shape[0]):
            for j in range(batch.shape[1]):
                token = tokenizer.decode(batch[i, j])
                ctx = tokenizer.decode(batch[i, j-10:j+10])
                # print(repr(token), repr(ctx))
        break

generate_feature_df(
    base_model,
    r1_model,
    crosscoder,
    tokenizer,
    dataset,
    layer_num,
)
# %%

# %%
def track_feature_activations(
    base_model,
    r1_model,
    crosscoder: BatchTopKCrosscoder,
    tokenizer,
    dataset,
    layer_num,
    batch_size=16,
    ctx_len=1024,
    max_examples=10000,
    top_k=10
):
    print(f"Tracking activations for {max_examples} examples with batch size {batch_size}")
    print(f"Using {crosscoder.dict_size} features, keeping top {top_k} examples per feature")
    
    # Initialize feature data structures
    activation_counts = np.zeros(crosscoder.dict_size, dtype=int)
    
    # Calculate norms of base and r1 decoder vectors
    base_decoder = crosscoder.W_decoder_FZ[:, :1536]
    r1_decoder = crosscoder.W_decoder_FZ[:, 1536:]
    
    base_norms = base_decoder.norm(dim=1).cpu().numpy()
    r1_norms = r1_decoder.norm(dim=1).cpu().numpy()
    
    # Calculate squared norms for the relative norm difference formula
    base_norms_squared = base_norms ** 2
    r1_norms_squared = r1_norms ** 2
    
    # Calculate relative norm difference using the formula
    # Δnorm(j) = 1/2 * ((‖dr1_j‖² - ‖dbase_j‖²)/max(‖dr1_j‖², ‖dbase_j‖²) + 1)
    max_norms_squared = np.maximum(base_norms_squared, r1_norms_squared)
    relative_norm_diff = 0.5 * ((r1_norms_squared - base_norms_squared) / max_norms_squared + 1)
    
    # Traditional norm difference (for reference)
    raw_norm_diff = r1_norms - base_norms
    
    # Calculate cosine similarity
    cosine_sims = torch.nn.functional.cosine_similarity(
        base_decoder, 
        r1_decoder, 
        dim=1).cpu().numpy()
    
    # Initialize top examples storage
    top_examples = [[] for _ in range(crosscoder.dict_size)]
    
    # Process batches
    my_data_generator = token_iter(
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        ctx_len=ctx_len,
    )
    
    pbar = tqdm(total=max_examples, desc="Processing examples")
    examples_processed = 0
    
    for batch in my_data_generator:
        batch = batch.to(device)
        
        # Get hidden states from both models
        base_out = base_model.forward(batch, output_hidden_states=True)
        r1_out = r1_model.forward(batch, output_hidden_states=True)
        
        base_hidden = base_out.hidden_states[layer_num][:, 1:]  # remove BOS
        r1_hidden = r1_out.hidden_states[layer_num][:, 1:]  # remove BOS
        
        # Concatenate hidden states
        concat_hidden = torch.cat([base_hidden, r1_hidden], dim=-1)
        flat_hidden = concat_hidden.view(-1, concat_hidden.shape[-1]).float()
        
        # Get feature activations (get actual values before applying top-k)
        raw_features = crosscoder.get_latent_activations(flat_hidden)
        
        # Apply batch top-k to get binary features
        features = crosscoder.apply_batchtopk(raw_features.clone())

        # Reshape to match the batch shape
        features = features.view(batch.shape[0], -1, features.shape[-1])
        raw_features = raw_features.view(batch.shape[0], -1, raw_features.shape[-1])

        # Process each token's activations
        for i in range(batch.shape[0]):
            for j in range(1, batch.shape[1]):  # Start from 1 to skip BOS token
                examples_processed += 1
                pbar.update(1)
                
                # Get token and context
                token = tokenizer.decode(batch[i, j].item())
                context_start = max(0, j-7)
                context_end = min(batch.shape[1], j+5)
                context = tokenizer.decode(batch[i, context_start:context_end].tolist())
                
                # Get activated features for this token
                token_features = features[i, j-1].nonzero()
                
                # Handle case of no features
                if token_features.size(0) == 0:
                    continue
                
                # Extract the feature indices
                token_features = token_features.squeeze(-1)
                
                # Handle case where squeeze removes dimension for single feature
                if token_features.dim() == 0:
                    token_features = token_features.unsqueeze(0)
                
                # Update activation counts
                feat_indices = token_features.cpu().numpy()
                activation_counts[feat_indices] += 1
                
                # Update examples for each activated feature
                for feat_id in token_features.cpu():
                    feat_id_int = feat_id.item()
                    
                    # Get the actual activation value from raw_features
                    activation_value = raw_features[i, j-1, feat_id_int].item()
                    
                    example = {
                        'token': token,
                        'context': context,
                        'activation': activation_value
                    }
                    
                    # Just append the example (no sorting by activation strength)
                    top_examples[feat_id_int].append(example)
                    # Keep only top k examples
                    if len(top_examples[feat_id_int]) > top_k:
                        # Sort by activation value and keep top k
                        top_examples[feat_id_int] = sorted(top_examples[feat_id_int], 
                                                         key=lambda x: x['activation'], 
                                                         reverse=True)[:top_k]
                
                if examples_processed >= max_examples:
                    break
            
            if examples_processed >= max_examples:
                break
        
        if examples_processed >= max_examples:
            break
    
    pbar.close()
    
    # Organize examples for each feature
    print("Preparing feature dataframe...")
    all_top_examples = []
    
    for feat_id in tqdm(range(crosscoder.dict_size), desc="Processing features"):
        # Make sure they're sorted by activation
        sorted_examples = sorted(top_examples[feat_id], 
                               key=lambda x: x['activation'], 
                               reverse=True)
        all_top_examples.append(sorted_examples)
    
    # Create the dataframe
    feature_df = pd.DataFrame({
        'feature_id': range(crosscoder.dict_size),
        'activation_count': activation_counts,
        'base_norm': base_norms,
        'r1_norm': r1_norms,
        'raw_norm_diff': raw_norm_diff,
        'relative_norm_diff': relative_norm_diff,
        'cosine_sim': cosine_sims,
        'top_examples': all_top_examples
    })
    
    return feature_df

# %%
# Visualization and analysis functions
def visualize_feature_characteristics(feature_df):
    # Plot distribution of activation counts
    fig = px.histogram(
        feature_df[feature_df['activation_count'] > 0], 
        x='activation_count',
        log_y=True,
        nbins=50,
        title='Distribution of Feature Activation Counts (log scale)'
    )
    fig.update_layout(
        xaxis_title='Activation Count',
        yaxis_title='Number of Features'
    )
    fig.show()
    
    # Plot norm difference vs activation count
    fig = px.scatter(
        feature_df[feature_df['activation_count'] > 0],
        x='raw_norm_diff',
        y='activation_count',
        color='cosine_sim',
        opacity=0.7,
        color_continuous_scale='Viridis',
        title='Feature Activation Count vs Norm Difference'
    )
    fig.update_layout(
        xaxis_title='Norm Difference (R1 - Base)',
        yaxis_title='Activation Count',
        coloraxis_colorbar_title='Cosine Similarity'
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.show()
    
    # Plot base norm vs r1 norm colored by activation count
    # Create a log-scaled column for color
    df_plot = feature_df[feature_df['activation_count'] > 0].copy()
    df_plot['log_activation_count'] = np.log10(df_plot['activation_count'])
    
    fig = px.scatter(
        df_plot,
        x='base_norm',
        y='r1_norm',
        color='log_activation_count',
        opacity=0.7,
        color_continuous_scale='Viridis',
        title='Base Norm vs R1 Norm by Activation Count (log scale)'
    )
    fig.update_layout(
        xaxis_title='Base Norm',
        yaxis_title='R1 Norm',
        coloraxis_colorbar_title='Log10(Activation Count)'
    )
    # Add diagonal line
    max_val = max(df_plot['base_norm'].max(), df_plot['r1_norm'].max())
    fig.add_trace(px.line(x=[0, max_val], y=[0, max_val]).data[0])
    fig.show()

# %%
# Run feature tracking
feature_df = track_feature_activations(
    base_model,
    r1_model,
    crosscoder,
    tokenizer,
    dataset,
    layer_num=layer_num,
    batch_size=8,
    max_examples=10000
)

# %%
# Basic statistics
print(f"Total features: {len(feature_df)}")
print(f"Features with activations: {(feature_df['activation_count'] > 0).sum()}")
print(f"Total activations recorded: {feature_df['activation_count'].sum()}")

# %%
# Visualize feature characteristics
visualize_feature_characteristics(feature_df)


# %%
# Save dataframe
feature_df.to_pickle("feature_activations.pkl")
print("\nDataframe saved to feature_activations.pkl")

# %%
# Additional feature analyses
def analyze_feature_distributions(feature_df, min_activations=10, total_tokens=10000):
    # Filter features that were activated at least min_activations times
    active_features = feature_df[feature_df['activation_count'] >= min_activations]
    print(f"Analyzing {len(active_features)} features with at least {min_activations} activations")
    print(f"Total tokens processed: {total_tokens}")
    
    # Create feature type column based on relative norm diff
    active_features['feature_type'] = 'Mixed'
    active_features.loc[active_features['relative_norm_diff'] <= 0.1, 'feature_type'] = 'Base-only'
    active_features.loc[active_features['relative_norm_diff'] >= 0.9, 'feature_type'] = 'R1-only'
    active_features.loc[(active_features['relative_norm_diff'] >= 0.4) & 
                       (active_features['relative_norm_diff'] <= 0.6), 'feature_type'] = 'Shared'
    
    # Count features by type
    type_counts = active_features['feature_type'].value_counts()
    print("\nFeature Type Distribution:")
    for feat_type, count in type_counts.items():
        print(f"{feat_type}: {count} features ({count/len(active_features)*100:.1f}%)")
    
    # 1. Top R1-only features (highest relative norm difference)
    print("\n===== Top R1-Only Features (relative_norm_diff: 0.9-1.0) =====")
    r1_only = active_features[active_features['relative_norm_diff'] >= 0.9].sort_values('relative_norm_diff', ascending=False).head(10)
    for _, row in r1_only.iterrows():
        pct_active = (row['activation_count'] / total_tokens) * 100
        print(f"\nFeature {row['feature_id']} (relative norm diff: {row['relative_norm_diff']:.4f})")
        print(f"Base norm: {row['base_norm']:.4f}, R1 norm: {row['r1_norm']:.4f}")
        print(f"Activated in {pct_active:.2f}% of tokens, cosine sim: {row['cosine_sim']:.4f}")
        
        # Create tabular display for examples
        examples_data = []
        for example in row['top_examples'][:10]:  # Show top 10 examples
            examples_data.append({
                'Token': example['token'],
                'Activation': f"{example['activation']:.4f}",
                'Context': example['context']
            })
        examples_df = pd.DataFrame(examples_data)
        display(examples_df)
    
    # 2. Top base-only features (lowest relative norm difference)
    print("\n===== Top Base-Only Features (relative_norm_diff: 0.0-0.1) =====")
    base_only = active_features[active_features['relative_norm_diff'] <= 0.1].sort_values('relative_norm_diff', ascending=True).head(10)
    for _, row in base_only.iterrows():
        pct_active = (row['activation_count'] / total_tokens) * 100
        print(f"\nFeature {row['feature_id']} (relative norm diff: {row['relative_norm_diff']:.4f})")
        print(f"Base norm: {row['base_norm']:.4f}, R1 norm: {row['r1_norm']:.4f}")
        print(f"Activated in {pct_active:.2f}% of tokens, cosine sim: {row['cosine_sim']:.4f}")
        
        # Create tabular display for examples
        examples_data = []
        for example in row['top_examples'][:10]:  # Show top 10 examples
            examples_data.append({
                'Token': example['token'],
                'Activation': f"{example['activation']:.4f}",
                'Context': example['context']
            })
        examples_df = pd.DataFrame(examples_data)
        display(examples_df)
    
    # 3. Top shared features (relative norm diff around 0.5)
    print("\n===== Top Shared Features (relative_norm_diff: 0.4-0.6) =====")
    shared = active_features[(active_features['relative_norm_diff'] >= 0.4) & 
                             (active_features['relative_norm_diff'] <= 0.6)].sort_values('activation_count', ascending=False).head(10)
    for _, row in shared.iterrows():
        pct_active = (row['activation_count'] / total_tokens) * 100
        print(f"\nFeature {row['feature_id']} (relative norm diff: {row['relative_norm_diff']:.4f})")
        print(f"Base norm: {row['base_norm']:.4f}, R1 norm: {row['r1_norm']:.4f}")
        print(f"Activated in {pct_active:.2f}% of tokens, cosine sim: {row['cosine_sim']:.4f}")
        
        # Create tabular display for examples
        examples_data = []
        for example in row['top_examples'][:10]:  # Show top 10 examples
            examples_data.append({
                'Token': example['token'],
                'Activation': f"{example['activation']:.4f}",
                'Context': example['context']
            })
        examples_df = pd.DataFrame(examples_data)
        display(examples_df)
    
    # 4. Features with lowest cosine similarity (most different direction)
    print("\n===== Features with Lowest Cosine Similarity =====")
    opposite_features = active_features.sort_values('cosine_sim', ascending=True).head(10)
    for _, row in opposite_features.iterrows():
        pct_active = (row['activation_count'] / total_tokens) * 100
        print(f"\nFeature {row['feature_id']} (cosine sim: {row['cosine_sim']:.4f}, relative norm diff: {row['relative_norm_diff']:.4f})")
        print(f"Base norm: {row['base_norm']:.4f}, R1 norm: {row['r1_norm']:.4f}")
        print(f"Activated in {pct_active:.2f}% of tokens, Type: {row['feature_type']}")
        
        # Create tabular display for examples
        examples_data = []
        for example in row['top_examples'][:10]:  # Show top 10 examples
            examples_data.append({
                'Token': example['token'],
                'Activation': f"{example['activation']:.4f}",
                'Context': example['context']
            })
        examples_df = pd.DataFrame(examples_data)
        display(examples_df)
    
    # 5. Shared features with negative cosine similarity
    print("\n===== Shared Features with Negative Cosine Similarity =====")
    shared_negative = active_features[
        (active_features['feature_type'] == 'Shared') & 
        (active_features['cosine_sim'] < 0)
    ].sort_values('cosine_sim', ascending=True).head(10)
    
    if len(shared_negative) > 0:
        print(f"Found {len(shared_negative)} shared features with negative cosine similarity")
        for _, row in shared_negative.iterrows():
            pct_active = (row['activation_count'] / total_tokens) * 100
            print(f"\nFeature {row['feature_id']} (cosine sim: {row['cosine_sim']:.4f}, relative norm diff: {row['relative_norm_diff']:.4f})")
            print(f"Base norm: {row['base_norm']:.4f}, R1 norm: {row['r1_norm']:.4f}")
            print(f"Activated in {pct_active:.2f}% of tokens")
            
            # Create tabular display for examples
            examples_data = []
            for example in row['top_examples'][:10]:  # Show top 10 examples
                examples_data.append({
                    'Token': example['token'],
                    'Activation': f"{example['activation']:.4f}",
                    'Context': example['context']
                })
            examples_df = pd.DataFrame(examples_data)
            display(examples_df)
    else:
        print("No shared features with negative cosine similarity found")
    
    # Visualize shared features with negative cosine similarity on scatter plot
    if len(shared_negative) > 0:
        # Create scatter plot highlighting shared features with negative cosine similarity
        fig = px.scatter(
            active_features,
            x='relative_norm_diff',
            y='cosine_sim',
            color='feature_type',
            opacity=0.5,
            color_discrete_map={'Base-only': 'blue', 'R1-only': 'red', 'Shared': 'green', 'Mixed': 'gray'},
            title='Relative Norm Difference vs Cosine Similarity',
            hover_data=['feature_id', 'activation_count']
        )
        
        # Highlight shared features with negative cosine similarity
        fig.add_trace(px.scatter(
            shared_negative,
            x='relative_norm_diff',
            y='cosine_sim',
            hover_data=['feature_id', 'activation_count']
        ).data[0])
        
        # Add horizontal line at cosine sim = 0
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        # Add vertical lines for shared feature boundaries (0.4 and 0.6)
        fig.add_vline(x=0.4, line_dash="dash", line_color="green")
        fig.add_vline(x=0.6, line_dash="dash", line_color="green")
        
        fig.update_layout(
            xaxis_title='Relative Norm Difference',
            yaxis_title='Cosine Similarity'
        )
        fig.show()
    
    # Visualize feature distributions
    # Create plot of relative norm difference distribution with feature type coloring
    fig = px.histogram(
        active_features,
        x='relative_norm_diff',
        color='feature_type',
        nbins=50,
        title='Distribution of Relative Norm Differences Between Base and R1',
        color_discrete_map={'Base-only': 'blue', 'R1-only': 'red', 'Shared': 'green', 'Mixed': 'gray'}
    )
    fig.update_layout(
        xaxis_title='Relative Norm Difference',
        yaxis_title='Number of Features',
        bargap=0.1
    )
    # Add vertical lines for category boundaries
    fig.add_vline(x=0.1, line_dash="dash", line_color="black")
    fig.add_vline(x=0.4, line_dash="dash", line_color="black")
    fig.add_vline(x=0.6, line_dash="dash", line_color="black")
    fig.add_vline(x=0.9, line_dash="dash", line_color="black")
    fig.show()
    
    # Create scatter plot of base norm vs r1 norm colored by feature type
    fig = px.scatter(
        active_features,
        x='base_norm',
        y='r1_norm',
        color='feature_type',
        hover_data=['feature_id', 'relative_norm_diff', 'activation_count'],
        opacity=0.7,
        color_discrete_map={'Base-only': 'blue', 'R1-only': 'red', 'Shared': 'green', 'Mixed': 'gray'},
        title='Base Norm vs R1 Norm by Feature Type'
    )
    fig.update_layout(
        xaxis_title='Base Norm',
        yaxis_title='R1 Norm'
    )
    # Add diagonal line
    max_val = max(active_features['base_norm'].max(), active_features['r1_norm'].max())
    fig.add_trace(px.line(x=[0, max_val], y=[0, max_val]).data[0])
    fig.show()

# %%
# Run additional analyses
analyze_feature_distributions(feature_df, min_activations=10, total_tokens=10000)

# %%
problem = "What is the sixth prime factor of 14360?"

# %%
def forward_from_layer(model, hidden_states, layer_idx, attention_mask=None):
    """
    Takes residual stream activations at a specific layer and computes the forward pass
    from that point onwards to get logits.
    
    Args:
        model: HuggingFace model (Qwen2ForCausalLM)
        hidden_states: Tensor of shape (batch, pos, d_model) with residual stream activations
        layer_idx: Index of the layer where hidden_states come from (0-indexed)
        attention_mask: Optional attention mask
    
    Returns:
        logits: The output logits from the model
    """
    batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
    
    # Get the transformer layers
    layers = model.model.layers
    
    # Create position IDs for rotary embeddings
    position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    
    # Create the rotary embeddings
    rotary_emb = model.model.rotary_emb
    
    # Get cos/sin embeddings for the position IDs
    # We need to pass a dummy tensor with correct device instead of None
    with torch.no_grad():
        # Use hidden_states for device, but the actual values don't matter
        cos_sin = rotary_emb(hidden_states, position_ids)
    
    # Convert attention_mask to the right dtype if it exists
    if attention_mask is not None:
        # Convert attention_mask to bool
        attention_mask = attention_mask.to(dtype=torch.bool)
    
    # Skip the layers before layer_idx as we're injecting at layer_idx
    for i in range(layer_idx, len(layers)):
        # Run through each transformer layer
        layer = layers[i]
        
        # Forward pass through the layer with position_embeddings (cos_sin tuple)
        outputs = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=cos_sin,
        )
        
        # Get the hidden states for the next layer
        hidden_states = outputs[0]
    
    # Run through final layer norm
    hidden_states = model.model.norm(hidden_states)
    
    # Project to vocabulary (get logits)
    logits = model.lm_head(hidden_states)
    
    return logits

# %%
# Test the forward_from_layer function
def test_forward_from_layer():
    # Generate a small test batch
    test_text = "Hello, how are you?"
    test_input = tokenizer(test_text, return_tensors="pt").to(device)
    
    # Get the normal model output
    with torch.no_grad():
        normal_outputs = r1_model(test_input.input_ids, output_hidden_states=True)
        normal_logits = normal_outputs.logits
        
        # Get hidden states at a specific layer
        layer_idx = 7  # Choose a middle layer
        hidden_states = normal_outputs.hidden_states[layer_idx + 1]  # +1 to account for embedding layer
        
        # Run forward from that layer
        injected_logits = forward_from_layer(
            r1_model, 
            hidden_states, 
            layer_idx, 
            attention_mask=test_input.attention_mask,
        )
        
        # Compare the outputs (they should be close)
        max_diff = (normal_logits - injected_logits).abs().max().item()
        print(f"Maximum difference between normal and injected logits: {max_diff}")
        
        if max_diff < 1e-5:
            print("Test passed! The forward_from_layer function works correctly.")
        else:
            print("Test failed. The outputs are significantly different.")
            
        # Print shapes for debugging
        print(f"Normal logits shape: {normal_logits.shape}")
        print(f"Injected logits shape: {injected_logits.shape}")
        
        # Compare top predictions
        normal_top_tokens = normal_logits[:,-1].argmax(dim=-1)
        injected_top_tokens = injected_logits[:,-1].argmax(dim=-1)
        print(f"\nTop token predictions for final position:")
        print(f"Normal: {tokenizer.decode(normal_top_tokens)}")
        print(f"Injected: {tokenizer.decode(injected_top_tokens)}")
    
    return normal_logits, injected_logits

# Uncomment to run the test
normal_logits, injected_logits = test_forward_from_layer()


# %%
