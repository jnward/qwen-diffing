# %%
# %load_ext autoreload
# %autoreload 2

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
device = "cuda"

torch.manual_seed(42)

# %%
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", device_map=device, torch_dtype=torch.bfloat16)

r1_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
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

# %%

# %%
from crosscoder import BatchTopKCrosscoder
from transformers import get_constant_schedule_with_warmup
from latent_tracker import LatentTracker
import wandb
import torch.optim as optim

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

optimizer = optim.Adam(crosscoder.parameters(), lr=1e-4)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
run = wandb.init(
    project="qwen-crosscoder",
    name=f"crosscoder-layer{layer_num}_{n_features}_{k}_fullshuffle_SNAI_aux",
)
latent_tracker = LatentTracker(
    n_features,
    device=device,
    dead_threshold=10_000_000,
)

# %%
from data_utils import cached_activation_generator
from tqdm import tqdm
import torch.nn.functional as F

# training loop
num_optimizer_steps = 48_000
virtual_batch_size = 8192
actual_batch_size = 512
accumulation_steps = virtual_batch_size // actual_batch_size
total_forward_passes = num_optimizer_steps * accumulation_steps
alpha = 1/32
crosscoder_path = f"crosscoder-layer{layer_num}_{n_features}_{k}_fullshuffle_aux.pt"

my_data_generator = cached_activation_generator(
    base_model=base_model,
    finetune_model=r1_model,
    tokenizer=r1_tokenizer,
    dataset=dataset,
    layer_num=layer_num,
    activation_batch_size=actual_batch_size,
    generator_batch_size=64,
    acts_per_run=100_000,
    ctx_len=1024,
    skip_first_n_tokens=1, # skip BOS
)

def auxiliary_loss(dead_latents, error, cc, kaux=512):
    """Calculate auxiliary loss using dead latents"""
    if not dead_latents.any():
        return torch.tensor(0.0, device=error.device)
    
    # Get pre-activations for dead latents only
    with torch.no_grad():
        pre_acts = cc.get_latent_activations(error)  # Get all pre-activations
        values = pre_acts * cc.W_decoder_FZ.norm(dim=1)
        dead_values = values[:, dead_latents]  # Select only dead latents
        dead_values = F.relu(dead_values)
    
    # Get top kaux dead latents
    k = min(kaux, dead_values.shape[1])
    top_k_values, top_k_indices = torch.topk(dead_values, k, dim=1)
    threshold = top_k_values[..., -1, None]
    mask = dead_values >= threshold
    
    # Only reconstruct using masked dead pre-activations
    masked_features = torch.zeros_like(pre_acts)
    masked_features[:, dead_latents] = pre_acts[:, dead_latents] * mask
    
    # Reconstruct error using dead latents
    error_reconstruction = cc.decode(masked_features)
    
    # Calculate MSE
    error_mse = F.mse_loss(error_reconstruction, error)
    
    return error_mse

d_model = r1_model.config.hidden_size

pbar = tqdm(range(num_optimizer_steps))
for optimizer_step in pbar:
    optimizer.zero_grad()
    for acc_step in range(accumulation_steps):
        try:
            target = next(my_data_generator).to(device).to(torch.float32)
        except TypeError as e:
            print(f"Invalid data encountered: {e}")
            print("Attempting to load a different batch...")
            while True:
                try:
                    target = next(my_data_generator).to(device).to(torch.float32)
                    break
                except TypeError as e:
                    print(f"Invalid data encountered: {e}")
                    print("Attempting to load a different batch...")

        out = crosscoder.forward(target)
        reconstruction = out["recon"]
        features = out["sparse_activations"]
        error = target - reconstruction

        base_loss = F.mse_loss(reconstruction[..., d_model:], target[..., d_model:])
        r1_loss = F.mse_loss(reconstruction[..., :d_model], target[..., :d_model])

        main_loss = (base_loss + r1_loss) / accumulation_steps

        latent_tracker.update(features)
        dead_latents = latent_tracker.get_dead_latents()

        aux_loss = auxiliary_loss(dead_latents, error, crosscoder) / accumulation_steps
        # aux_loss = torch.zeros_like(main_loss)
        
        # Combined loss
        loss = main_loss + alpha * aux_loss
        # loss = main_loss
    
        # if torch.isnan(aux_loss):
        #     loss = main_loss  # Zero out aux loss if NaN
        loss.backward()

        if acc_step == 0:

            e = reconstruction - target
            total_variance = (target - target.mean(0)).pow(2).sum()
            squared_error = e.pow(2)
            fvu = squared_error.sum() / total_variance

            run_data = {
                    "main_loss": main_loss.item() * accumulation_steps,
                    "loss": loss.item() * accumulation_steps,
                    "base_loss": base_loss.item(),
                    "r1_loss": r1_loss.item(),
                    "aux_loss": aux_loss.item() * accumulation_steps,
                    "fvu": fvu,
                    "dead_latents": dead_latents.sum().item(),
                }
            pbar.set_description(
                f"Loss: {main_loss.item() * accumulation_steps:.4f}, "
                f"fvu: {fvu:.4f}"
            )
            run.log(run_data, step=optimizer_step)

    optimizer.step()
    scheduler.step()

    if optimizer_step and optimizer_step % 1000 == 0 or optimizer_step == num_optimizer_steps-1:
        print(f"Saving crosscoder to {crosscoder_path}")
        torch.save(crosscoder.state_dict(), crosscoder_path)

run.finish()
# %%
