# %%
import torch
import plotly.express as px
from crosscoder import BatchTopKCrosscoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
device = "cuda"

torch.manual_seed(42)

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
