# %%
%load_ext autoreload
%autoreload 2

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
device = "mps"

# %%
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", device_map=device, torch_dtype=torch.bfloat16)

r1_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
r1_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map=device, torch_dtype=torch.bfloat16)
# %%
dataset = load_dataset(
    "a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
    "am_0.9M",
    split="train",
    streaming=True,
)

# %%
from data_utils import token_iter
my_token_iter = token_iter(r1_tokenizer, dataset, 16, 1024)
example_tokens = next(my_token_iter)

# %%
example_tokens.shape
# %%
r1_tokenizer.decode(example_tokens[0])
# %%
from crosscoder import BatchTopKCrosscoder
crosscoder = BatchTopKCrosscoder(
    d_model=r1_model.config.hidden_size,
    dict_size=16384,
    k=100,
)

crosscoder.to(device)
crosscoder


# %%
crosscoder.W_encoder_ZF.shape
# %%
crosscoder.W_decoder_FZ.shape
# %%
example_tokens = example_tokens.to(device)
test_act = r1_model.forward(example_tokens, output_hidden_states=True).hidden_states[-1]
# %%
test_act.shape
# %%
