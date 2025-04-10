# %%
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
from crosscoder import BatchTopKCrosscoder
device = "cuda"
torch.set_grad_enabled(False)

# %%
model = LanguageModel(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    device_map=device,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# %%
n_features = 49152
k = 100
layer_num = 13  # 14 - 1

crosscoder = BatchTopKCrosscoder(
    d_model=model.config.hidden_size,
    dict_size=n_features,
    k=k,
)
crosscoder.to(device)
state_dict = torch.load("crosscoder-layer14_49152_100_fullshuffle_aux.pt")
crosscoder.load_state_dict(state_dict)

# %%
feature_idx = 153
steering_vector = crosscoder.W_decoder_FZ[feature_idx, 1536:]

# %%
problem = "What is the tallest building in Manhattan?."
prompt = f"<｜end▁of▁sentence｜><｜User｜>{problem}<｜Assistant｜><think>\n"

# prompt = "Jack and Alice went to the store."

with model.generate(prompt, max_new_tokens=100) as trace:
    model.model.layers[layer_num].output[0][:] += steering_vector * 20

    out = model.generator.output.save()

# %%
print(tokenizer.decode(out[0]))

# %%
