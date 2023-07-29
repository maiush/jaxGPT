# TODO: dropout

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import fire
import numpy as np
from tqdm import tqdm

def gpt(inputs, wte, wpe, blocks, ln_f, n_head):
    # Index into token embeddings and add positional encoding.
    x = wte[inputs] + wpe[:len(inputs)]
    # Forward pass through blocks.
    for block in blocks:
        x = decoder_block(x, **block, n_head=n_head)
    # Get logits.
    x = layernorm(x, **ln_f) # (GPT2-style)
    return x @ wte.T

def generate(input, params, n_head, n_output):
    for _ in tqdm(range(n_output), "Generating:"):
        # Forward pass.
        logits = gpt(input, **params, n_head=n_head)
        # Greedy sampling.
        out = np.argmax(logits[-1])
        input.append(int(out))
    return input[-n_output:]

def gelu(x): 
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x + 0.044715 * x**3)))

def softmax(x, temp=1.):
    x = np.exp(x-np.max(x, -1, keepdims=True) / temp)
    return x / np.sum(x, -1, keepdims=True)

def layernorm(x, g, b, eps=1e-6):
    mean = np.mean(x, -1, keepdims=True)
    std = np.std(x, -1, keepdims=True)
    return g * (x-mean)/(std+eps) + b

def linear(x, w, b): 
    return x @ w + b

def decoder_block(x, attn, ln_1, mlp, ln_2, n_head):
    # pre-layernorm -> multi-head causal self-attention -> residual connection
    x += mha(layernorm(x, **ln_1), **attn, n_head=n_head)
    # pre-layernorm -> position-wise MLP -> residual connection
    x += ffnn(layernorm(x, **ln_2), **mlp)
    return x

def ffnn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)
    
def causal_self_attention(q, k, v):
    axs = list(range(k.ndim))
    axs[-1], axs[-2] = axs[-2], axs[-1]
    scores = q @ k.transpose(axs) / np.sqrt(q.shape[-1])
    # Masking for causal attention.
    mask = np.triu(np.ones(scores.shape), k=1) == 0
    return softmax(np.where(mask, scores, -1e9)) @ v

def mha(x, c_attn, c_proj, n_head):
    # QKV projection.
    x = linear(x, **c_attn)
    # Split into QKV.
    qkv = np.split(x, 3, -1)
    # Split into attention heads.
    qkv_heads = list(map(lambda x: np.split(x, n_head, -1), qkv))
    # Compute attention for each head.
    # TODO: Parallelize.
    attn_outputs = [causal_self_attention(q, k, v) for q, k, v in zip(*qkv_heads)]
    # Concat outputs.
    x = np.hstack(attn_outputs)
    # Final projection.
    return linear(x, **c_proj)







def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text

if __name__ == "__main__":
    # fire.Fire(main)
    print(main("Alan Turing theorized that computers would one day become", 8))