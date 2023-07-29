# TODO: dropout, nucleus sampling, parallelise attention heads

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
from functools import partial
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


# We need to tell JIT not to worry about n_head since this won't actually change.
@partial(jit, static_argnames=["n_head"])
def gpt(inputs, kv_cache, wte, wpe, blocks, ln_f, n_head):
    # Input slicing based on the length of the KV-cache.
    kvc_len = kv_cache[0][0].shape[0]
    pos = [kvc_len, len(inputs)]
    inputs = inputs[kvc_len:]
    # Index into token embeddings and add positional encoding.
    x = wte[inputs, :] + wpe[pos[0]:pos[1], :]
    # Forward pass through blocks.
    for i in range(len(blocks)):
        x, kv = decoder_block(x, kv_cache[i], **blocks[i], n_head=n_head)
        kv_cache[i] = kv
    # Get logits.
    x = layernorm(x, **ln_f) # (GPT2-style)
    return x @ wte.T, kv_cache

def generate(input, params, hparams, n_output):
    n_embd = hparams["n_embd"]; n_head = hparams["n_head"]; n_layer = hparams["n_layer"]
    # Initialise KV-cache.
    kv_cache = [(jnp.zeros((0, n_embd)), jnp.zeros((0, n_embd)))]*n_layer
    for _ in tqdm(range(n_output), "Generating:"):
        # Forward pass.
        logits, kv_cache = gpt(input, kv_cache, **params, n_head=n_head)
        # Greedy sampling.
        out = jnp.argmax(logits[-1])
        input.append(int(out))
    return input[-n_output:]

def gelu(x): 
    return 0.5*x*(1+jnp.tanh(jnp.sqrt(2/jnp.pi)*(x + 0.044715 * x**3)))

def softmax(x, temp=1.):
    x = jnp.exp(x-jnp.max(x, -1, keepdims=True) / temp)
    return x / jnp.sum(x, -1, keepdims=True)

def layernorm(x, g, b, eps=1e-6):
    mean = jnp.mean(x, -1, keepdims=True)
    std = jnp.std(x, -1, keepdims=True)
    return g * (x-mean)/(std+eps) + b

def linear(x, w, b): 
    return x @ w + b

def decoder_block(x, kv, attn, ln_1, mlp, ln_2, n_head):
    # pre-layernorm -> multi-head causal self-attention -> residual connection
    o, kv = mha(layernorm(x, **ln_1), kv, **attn, n_head=n_head); x += o
    # pre-layernorm -> position-wise MLP -> residual connection
    x += ffnn(layernorm(x, **ln_2), **mlp)
    return x, kv

def ffnn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)
    
def causal_self_attention(q, k, v):
    axs = list(range(k.ndim))
    axs[-1], axs[-2] = axs[-2], axs[-1]
    scores = q @ k.transpose(axs) / jnp.sqrt(q.shape[-1])
    # Masking for causal attention.
    # Note that we don't mask if we are using the KV-cache.
    mask = jnp.ones(scores.shape)
    mask = jnp.tril(mask, k=0) if scores.shape[0] > 1 else mask
    return softmax(jnp.where(mask == 1, scores, -1e9)) @ v

def mha(x, past_kv, c_attn, c_proj, n_head):
    # QKV projection.
    x = linear(x, **c_attn)
    # Split into QKV.
    qkv = jnp.split(x, 3, -1)
    # Retrieve KV-cache.
    # TODO: Avoid concatenation by using a single KV-cache initialised to a context length.
    new_k = jnp.concatenate([past_kv[0], qkv[1]], axis=0)
    new_v = jnp.concatenate([past_kv[1], qkv[2]], axis=0)
    # Split into attention heads.
    qkv_heads = list(map(lambda x: jnp.array(jnp.split(x, n_head, -1)), [qkv[0], new_k, new_v]))
    # Compute attention for each head.
    # TODO: Can MHA be done in one go?
    attn_outputs = vmap(causal_self_attention)(*qkv_heads)
    # Concat outputs.
    x = jnp.hstack(attn_outputs)
    # Final projection.
    return linear(x, **c_proj), (new_k, new_v)


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams, n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text

if __name__ == "__main__":
    print(main("Alan Turing theorized that computers would one day become", 8))