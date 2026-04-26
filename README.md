# microgpt — Enhanced with Modern LLM Algorithms

This repository contains:

| File | Description |
|------|-------------|
| `microgpt_original.py` | Karpathy's original microgpt, committed without modification |
| `microgpt_enhanced.py` | Extended version implementing four modern LLM algorithms |

The original microgpt is a beautiful 200-line, dependency-free Python file that trains and runs a GPT. It contains the complete algorithmic core: dataset, tokenizer, autograd, transformer architecture, Adam optimizer, training loop, and inference.

---

## Implemented Algorithms

### 1. Gaussian Error Linear Units (GELUs)

**Paper:** Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)

**Underlying Idea:**

The original microgpt uses ReLU as the non-linearity in its MLP block: `ReLU(x) = max(0, x)`. ReLU is a hard gate — it completely zeros out any negative input, which can cause "dead neurons" where gradients stop flowing.

GELU takes a probabilistic approach. It asks: *given that this input came from a Gaussian distribution, how likely is it to be positive?* Formally:

```
GELU(x) = x · Φ(x)
```

where `Φ(x)` is the cumulative distribution function (CDF) of the standard normal distribution. In practice a fast `tanh`-based approximation is used:

```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

The key difference from ReLU is that GELU is a *smooth* gate. Small negative values are not hard-zeroed — they are attenuated in proportion to how unlikely they are under the Gaussian prior. This preserves gradient signal through negative activations, improves training stability, and empirically outperforms ReLU on most NLP benchmarks. GELU is the default activation in GPT-2, GPT-3, and BERT.

**Where in the code:** `gelu()` function; used inside each expert's MLP in the MoE block.

```python
def gelu(x):
    inner = _SQRT_2_OVER_PI * (x + _GELU_COEFF * x ** 3)
    return x * (Value(0.5) + Value(0.5) * inner.tanh())
```

---

### 2. LoRA — Low-Rank Adaptation

**Paper:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

**Underlying Idea:**

When fine-tuning a large pre-trained model, updating all parameters is expensive. The key insight in LoRA is that the *change* in weights during fine-tuning has low intrinsic rank — the meaningful updates lie in a much smaller subspace than the full weight matrix.

LoRA exploits this by constraining each weight update `ΔW` to a low-rank decomposition:

```
W_effective = W_base + ΔW = W_base + B · A
```

where:
- `W_base` is the frozen (or slowly-updated) base matrix of shape `(d_out × d_in)`
- `A` is a small matrix of shape `(r × d_in)` — projects down to rank `r`
- `B` is a small matrix of shape `(d_out × r)` — projects back up

`B` is initialised to zero so `ΔW = 0` at the start, meaning the model begins identical to the base. Only `A` and `B` are updated. The number of trainable parameters drops from `d²` to `2·d·r`, which for large models (d = 4096, r = 8) is a 256× reduction.

In this implementation, LoRA adapters are applied to the **Q** and **V** attention projection matrices (as recommended in the paper), with `r = 2`.

**Where in the code:** `lora_linear()` function; `attn_wq_lora_A/B` and `attn_wv_lora_A/B` parameters.

```python
def lora_linear(x, w_base, lora_A, lora_B):
    base_out = linear(x, w_base)       # W_base @ x
    a_out    = linear(x, lora_A)       # A @ x  (rank r)
    delta    = linear(a_out, lora_B)   # B @ A @ x
    return [b + d for b, d in zip(base_out, delta)]
```

---

### 3. RoPE — Rotary Position Embeddings

**Paper:** Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

**Underlying Idea:**

Transformers have no inherent sense of order — they treat all tokens as a set. The original microgpt handles this by adding a learned position embedding vector to each token embedding. RoPE takes a fundamentally different approach.

Instead of adding position information to the *input*, RoPE encodes position by *rotating* the query and key vectors inside the attention computation. Each pair of dimensions `(2i, 2i+1)` is rotated by an angle that is a function of the token's absolute position and the dimension index:

```
θᵢ = position / 10000^(2i / d)

[x₂ᵢ' ]   = [ cos θᵢ  -sin θᵢ ] · [x₂ᵢ  ]
[x₂ᵢ₊₁']   [ sin θᵢ   cos θᵢ ]   [x₂ᵢ₊₁]
```

The beautiful property of rotations is that the **dot product** `q · k` becomes a function only of the *relative* distance between positions, not their absolute values:

```
RoPE(q, pos_q) · RoPE(k, pos_k) depends only on (pos_q - pos_k)
```

This gives the model natural relative-position reasoning and better extrapolation to sequence lengths not seen during training. No extra parameters are needed. RoPE is used in LLaMA, Mistral, Falcon, and most modern open-source LLMs. Here it completely replaces the `wpe` (learned position embedding table).

**Where in the code:** `rope_rotate()` function; applied to each head's Q and K slices before the attention dot product.

```python
def rope_rotate(x, pos_id):
    for i in range(len(x) // 2):
        theta = pos_id / (10000 ** (2 * i / len(x)))
        x0, x1 = x[2*i], x[2*i+1]
        out[2*i]   = x0 * cos(theta) - x1 * sin(theta)
        out[2*i+1] = x0 * sin(theta) + x1 * cos(theta)
```

---

### 4. Mixture of Experts (MoE)

**Reference:** Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017.  
HuggingFace MoE blog: [https://huggingface.co/blog/moe](https://huggingface.co/blog/moe)

**Underlying Idea:**

In the standard transformer, every token passes through the same MLP block. MoE replaces this single MLP with `E` independent "expert" MLPs and a small **router** network.

For each token, the router learns to assign it to the most relevant subset of experts. In top-k routing (k=2 here):

1. The router projects the token's hidden state to `E` logits and applies softmax to get routing probabilities.
2. The top-k experts (by probability) are selected.
3. Only those k experts process the token; their outputs are summed, weighted by the router probabilities.

```
output = Σ_{e ∈ top-k} router_prob[e] · Expert_e(x)
```

The critical insight: total model parameters scale as `O(E)`, but compute per token scales as `O(k)`. If `k << E`, you get a much larger model (more knowledge, more capacity) at roughly the same computational cost as a dense model with k experts. Mixture of Experts is used in GPT-4, Mixtral 8×7B, and many other frontier models.

In this implementation: `n_experts=4`, `n_experts_used=2` (top-2 routing), with each expert being a GELU MLP identical in shape to the original microgpt MLP.

**Where in the code:** `moe_mlp()` function; `expert{e}_fc1/fc2` and `moe_router` parameters.

```python
def moe_mlp(x, li):
    router_probs = softmax(linear(x, router_weights))
    top_k = argsort(router_probs)[:n_experts_used]
    out = sum(router_probs[e] * expert_e(x) for e in top_k)
    return out
```

---

## How to Run

```bash
# Original microgpt (no modifications)
python microgpt_original.py

# Enhanced version with GELUs, LoRA, RoPE, MoE
python microgpt_enhanced.py
```

No dependencies required — pure Python standard library only.

---

## References

| Algorithm | Paper | Link |
|-----------|-------|-------|
| GELUs | Hendrycks & Gimpel, 2016 | https://arxiv.org/abs/1606.08415 |
| LoRA | Hu et al., 2021 | https://arxiv.org/abs/2106.09685 |
| RoPE | Su et al., 2021 | https://arxiv.org/abs/2104.09864 |
| MoE | Shazeer et al., 2017 / HuggingFace | https://huggingface.co/blog/moe |
| microgpt | Karpathy, 2026 | https://karpathy.github.io/2026/02/12/microgpt/ |

---

## AI Tools Used

- **Claude (Anthropic)** — algorithm implementation, documentation
- Development assisted with **Claude Code** as recommended by the assignment
