"""
microgpt_enhanced.py

The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
Based on Karpathy's microgpt (https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

This enhanced version adds four modern LLM techniques:
  1. GELU activation (replaces ReLU in the MLP block)
  2. LoRA (Low-Rank Adaptation) applied to attention weight matrices
  3. RoPE (Rotary Position Embeddings) replacing learned position embeddings
  4. Mixture of Experts (MoE) replacing the standard MLP block

@karpathy (original), extended for MASC 515 Assignment 3
"""

import os
import math
import random
random.seed(42)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Autograd (unchanged from original microgpt)
# ---------------------------------------------------------------------------
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t**2,))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
n_layer    = 1      # number of transformer layers
n_embd     = 16     # embedding dimension
block_size = 16     # maximum context length
n_head     = 4      # number of attention heads
head_dim   = n_embd // n_head

# LoRA rank (small = fewer extra params; r=2 is tiny but illustrative)
lora_rank  = 2

# MoE settings
n_experts      = 4   # total number of expert MLPs
n_experts_used = 2   # top-k experts activated per token

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
zeros  = lambda nout, nin:           [[Value(0.0)                   for _ in range(nin)] for _ in range(nout)]

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
state_dict = {
    'wte':     matrix(vocab_size, n_embd),   # token embeddings  (no wpe — using RoPE instead)
    'lm_head': matrix(vocab_size, n_embd),
}

for i in range(n_layer):
    # Attention weight matrices (full-rank base)
    for name in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo']:
        state_dict[f'layer{i}.{name}'] = matrix(n_embd, n_embd)

    # LoRA adapter pairs for Q and V projections:
    #   W_effective = W_base + B @ A,  where A: (r x d_in), B: (d_out x r)
    for name in ['attn_wq', 'attn_wv']:
        state_dict[f'layer{i}.{name}_lora_A'] = matrix(lora_rank, n_embd, std=0.02)
        state_dict[f'layer{i}.{name}_lora_B'] = zeros(n_embd, lora_rank)   # init B=0 so delta=0 at start

    # MoE: each expert is an independent two-layer MLP (same shape as original)
    for e in range(n_experts):
        state_dict[f'layer{i}.expert{e}_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.expert{e}_fc2'] = matrix(n_embd, 4 * n_embd)

    # MoE router: projects x to n_experts logits
    state_dict[f'layer{i}.moe_router'] = matrix(n_experts, n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def linear(x, w):
    """Matrix-vector multiply: returns w @ x."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# ---------------------------------------------------------------------------
# 1. GELU Activation
#    Paper: Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)"
#           https://arxiv.org/abs/1606.08415
#
#    Idea: Instead of the hard threshold of ReLU (0 if x<0, x if x>0),
#    GELU weights each input by the probability that a standard Gaussian
#    random variable is less than that input:
#        GELU(x) = x * Phi(x)
#    where Phi is the Gaussian CDF.  A fast tanh approximation is used:
#        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#    This smooth gating means small negative values are not completely
#    zeroed out, which improves gradient flow and empirically yields better
#    performance than ReLU on most language-model benchmarks.
# ---------------------------------------------------------------------------
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_COEFF     = 0.044715

def gelu(x):
    """GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))"""
    inner = _SQRT_2_OVER_PI * (x + _GELU_COEFF * x ** 3)
    return x * (Value(0.5) + Value(0.5) * inner.tanh())

# ---------------------------------------------------------------------------
# 2. LoRA — Low-Rank Adaptation
#    Paper: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
#           https://arxiv.org/abs/2106.09685
#
#    Idea: For a pre-trained weight matrix W (d_out × d_in), instead of
#    updating all d_out*d_in parameters during fine-tuning, constrain the
#    update to a low-rank decomposition:
#        W_effective = W_base + B @ A
#    where A is (r × d_in) and B is (d_out × r), with r << d_in.
#    B is initialised to zero so the model starts identical to the base.
#    During training only A and B are updated (though here we train
#    everything together for simplicity).  The number of trainable params
#    drops from d^2 to 2*d*r.  Applied here to the Q and V attention
#    projections, following the recommendation in the paper.
# ---------------------------------------------------------------------------
def lora_linear(x, w_base, lora_A, lora_B):
    """
    Compute (W_base + B @ A) @ x using separate base and LoRA matrices.
    lora_A: (r x d_in), lora_B: (d_out x r)
    """
    base_out = linear(x, w_base)                     # d_out-length list
    a_out    = linear(x, lora_A)                     # r-length list  (A @ x)
    delta    = linear(a_out, lora_B)                 # d_out-length list (B @ A @ x)
    return [b + d for b, d in zip(base_out, delta)]

# ---------------------------------------------------------------------------
# 3. RoPE — Rotary Position Embeddings
#    Paper: Su et al., "RoFormer: Enhanced Transformer with Rotary
#           Position Embedding"  https://arxiv.org/abs/2104.09864
#
#    Idea: Encode position information by *rotating* query and key vectors
#    rather than adding a learned position vector to the input.  Each pair
#    of dimensions (2i, 2i+1) of the vector is rotated by angle pos * θ_i
#    where θ_i = 1 / 10000^(2i/d).  The crucial property is that the dot
#    product q·k then depends only on the *relative* distance (pos_q - pos_k),
#    not on absolute positions, giving the model better length generalisation.
#    No extra parameters are needed; the rotation is a pure function of
#    position and dimension index.
# ---------------------------------------------------------------------------
def rope_rotate(x, pos_id):
    """
    Apply RoPE to a vector x at sequence position pos_id.
    Works on pairs of dimensions: (x[2i], x[2i+1]) -> rotated pair.
    """
    d = len(x)
    out = list(x)  # copy
    for i in range(d // 2):
        theta = pos_id / (10000 ** (2 * i / d))
        cos_t = Value(math.cos(theta))
        sin_t = Value(math.sin(theta))
        x0, x1 = x[2 * i], x[2 * i + 1]
        out[2 * i]     = x0 * cos_t - x1 * sin_t
        out[2 * i + 1] = x0 * sin_t + x1 * cos_t
    return out

# ---------------------------------------------------------------------------
# 4. Mixture of Experts (MoE)
#    Reference: Shazeer et al. (2017) "Outrageously Large Neural Networks"
#               https://arxiv.org/abs/1701.06538
#               HuggingFace MoE blog: https://huggingface.co/blog/moe
#
#    Idea: Replace the single MLP block with E independent "expert" MLPs
#    and a learned router network.  For each token the router computes a
#    probability distribution over experts and selects the top-k (here k=2).
#    Only those k experts run; their outputs are summed weighted by the
#    router probabilities.  Because most experts are idle for any given token,
#    total parameters scale with E while compute stays proportional to k.
#    This allows massive model capacity with sub-linear compute cost.
#    The router is a simple linear projection followed by softmax.
# ---------------------------------------------------------------------------
def moe_mlp(x, li):
    """
    Mixture-of-Experts MLP block.
    1. Router scores every expert.
    2. Top-k experts are selected (hard, non-differentiable gate for simplicity).
    3. Selected experts process x; outputs are weighted-summed by router probs.
    """
    # Router: compute logits over all experts
    router_logits = linear(x, state_dict[f'layer{li}.moe_router'])   # list of n_experts Values
    router_probs  = softmax(router_logits)                             # normalised probabilities

    # Top-k selection (pick indices of largest router probs)
    scored = sorted(range(n_experts), key=lambda e: router_probs[e].data, reverse=True)
    top_k  = scored[:n_experts_used]

    # Weighted sum of expert outputs
    out = [Value(0.0)] * n_embd
    for e in top_k:
        # Expert MLP: fc1 -> GELU -> fc2
        h = linear(x, state_dict[f'layer{li}.expert{e}_fc1'])
        h = [gelu(xi) for xi in h]
        h = linear(h, state_dict[f'layer{li}.expert{e}_fc2'])
        # Weight by router probability for this expert
        w = router_probs[e]
        out = [o + w * hi for o, hi in zip(out, h)]

    return out

# ---------------------------------------------------------------------------
# Model forward pass
# ---------------------------------------------------------------------------
def gpt(token_id, pos_id, keys, values):
    # Token embedding only (RoPE handles position — no wpe lookup)
    x = list(state_dict['wte'][token_id])   # copy so we don't mutate the table
    x = rmsnorm(x)

    for li in range(n_layer):
        # ---- 1) Multi-head Attention with RoPE + LoRA ----
        x_residual = x
        x = rmsnorm(x)

        # LoRA-augmented Q and V; standard K and O
        q = lora_linear(x, state_dict[f'layer{li}.attn_wq'],
                         state_dict[f'layer{li}.attn_wq_lora_A'],
                         state_dict[f'layer{li}.attn_wq_lora_B'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = lora_linear(x, state_dict[f'layer{li}.attn_wv'],
                         state_dict[f'layer{li}.attn_wv_lora_A'],
                         state_dict[f'layer{li}.attn_wv_lora_B'])

        # Apply RoPE to each head slice of q and k
        q_rope, k_rope = [], []
        for h in range(n_head):
            hs = h * head_dim
            q_rope.extend(rope_rotate(q[hs:hs+head_dim], pos_id))
            k_rope.extend(rope_rotate(k[hs:hs+head_dim], pos_id))

        keys[li].append(k_rope)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q_rope[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # ---- 2) MoE MLP block (with GELU inside each expert) ----
        x_residual = x
        x = rmsnorm(x)
        x = moe_mlp(x, li)
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# ---------------------------------------------------------------------------
# Optimizer (Adam, unchanged)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
num_steps = 1000
for step in range(num_steps):
    doc    = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n      = min(block_size, len(tokens) - 1)

    keys_cache   = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits  = gpt(token_id, pos_id, keys_cache, values_cache)
        probs   = softmax(logits)
        loss_t  = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat    = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat    = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data  -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad   = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys_cache   = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    token_id = BOS
    sample   = []
    for pos_id in range(block_size):
        logits   = gpt(token_id, pos_id, keys_cache, values_cache)
        probs    = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
