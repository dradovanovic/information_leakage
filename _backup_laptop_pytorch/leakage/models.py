"""Model builders (PyTorch). Every model returns (real_logits, aux_logits), no softmax.

Every model has the attributes `real_logits` and `aux_logit` (the two heads), so training and
evaluation code does not depend on the architecture.

Initialisation follows the Keras defaults of the earlier TensorFlow code (Glorot-uniform
weights, zero biases, N(0, 0.05) positional embeddings), so the numbers stay comparable.
Inputs are (batch, days, slots) weeks.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


def keras_init_(module):
    """Glorot-uniform weights and zero biases for every Linear / Conv2d (Keras defaults)."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


class MLP(nn.Module):
    """Plain MLP as in Cloud et al. (2025), MNIST: (in, 256, 256, n_classes + m)."""

    def __init__(self, input_shape, n_classes, aux_logits=3, hidden=(256, 256), dropout=0.0):
        super().__init__()
        d = math.prod(input_shape)
        layers = [nn.Flatten()]
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.body = nn.Sequential(*layers)
        self.real_logits = nn.Linear(d, n_classes)
        self.aux_logit = nn.Linear(d, aux_logits)
        keras_init_(self)

    def forward(self, x):
        z = self.body(x)
        return self.real_logits(z), self.aux_logit(z)


class WangCNN(nn.Module):
    """Wang et al. (2019) 2D CNN on a (days, slots) week, same layers as Models/CNNs.py."""

    def __init__(self, input_shape, n_classes, aux_logits=3, dropout=0.5):
        super().__init__()
        days, slots = input_shape[0], input_shape[1]
        self.body = nn.Sequential(
            nn.Conv2d(1, 8, (2, 3)), nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)), nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.body(torch.zeros(1, 1, days, slots)).shape[1]
        self.d1 = nn.Linear(flat, 32)
        self.real_logits = nn.Linear(32, n_classes)
        self.aux_logit = nn.Linear(32, aux_logits)
        keras_init_(self)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, days, slots)
        z = F.relu(self.d1(self.body(x)))
        return self.real_logits(z), self.aux_logit(z)


class KerasStyleMHA(nn.Module):
    """Multi-head self-attention with Keras semantics: every head has its own `key_dim`
    (independent of d_model), output projected back to d_model."""

    def __init__(self, d_model, num_heads, key_dim, dropout=0.0):
        super().__init__()
        self.h, self.dk, self.dropout = num_heads, key_dim, dropout
        self.q = nn.Linear(d_model, num_heads * key_dim)
        self.k = nn.Linear(d_model, num_heads * key_dim)
        self.v = nn.Linear(d_model, num_heads * key_dim)
        self.o = nn.Linear(num_heads * key_dim, d_model)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = (p(x).view(b, t, self.h, self.dk).transpose(1, 2) for p in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.o(a.transpose(1, 2).reshape(b, t, self.h * self.dk))


class EncoderBlock(nn.Module):
    """Pre-LN transformer block, as TransformerKeras.transformer_encoder in Models/Transformer.py."""

    def __init__(self, d_model, head_size, num_heads, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.mha = KerasStyleMHA(d_model, num_heads, head_size, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.mha(self.ln1(x)))
        y = self.drop(F.gelu(self.ff1(self.ln2(x))))
        return x + self.drop(self.ff2(y))


class Transformer2DPE(nn.Module):
    """Port of TransformerKeras.build_model_with_aux (Models/Transformer.py): one token per
    (day, slot) cell, learned day + slot embeddings, 4 pre-LN blocks, mean pooling, MLP head."""

    def __init__(self, input_shape, n_classes, aux_logits=3, dropout=0.1, d_model=128, head_size=128,
                 num_heads=4, ff_dim=128, num_blocks=4, mlp_units=(128,), mlp_dropout=0.3):
        super().__init__()
        days, slots = input_shape[0], input_shape[1]
        self.token = nn.Linear(1, d_model, bias=False)
        self.day_emb = nn.Parameter(torch.empty(days, d_model))
        self.slot_emb = nn.Parameter(torch.empty(slots, d_model))
        self.blocks = nn.ModuleList(EncoderBlock(d_model, head_size, num_heads, ff_dim, dropout)
                                    for _ in range(num_blocks))
        self.head_ln = nn.LayerNorm(d_model, eps=1e-6)
        mlp, d = [], d_model
        for u in mlp_units:
            mlp += [nn.Linear(d, u), nn.GELU(), nn.Dropout(mlp_dropout)]
            d = u
        self.mlp = nn.Sequential(*mlp)
        self.d1 = nn.Linear(d, mlp_units[-1])
        self.real_logits = nn.Linear(mlp_units[-1], n_classes)
        self.aux_logit = nn.Linear(mlp_units[-1], aux_logits)
        keras_init_(self)
        nn.init.normal_(self.day_emb, std=0.05)
        nn.init.normal_(self.slot_emb, std=0.05)

    def forward(self, x):
        b, days, slots = x.shape[:3]
        z = self.token(x.reshape(b, days, slots, 1))
        z = z + self.day_emb[None, :, None, :] + self.slot_emb[None, None, :, :]
        z = z.reshape(b, days * slots, -1)
        for blk in self.blocks:
            z = blk(z)
        z = self.head_ln(z).mean(dim=1)
        rep = F.gelu(self.d1(self.mlp(z)))
        return self.real_logits(rep), self.aux_logit(rep)


BUILDERS = {"mlp": MLP, "wang": WangCNN, "transformer": Transformer2DPE}


def make_builder(arch, input_shape, n_classes, aux_logits, device="cpu", **kwargs):
    """Return a function `build(name=None)` that creates a fresh, randomly initialised model."""
    if arch not in BUILDERS:
        raise ValueError(f"Unknown architecture '{arch}', choose from {list(BUILDERS)}")
    cls = BUILDERS[arch]

    def _build(name=None):  # name kept for API compatibility with the Keras version
        return cls(tuple(input_shape), n_classes, aux_logits=aux_logits, **kwargs).to(device)

    return _build


def clone(model):
    """Independent copy with identical weights (shared initialisation)."""
    return copy.deepcopy(model)
