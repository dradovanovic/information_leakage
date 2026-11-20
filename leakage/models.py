"""PyTorch model builders. Every model returns (real_logits, aux_logits), no softmax.

All architectures expose the heads as `model.real_logits` and `model.aux_logit`, so training
and evaluation do not depend on the architecture. Linear and conv layers use Glorot-uniform
weights and zero biases (the Keras default), which keeps runs comparable with the earlier
TensorFlow results. Pass init="torch" to keep PyTorch's default initialisation instead.
"""
import torch
from torch import nn


def _glorot(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MLP(nn.Module):
    """Plain MLP as in Cloud et al. (2025), MNIST: (in, 256, 256, n_classes + m)."""

    def __init__(self, input_shape, n_classes, aux_logits=3, hidden=(256, 256), dropout=0.0):
        super().__init__()
        d = 1
        for s in input_shape:
            d *= s
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.body = nn.Sequential(nn.Flatten(), *layers)
        self.real_logits = nn.Linear(d, n_classes)
        self.aux_logit = nn.Linear(d, aux_logits)

    def forward(self, x):
        z = self.body(x)
        return self.real_logits(z), self.aux_logit(z)


class WangCNN(nn.Module):
    """Wang et al. (2019) 2D CNN on a (days, slots) week."""

    def __init__(self, input_shape, n_classes, aux_logits=3, dropout=0.5):
        super().__init__()
        days, slots = input_shape[0], input_shape[1]
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (2, 3)), nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout), nn.Flatten())
        flat = 16 * ((days - 3) // 2) * ((slots - 4) // 2)
        self.d1 = nn.Sequential(nn.Linear(flat, 32), nn.ReLU())
        self.real_logits = nn.Linear(32, n_classes)
        self.aux_logit = nn.Linear(32, aux_logits)

    def forward(self, x):
        z = self.d1(self.features(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])))
        return self.real_logits(z), self.aux_logit(z)


class WeekTransformer(nn.Module):
    """Transformer over the (days x slots) grid with learned day and slot embeddings."""

    def __init__(self, input_shape, n_classes, aux_logits=3, dropout=0.1, d_model=128, heads=4,
                 ff_dim=128, blocks=4, mlp_dropout=0.3):
        super().__init__()
        days, slots = input_shape[0], input_shape[1]
        self.proj = nn.Linear(1, d_model, bias=False)
        self.day_emb = nn.Parameter(torch.randn(days, 1, d_model) * 0.02)
        self.slot_emb = nn.Parameter(torch.randn(1, slots, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(mlp_dropout),
                                  nn.Linear(128, 128), nn.GELU())
        self.real_logits = nn.Linear(128, n_classes)
        self.aux_logit = nn.Linear(128, aux_logits)

    def forward(self, x):
        n, d, s = x.shape
        t = self.proj(x.unsqueeze(-1)) + self.day_emb + self.slot_emb
        z = self.norm(self.encoder(t.reshape(n, d * s, -1))).mean(dim=1)
        z = self.head(z)
        return self.real_logits(z), self.aux_logit(z)


class GRUNet(nn.Module):
    """GRU over the week as one sequence of days*slots steps with one feature per step."""

    def __init__(self, input_shape, n_classes, aux_logits=3, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.d1 = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU())
        self.real_logits = nn.Linear(32, n_classes)
        self.aux_logit = nn.Linear(32, aux_logits)

    def forward(self, x):
        _, h = self.gru(x.reshape(x.shape[0], -1, 1))
        z = self.d1(h[-1])
        return self.real_logits(z), self.aux_logit(z)


ARCHS = {"mlp": MLP, "wang": WangCNN, "transformer": WeekTransformer, "gru": GRUNet}


def make_builder(arch, input_shape, n_classes, aux_logits, device="cpu", init="glorot", **kwargs):
    """Return a zero-argument function that builds a fresh (randomly initialised) model."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture '{arch}', choose from {list(ARCHS)}")

    def _build():
        m = ARCHS[arch](tuple(input_shape), n_classes, aux_logits=aux_logits, **kwargs)
        if init == "glorot":
            _glorot(m)
        return m.to(device)

    return _build


def clone(builder, model):
    """Fresh model with the same weights (the shared initialisation)."""
    m = builder()
    m.load_state_dict(model.state_dict())
    return m


@torch.no_grad()
def weight_distance(model, ref):
    """||w_model - w_ref|| / ||w_ref|| over all parameters (relative L2 distance from the init)."""
    ref_params = dict(ref.named_parameters())
    num = sum(((p - ref_params[n]) ** 2).sum() for n, p in model.named_parameters())
    den = sum((p ** 2).sum() for p in ref_params.values())
    return float(torch.sqrt(num / den))
