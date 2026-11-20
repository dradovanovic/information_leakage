"""Teacher training and aux-only distillation on noise (PyTorch)."""
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def to_tensor(a, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


# --------------------------------------------------------------------------------------
# Noise inputs for distillation
# --------------------------------------------------------------------------------------
NOISE_TYPES = ("uniform", "gaussian", "uniform_centered", "gaussian_matched")


def make_noise(kind, n, shape, rng, x_ref=None, k_std=3.0):
    """Return n noise samples (numpy) of the given per-sample shape.

    uniform          : U[-1, 1] (Cloud et al. MNIST setup)
    gaussian         : N(0, 1)  (natural choice when inputs are standardised)
    uniform_centered : U[mu - k*sigma, mu + k*sigma] clipped to the data range; global
                       statistics of x_ref
    gaussian_matched : N(mu_j, sigma_j) per input feature of x_ref. Uses more knowledge about
                       the training data than the others; treat as an ablation.
    """
    size = (n, *shape)
    if kind == "uniform":
        return rng.uniform(-1.0, 1.0, size=size).astype("float32")
    if kind == "gaussian":
        return rng.standard_normal(size=size).astype("float32")
    if x_ref is None:
        raise ValueError(f"noise '{kind}' needs reference data")
    if kind == "uniform_centered":
        mu, sd = float(x_ref.mean()), float(x_ref.std())
        low, high = max(mu - k_std * sd, float(x_ref.min())), min(mu + k_std * sd, float(x_ref.max()))
        return rng.uniform(low, high, size=size).astype("float32")
    if kind == "gaussian_matched":
        mu, sd = x_ref.mean(axis=0), x_ref.std(axis=0) + 1e-6
        return (mu + sd * rng.standard_normal(size=size)).astype("float32")
    raise ValueError(f"unknown noise type '{kind}', choose from {NOISE_TYPES}")


def _batches(n, batch_size, device, shuffle=True):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


# --------------------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------------------
def train_on_real_logits(model, x, y, epochs, batch_size, lr=1e-3, weight_decay=0.0,
                         x_val=None, y_val=None, patience=None, class_weight=None):
    """Cross-entropy on the real logits only; the aux head gets no direct gradient.

    x, y (and x_val, y_val) are tensors on the model's device. Early stopping with restore of
    the best weights is used only when validation data and `patience` are given.
    Returns the number of epochs actually trained.
    """
    clip = weight_decay > 0
    opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if clip
           else torch.optim.Adam(model.parameters(), lr=lr))
    w = None
    if class_weight is not None:
        w = torch.as_tensor(class_weight, dtype=torch.float32, device=x.device)
    best, best_state, bad, trained = float("inf"), None, 0, 0
    for _ in range(epochs):
        model.train()
        for b in _batches(len(x), batch_size, x.device):
            real, _aux = model(x[b])
            loss = F.cross_entropy(real, y[b], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        trained += 1
        if x_val is not None and patience:
            model.eval()
            with torch.no_grad():
                tot = sum(F.cross_entropy(model(x_val[b])[0], y_val[b], weight=w, reduction="sum").item()
                          for b in _batches(len(x_val), 4096, x.device, shuffle=False))
            val = tot / len(x_val)
            if val < best - 1e-6:
                best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return trained


def balanced_class_weight(y, n_classes):
    counts = np.bincount(np.asarray(y).astype(int), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return (len(y) / (n_classes * counts)).tolist()


# --------------------------------------------------------------------------------------
# Student: KL on aux logits only, noise inputs only
# --------------------------------------------------------------------------------------
def distill_aux(student, teacher, noise, epochs, batch_size, lr=1e-3, use_all_logits=False,
                temperature=1.0):
    """Minimise KL(softmax(t_aux/T) || softmax(s_aux/T)) on noise (a tensor on the device).

    Both models are in eval mode, so dropout is off during distillation.
    Returns the list of mean losses per epoch.
    """
    teacher.eval()
    student.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    with torch.no_grad():  # teacher targets are fixed, compute once
        parts = []
        for b in _batches(len(noise), 4096, noise.device, shuffle=False):
            t_real, t_aux = teacher(noise[b])
            parts.append(torch.cat([t_real, t_aux], dim=-1) if use_all_logits else t_aux)
        target = F.softmax(torch.cat(parts) / temperature, dim=-1)
    losses = []
    for _ in range(epochs):
        run = []
        for b in _batches(len(noise), batch_size, noise.device):
            s_real, s_aux = student(noise[b])
            out = torch.cat([s_real, s_aux], dim=-1) if use_all_logits else s_aux
            loss = F.kl_div(F.log_softmax(out / temperature, dim=-1), target[b], reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run.append(loss.item())
        losses.append(float(np.mean(run)))
    return losses


@torch.no_grad()
def predict_probs(model, x, batch_size=4096):
    model.eval()
    out = [F.softmax(model(x[b])[0], dim=-1) for b in _batches(len(x), batch_size, x.device, shuffle=False)]
    return torch.cat(out).cpu().numpy()
