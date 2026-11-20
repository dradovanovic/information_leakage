"""Teacher training and aux-only distillation on noise (PyTorch).

Optimiser settings follow the earlier Keras code: Adam(lr, eps=1e-7); with weight decay AdamW and
per-tensor gradient clipping at norm 1.0 (Keras `clipnorm`). Class weights scale each sample's
loss and the batch loss is divided by the batch size (Keras `class_weight` semantics).
"""
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------------------
def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _device(model):
    return next(model.parameters()).device


def _batches(n, batch_size, shuffle):
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


# --------------------------------------------------------------------------------------
# Noise inputs for distillation
# --------------------------------------------------------------------------------------
NOISE_TYPES = ("uniform", "gaussian", "uniform_centered", "gaussian_matched")


def make_noise(kind, n, shape, rng, x_ref=None, k_std=3.0):
    """Return n noise samples of the given per-sample shape (numpy, float32).

    uniform          : U[-1, 1] (Cloud et al. MNIST setup, previous test_mnist.py)
    gaussian         : N(0, 1)  (natural choice when inputs are standardised)
    uniform_centered : U[mu - k*sigma, mu + k*sigma] clipped to the data range (previous
                       run_subliminal_learning.py); global statistics of x_ref
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


# --------------------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _mean_ce(model, x, y, batch_size=1024):
    model.eval()
    dev, tot = _device(model), 0.0
    for idx in _batches(len(x), batch_size, shuffle=False):
        real, _ = model(x[idx].to(dev))
        tot += F.cross_entropy(real, y[idx].to(dev), reduction="sum").item()
    return tot / len(x)


def train_on_real_logits(model, x, y, epochs, batch_size, lr=1e-3, weight_decay=0.0,
                         x_val=None, y_val=None, patience=None, class_weight=None, verbose=0):
    """Cross-entropy on `real_logits` only; the aux head is not updated.

    Early stopping on validation loss (unweighted) only when validation data and `patience`
    are given; the best weights are restored. Returns the number of epochs actually trained.
    """
    dev = _device(model)
    params = [p for n, p in model.named_parameters() if not n.startswith("aux_logit.")]
    if weight_decay > 0:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, eps=1e-7)
    else:
        opt = torch.optim.Adam(params, lr=lr, eps=1e-7)
    x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.long)
    cw = None
    if class_weight is not None:
        cw = torch.tensor([class_weight[i] for i in range(len(class_weight))], dtype=torch.float32, device=dev)
    use_val = x_val is not None and patience
    if use_val:
        x_val, y_val = torch.as_tensor(x_val), torch.as_tensor(y_val, dtype=torch.long)
    best, best_state, wait, trained = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        for idx in _batches(len(x), batch_size, shuffle=True):
            xb, yb = x[idx].to(dev), y[idx].to(dev)
            real, _ = model(xb)
            loss = F.cross_entropy(real, yb, reduction="none")
            if cw is not None:
                loss = loss * cw[yb]
            loss = loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if weight_decay > 0:
                for p in params:
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_([p], 1.0)
            opt.step()
        trained = ep + 1
        if use_val:
            vl = _mean_ce(model, x_val, y_val)
            if verbose:
                print(f"  epoch {trained}: val_loss={vl:.4f}")
            if vl < best:
                best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return trained


def balanced_class_weight(y, n_classes):
    counts = np.bincount(np.asarray(y).astype(int), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return {i: float(w[i]) for i in range(n_classes)}


# --------------------------------------------------------------------------------------
# Student: KL on aux logits only, noise inputs only
# --------------------------------------------------------------------------------------
def distill_aux(student, teacher, noise, epochs, batch_size, lr=1e-3, use_all_logits=False,
                temperature=1.0, eval_fn=None):
    """Minimise KL(softmax(t_aux/T) || softmax(s_aux/T)) on noise.

    Both models run in eval mode, i.e. dropout is off during distillation. Only parameters
    that receive a gradient are updated (the real head of the student stays untouched unless
    `use_all_logits`). `eval_fn(student, epoch)` is called after every epoch if given.
    Returns the list of mean losses per epoch.
    """
    dev = _device(student)
    opt = torch.optim.Adam(student.parameters(), lr=lr, eps=1e-7)
    noise = torch.as_tensor(noise)
    t = float(temperature)
    teacher.eval()
    losses = []
    for ep in range(1, epochs + 1):
        student.eval()
        run = []
        for idx in _batches(len(noise), batch_size, shuffle=True):
            xb = noise[idx].to(dev)
            with torch.no_grad():
                t_real, t_aux = teacher(xb)
                if use_all_logits:
                    t_aux = torch.cat([t_real, t_aux], dim=-1)
            s_real, s_aux = student(xb)
            if use_all_logits:
                s_aux = torch.cat([s_real, s_aux], dim=-1)
            loss = F.kl_div(F.log_softmax(s_aux / t, dim=-1), F.softmax(t_aux / t, dim=-1),
                            reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run.append(loss.item())
        losses.append(float(np.mean(run)))
        if eval_fn is not None:
            eval_fn(student, ep)
    return losses


@torch.no_grad()
def predict_probs(model, x, batch_size=1024):
    model.eval()
    dev = _device(model)
    x = torch.as_tensor(x)
    out = [torch.softmax(model(x[idx].to(dev))[0], dim=-1).cpu() for idx in _batches(len(x), batch_size, False)]
    return torch.cat(out).numpy()
