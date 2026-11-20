"""MNIST pilot: subliminal learning of an MLP classifier from aux logits on noise.

Reproduces Cloud et al. (2025), Sec. 6.2 / Fig. 10, and adds controls. CPU-friendly.

Models evaluated per run (all on the MNIST test set, via the 10 regular logits):
  reference            untrained shared initialisation (chance level)
  teacher              reference trained 5 epochs on MNIST (regular logits only)
  student_same_aux     copy of reference, distilled on the teacher's aux logits on noise
  student_diff_aux     fresh random init, same distillation (paper: "cross-model aux only")
  student_same_shuffled  control: copy of reference, distilled from a teacher trained on
                       shuffled labels (same init). Tells whether anything but the labels transfers.
  student_same_all     (with --all-logits) distilled on all 13 logits

Error bars: the paper shows 95% CIs of the mean over 100 runs. summary.csv reports mean,
std and ci95 so both can be shown; bar plot has both side by side.

Examples
  python run_mnist.py --runs 5                       # quick laptop check (~3 min on CPU)
  python run_mnist.py --runs 100 --preset paper      # full replication
  python run_mnist.py --runs 100 --preset previous   # settings of the old test_mnist.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from leakage.metrics import classification_metrics
from leakage.models import MLP, clone
from leakage.report import bar_plot, summarise
from leakage.training import distill_aux, get_device, make_noise, predict_probs, set_seed, train_on_real_logits

PRESETS = {
    # Cloud et al.: 5 + 5 epochs, MLP 256-256, m=3. Batch size and lr are not stated in the paper.
    "paper": dict(teacher_epochs=5, student_epochs=5, batch_size=128, lr=1e-3, weight_decay=0.0,
                  aux_logits=3, noise="uniform", n_noise=60_000, val_patience=None),
    # Settings of the previous test_mnist.py (seed-stability figure in the IT-Talk)
    "previous": dict(teacher_epochs=30, student_epochs=10, batch_size=1024, lr=1e-3, weight_decay=1e-4,
                     aux_logits=3, noise="uniform", n_noise=60_000, val_patience=15),
}
ORDER = ["reference", "teacher", "student_same_aux", "student_same_all", "student_diff_aux",
         "student_same_shuffled"]


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist(path):
    """Same file and split as tf.keras.datasets.mnist (60k train / 10k test), cached locally."""
    path = Path(path)
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {MNIST_URL} -> {path}")
        urllib.request.urlretrieve(MNIST_URL, path)
    with np.load(path) as f:
        return (f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(PRESETS), default="paper")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--teacher-epochs", type=int)
    p.add_argument("--student-epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--aux-logits", type=int)
    p.add_argument("--noise", choices=["uniform", "gaussian"])
    p.add_argument("--all-logits", action="store_true", help="also train a student on all logits")
    p.add_argument("--no-shuffled-control", action="store_true")
    p.add_argument("--out", default="results/mnist")
    p.add_argument("--mnist", default=".cache/mnist.npz", help="local copy of mnist.npz (downloaded once)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(PRESETS[args.preset])
    for k in ["teacher_epochs", "student_epochs", "batch_size", "aux_logits", "noise"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    out = Path(args.out) / f"{args.preset}_t{cfg['teacher_epochs']}_s{cfg['student_epochs']}_bs{cfg['batch_size']}_m{cfg['aux_logits']}_{cfg['noise']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, **vars(args)}, indent=2))
    print(f"Config: {cfg}\nWriting to {out}")

    device = get_device(args.device)
    print(f"PyTorch {torch.__version__}, device {device}")
    (x_tr, y_tr), (x_te, y_te) = load_mnist(args.mnist)
    x_tr = (x_tr.astype("float32") / 255.0)
    x_te = (x_te.astype("float32") / 255.0)
    shape = x_tr.shape[1:]

    # validation split only for the "previous" preset (early stopping)
    x_val = y_val = None
    if cfg["val_patience"]:
        x_val, y_val = x_tr[-12_000:], y_tr[-12_000:]
        x_fit, y_fit = x_tr[:-12_000], y_tr[:-12_000]
    else:
        x_fit, y_fit = x_tr, y_tr

    build = lambda name=None: MLP(shape, 10, aux_logits=cfg["aux_logits"]).to(device)  # noqa: E731
    rows = []
    rows_path = out / "runs.csv"
    for seed in range(args.first_seed, args.first_seed + args.runs):
        t0 = time.perf_counter()
        set_seed(seed)
        rng = np.random.default_rng(seed)
        ref = build("reference")

        def evaluate(model, name, extra=None):
            m = classification_metrics(y_te, predict_probs(model, x_te), 10)
            rows.append({"seed": seed, "model": name, **m, **(extra or {})})
            return m

        evaluate(ref, "reference")

        teacher = clone(ref)
        ep = train_on_real_logits(teacher, x_fit, y_fit, cfg["teacher_epochs"], cfg["batch_size"], cfg["lr"],
                                  cfg["weight_decay"], x_val, y_val, cfg["val_patience"])
        evaluate(teacher, "teacher", {"epochs_trained": ep})

        noise = make_noise(cfg["noise"], cfg["n_noise"], shape, rng)
        d = dict(epochs=cfg["student_epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"])

        s = clone(ref)
        loss = distill_aux(s, teacher, noise, **d)
        evaluate(s, "student_same_aux", {"final_kl": loss[-1]})

        if args.all_logits:
            s = clone(ref)
            loss = distill_aux(s, teacher, noise, use_all_logits=True, **d)
            evaluate(s, "student_same_all", {"final_kl": loss[-1]})

        s = build("student_diff_aux")  # fresh random init
        loss = distill_aux(s, teacher, noise, **d)
        evaluate(s, "student_diff_aux", {"final_kl": loss[-1]})

        if not args.no_shuffled_control:
            t_shuf = clone(ref)
            train_on_real_logits(t_shuf, x_fit, rng.permutation(y_fit), cfg["teacher_epochs"], cfg["batch_size"],
                                 cfg["lr"], cfg["weight_decay"])
            s = clone(ref)
            loss = distill_aux(s, t_shuf, noise, **d)
            evaluate(s, "student_same_shuffled", {"final_kl": loss[-1]})

        pd.DataFrame(rows).to_csv(rows_path, index=False)  # save after every run
        last = {r["model"]: round(r["acc"], 3) for r in rows if r["seed"] == seed}
        print(f"[seed {seed}] {time.perf_counter() - t0:.0f}s  {last}", flush=True)

    df = pd.DataFrame(rows)
    summ = summarise(df, ["model"], ["acc", "balanced_acc", "leakage"])
    summ["order"] = summ["model"].map({m: i for i, m in enumerate(ORDER)})
    summ = summ.sort_values("order").drop(columns="order")
    summ.to_csv(out / "summary.csv", index=False)
    print(summ[["model", "acc_mean", "acc_std", "acc_ci95", "n_runs"]].to_string(index=False))
    bar_plot(summ, "acc", ORDER, out / "accuracy.png", chance=0.1, ylabel="MNIST test accuracy",
             title=f"Subliminal learning on MNIST ({args.preset}, {len(df.seed.unique())} runs)")
    print(f"Saved {rows_path}, {out / 'summary.csv'}, {out / 'accuracy.png'}")


if __name__ == "__main__":
    main()
