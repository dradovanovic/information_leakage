"""MNIST pilot: subliminal learning of an MLP classifier from aux logits on noise.

Reproduces Cloud et al. (2025), Sec. 6.2 / Fig. 10, and adds controls. PyTorch, CPU-friendly.

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
from leakage.data import load_mnist
from leakage.models import clone, make_builder
from leakage.report import bar_plot, summarise
from leakage.training import (distill_aux, get_device, make_noise, predict_probs, set_seed, to_tensor,
                              train_on_real_logits)

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
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--data-folder", default=str(Path(__file__).resolve().parent / "data"))
    p.add_argument("--out", default="results/mnist")
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
    print(f"Device: {device}, torch {torch.__version__}")
    (x_tr, y_tr), (x_te, y_te_np) = load_mnist(args.data_folder)
    shape = x_tr.shape[1:]
    x_tr, y_tr = to_tensor(x_tr, device), to_tensor(y_tr, device, torch.long)
    x_te = to_tensor(x_te, device)

    # validation split only for the "previous" preset (early stopping)
    x_val = y_val = None
    if cfg["val_patience"]:
        x_val, y_val = x_tr[-12_000:], y_tr[-12_000:]
        x_fit, y_fit = x_tr[:-12_000], y_tr[:-12_000]
    else:
        x_fit, y_fit = x_tr, y_tr

    build = make_builder("mlp", shape, 10, cfg["aux_logits"], device=device)
    rows = []
    rows_path = out / "runs.csv"
    for seed in range(args.first_seed, args.first_seed + args.runs):
        t0 = time.perf_counter()
        set_seed(seed)
        rng = np.random.default_rng(seed)
        ref = build()

        def evaluate(model, name, extra=None):
            m = classification_metrics(y_te_np, predict_probs(model, x_te), 10)
            rows.append({"seed": seed, "model": name, **m, **(extra or {})})
            return m

        evaluate(ref, "reference")

        teacher = clone(build, ref)
        ep = train_on_real_logits(teacher, x_fit, y_fit, cfg["teacher_epochs"], cfg["batch_size"], cfg["lr"],
                                  cfg["weight_decay"], x_val, y_val, cfg["val_patience"])
        evaluate(teacher, "teacher", {"epochs_trained": ep})

        noise = to_tensor(make_noise(cfg["noise"], cfg["n_noise"], shape, rng), device)
        d = dict(epochs=cfg["student_epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"])

        s = clone(build, ref)
        loss = distill_aux(s, teacher, noise, **d)
        evaluate(s, "student_same_aux", {"final_kl": loss[-1]})

        if args.all_logits:
            s = clone(build, ref)
            loss = distill_aux(s, teacher, noise, use_all_logits=True, **d)
            evaluate(s, "student_same_all", {"final_kl": loss[-1]})

        s = build()  # fresh random init
        loss = distill_aux(s, teacher, noise, **d)
        evaluate(s, "student_diff_aux", {"final_kl": loss[-1]})

        if not args.no_shuffled_control:
            t_shuf = clone(build, ref)
            y_shuf = y_fit[torch.as_tensor(rng.permutation(len(y_fit)), device=device)]
            train_on_real_logits(t_shuf, x_fit, y_shuf, cfg["teacher_epochs"], cfg["batch_size"],
                                 cfg["lr"], cfg["weight_decay"])
            s = clone(build, ref)
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
