"""Attribute leakage through aux-only distillation on smart meter data.

Per seed (= one household-level split + one shared initialisation):
  reference              untrained shared init (chance check)
  teacher                trained on the hidden attribute (real logits only), class-balanced,
                         early stopping on validation HOUSEHOLDS
  student_same_aux       copy of reference, distilled on the teacher's aux logits on noise
  student_diff_aux       fresh init, same distillation (control: init must be shared)
  student_same_shuffled  copy of reference, distilled from a teacher trained on household-shuffled
                         labels (control: does anything except the attribute transfer?)

Readouts (attribute predicted from the student's untouched real_logits head):
  heldout_week       weeks of test households (never seen by the teacher)
  heldout_household  same households, probabilities averaged over each household's weeks
  member_week        weeks of the teacher's training households (the threat-model victims)

Main score: leakage = chance-adjusted balanced accuracy, (BA - 1/K) / (1 - 1/K).
Also reported: gini = 2*AUC - 1 and leakage_prior (binary threshold set to the class prior),
see leakage/metrics.py for why.

Examples
  # smoke test without data (CPU, ~1 min)
  python run_leakage.py --synthetic --seeds 2
  # CER, laptop-sized: MLP on hourly weeks, 10 weeks per household, 3 seeds
  python run_leakage.py --load-data cer_weekly_structure_2011.h5 --categorical cer_filtered_categorical_2011.h5 \
      --label Single --arch mlp --resolution hourly --max-weeks 10 --seeds 3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from leakage.data import household_split, load_weekly, standardise, synthetic_weekly, to_hourly
from leakage.metrics import chance_baselines, classification_metrics
from leakage.models import clone, make_builder
from leakage.report import bar_plot, summarise
from leakage.training import (balanced_class_weight, distill_aux, get_device, make_noise, predict_probs,
                              set_seed, train_on_real_logits)

ORDER = ["reference", "teacher", "student_same_aux", "student_diff_aux", "student_same_shuffled"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--data-folder", default=str(Path(__file__).resolve().parent / "data"))
    d.add_argument("--load-data", help="weekly load h5 inside --data-folder")
    d.add_argument("--categorical", help="categorical h5 inside --data-folder")
    d.add_argument("--label", default="Single", help="attribute column, e.g. Single, Family_Home, Apartment")
    d.add_argument("--synthetic", action="store_true", help="planted-attribute fake data (smoke test)")
    d.add_argument("--resolution", choices=["native", "hourly"], default="hourly")
    d.add_argument("--scaling", choices=["global", "log", "none"], default="log")
    d.add_argument("--max-weeks", type=int, default=None, help="weeks per household (speed)")
    m = p.add_argument_group("model")
    m.add_argument("--arch", choices=["mlp", "wang", "transformer"], default="mlp")
    m.add_argument("--aux-logits", type=int, default=3)
    m.add_argument("--teacher-epochs", type=int, default=50)
    m.add_argument("--patience", type=int, default=5)
    m.add_argument("--student-epochs", type=int, default=5)
    m.add_argument("--batch-size", type=int, default=128)
    m.add_argument("--lr", type=float, default=1e-3)
    m.add_argument("--noise", default="gaussian",
                   choices=["uniform", "gaussian", "uniform_centered", "gaussian_matched"])
    m.add_argument("--n-noise", type=int, default=None, help="default: number of training weeks")
    r = p.add_argument_group("runs")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--first-seed", type=int, default=0)
    r.add_argument("--no-shuffled-control", action="store_true")
    r.add_argument("--out", default="results/leakage")
    r.add_argument("--name", default=None)
    r.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    r.add_argument("--threads", type=int, default=None, help="CPU threads for torch (default: torch default)")
    return p.parse_args()


def household_level(y, probs, hh):
    df = pd.DataFrame(probs).assign(hh=hh, y=y)
    g = df.groupby("hh")
    return g["y"].first().values, g[list(range(probs.shape[1]))].mean().values


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = get_device(args.device)
    print(f"PyTorch {torch.__version__}, device {device}, {torch.get_num_threads()} CPU threads", flush=True)
    if args.synthetic:
        x, y, hh = synthetic_weekly(seed=12345)
        dataset = "synthetic"
    else:
        if not (args.load_data and args.categorical):
            raise SystemExit("Give --load-data and --categorical, or use --synthetic")
        folder = Path(args.data_folder)
        x, y, hh = load_weekly(folder / args.load_data, folder / args.categorical, args.label)
        dataset = Path(args.load_data).stem
    if args.resolution == "hourly":
        x = to_hourly(x)
    print(f"Data: {x.shape[0]} weeks, {len(np.unique(hh))} households, week shape {x.shape[1:]}, "
          f"class counts (weeks) {np.bincount(y).tolist()}")

    name = args.name or f"{dataset}_{args.label}_{args.arch}_{args.resolution}_{args.noise}_m{args.aux_logits}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    rows = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.perf_counter()
        set_seed(seed)
        rng = np.random.default_rng(seed)
        sp = standardise(household_split(x, y, hh, seed, max_weeks_per_household=args.max_weeks), args.scaling)
        k = sp.n_classes
        build = make_builder(args.arch, sp.x_train.shape[1:], k, args.aux_logits, device=device)

        member_idx = rng.choice(len(sp.x_train), min(len(sp.x_train), len(sp.x_test) * 2), replace=False)
        readouts = {
            "heldout_week": (sp.x_test, sp.y_test, None),
            "heldout_household": (sp.x_test, sp.y_test, sp.hh_test),
            "member_week": (sp.x_train[member_idx], sp.y_train[member_idx], None),
        }
        base = chance_baselines(sp.y_test, k)
        prior = float(np.mean(sp.y_train == 1)) if k == 2 else None

        def evaluate(model, model_name, extra=None):
            cache = {}
            for ro, (xr, yr, hr) in readouts.items():
                key = id(xr)
                if key not in cache:
                    cache[key] = predict_probs(model, xr)
                probs = cache[key]
                yy = yr
                if hr is not None:
                    yy, probs = household_level(yr, probs, hr)
                rows.append({"seed": seed, "model": model_name, "readout": ro, **classification_metrics(yy, probs, k, prior),
                             **base, **(extra or {})})

        ref = build("reference")
        evaluate(ref, "reference")

        cw = balanced_class_weight(sp.y_train, k)
        teacher = clone(ref)
        ep = train_on_real_logits(teacher, sp.x_train, sp.y_train, args.teacher_epochs, args.batch_size, args.lr,
                                  x_val=sp.x_val, y_val=sp.y_val, patience=args.patience, class_weight=cw)
        evaluate(teacher, "teacher", {"epochs_trained": ep})

        n_noise = args.n_noise or len(sp.x_train)
        noise = make_noise(args.noise, n_noise, sp.x_train.shape[1:], rng, x_ref=sp.x_train)
        d = dict(epochs=args.student_epochs, batch_size=args.batch_size, lr=args.lr)

        s = clone(ref)
        evaluate(s, "student_same_aux", {"final_kl": distill_aux(s, teacher, noise, **d)[-1]})

        s = build("student_diff_aux")
        evaluate(s, "student_diff_aux", {"final_kl": distill_aux(s, teacher, noise, **d)[-1]})

        if not args.no_shuffled_control:
            # shuffle labels between households (keeps class balance and within-household consistency)
            uh = np.unique(sp.hh_train)
            lab = pd.Series(sp.y_train, index=sp.hh_train).groupby(level=0).first().loc[uh].values
            perm = dict(zip(uh, rng.permutation(lab)))
            y_shuf = np.array([perm[h] for h in sp.hh_train])
            t_shuf = clone(ref)
            train_on_real_logits(t_shuf, sp.x_train, y_shuf, ep, args.batch_size, args.lr, class_weight=cw)
            s = clone(ref)
            evaluate(s, "student_same_shuffled", {"final_kl": distill_aux(s, t_shuf, noise, **d)[-1]})

        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
        cur = pd.DataFrame([r for r in rows if r["seed"] == seed and r["readout"] == "heldout_household"])
        print(f"[seed {seed}] {time.perf_counter() - t0:.0f}s teacher epochs={ep} | leakage (held-out households): "
              + ", ".join(f"{m}={v:.3f}" for m, v in zip(cur.model, cur.leakage)), flush=True)

    df = pd.DataFrame(rows)
    summ = summarise(df, ["readout", "model"], ["leakage", "leakage_prior", "gini", "balanced_acc", "auc", "acc", "mcc"])
    summ.to_csv(out / "summary.csv", index=False)
    show = summ[summ.readout.isin(["heldout_household", "member_week"])]
    print(show[["readout", "model", "leakage_mean", "leakage_std", "leakage_prior_mean", "gini_mean", "gini_ci95", "n_runs"]]
          .to_string(index=False))
    for ro in ["heldout_household", "member_week"]:
        bar_plot(summ[summ.readout == ro], "leakage", ORDER, out / f"leakage_{ro}.png", chance=0.0,
                 ylabel="leakage (chance-adjusted balanced acc.)", title=f"{name} – {ro}")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
