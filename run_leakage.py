"""Attribute leakage through aux-only distillation on smart meter data.

Per seed (= one household-level split + one shared initialisation):
  reference              untrained shared init (chance check)
  baseline_level_only    logistic regression on log1p(mean raw kWh of the week) only, fitted on the
                         training households' weeks: how much is explained by the load level alone
  baseline_linear        logistic regression on the flattened, standardised week (what the networks see)
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
see leakage/metrics.py for why; gini_abs = |gini| (a random head can separate with either sign);
leakage_vs_ref / gini_vs_ref = score minus that of the untrained reference (same seed and readout);
dist_from_init (teacher rows) = ||w_teacher - w_ref|| / ||w_ref|| over all parameters.

--remove-level divides every week by its own mean, so only the shape of the profile remains.
--patience 0 disables early stopping (the teacher trains exactly --teacher-epochs epochs).

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
from sklearn.linear_model import LogisticRegression

from leakage.data import household_split, load_weekly, standardise, synthetic_weekly, to_hourly
from leakage.metrics import chance_baselines, classification_metrics
from leakage.models import clone, make_builder, weight_distance
from leakage.report import bar_plot, summarise
from leakage.training import (balanced_class_weight, distill_aux, get_device, make_noise, predict_probs,
                              set_seed, to_tensor, train_on_real_logits)

ORDER = ["reference", "baseline_level_only", "baseline_linear", "teacher", "student_same_aux", "student_diff_aux", "student_same_shuffled"]


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
    d.add_argument("--remove-level", action="store_true",
                   help="divide every week by its own mean consumption (keep only the profile shape)")
    m = p.add_argument_group("model")
    m.add_argument("--arch", choices=["mlp", "wang", "transformer", "gru"], default="mlp")
    m.add_argument("--dropout", type=float, default=None,
                   help="dropout for mlp / wang (default: architecture default)")
    m.add_argument("--aux-logits", type=int, default=3)
    m.add_argument("--teacher-epochs", type=int, default=50)
    m.add_argument("--patience", type=int, default=5, help="0 = no early stopping")
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
    r.add_argument("--device", default="auto", help="auto, cpu or cuda")
    r.add_argument("--out", default="results/leakage")
    r.add_argument("--name", default=None)
    return p.parse_args()


def household_level(y, probs, hh):
    df = pd.DataFrame(probs).assign(hh=hh, y=y)
    g = df.groupby("hh")
    return g["y"].first().values, g[list(range(probs.shape[1]))].mean().values


def add_relative_scores(df):
    """gini_abs, and leakage / gini minus those of the untrained reference (same seed and readout)."""
    df["gini_abs"] = df["gini"].abs()
    ref = df[df.model == "reference"].set_index(["seed", "readout"])
    key = pd.MultiIndex.from_frame(df[["seed", "readout"]])
    for m in ["leakage", "gini"]:
        df[f"{m}_vs_ref"] = df[m].values - ref[m].reindex(key).values
    return df


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}, torch {torch.__version__}")
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

    name = args.name or (f"{dataset}_{args.label}_{args.arch}_{args.resolution}_{args.noise}_m{args.aux_logits}"
                         + ("_nolevel" if args.remove_level else ""))
    model_kwargs = {}
    if args.dropout is not None:
        if args.arch in ("mlp", "wang"):
            model_kwargs["dropout"] = args.dropout
        else:
            print(f"Note: --dropout is only used for mlp and wang, ignored for {args.arch}")
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    rows = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.perf_counter()
        set_seed(seed)
        rng = np.random.default_rng(seed)
        sp = household_split(x, y, hh, seed, max_weeks_per_household=args.max_weeks)
        # raw load level per week (before any scaling and before --remove-level)
        lvl_train, lvl_test = (np.log1p(np.clip(a, 0, None).reshape(len(a), -1).mean(axis=1, keepdims=True))
                               for a in (sp.x_train, sp.x_test))
        if args.remove_level:
            for part in ("x_train", "x_val", "x_test"):
                a = getattr(sp, part)
                setattr(sp, part, (a / (a.mean(axis=(1, 2), keepdims=True) + 1e-8)).astype("float32"))
        sp = standardise(sp, args.scaling)
        k = sp.n_classes
        build = make_builder(args.arch, sp.x_train.shape[1:], k, args.aux_logits, device=device, **model_kwargs)

        x_tr, y_tr = to_tensor(sp.x_train, device), to_tensor(sp.y_train, device, torch.long)
        x_va, y_va = to_tensor(sp.x_val, device), to_tensor(sp.y_val, device, torch.long)
        x_te = to_tensor(sp.x_test, device)
        member_idx = rng.choice(len(sp.x_train), min(len(sp.x_train), len(sp.x_test) * 2), replace=False)
        x_mem = x_tr[torch.as_tensor(member_idx, device=device)]
        # inputs per source: (tensor, standardised numpy, raw level feature)
        sources = {"test": (x_te, sp.x_test, lvl_test),
                   "member": (x_mem, sp.x_train[member_idx], lvl_train[member_idx])}
        readouts = {
            "heldout_week": ("test", sp.y_test, None),
            "heldout_household": ("test", sp.y_test, sp.hh_test),
            "member_week": ("member", sp.y_train[member_idx], None),
        }
        base = chance_baselines(sp.y_test, k)
        prior = float(np.mean(sp.y_train == 1)) if k == 2 else None

        def evaluate(model, model_name, extra=None, predict=None):
            """predict(source) -> class probabilities; default: the model's real_logits head."""
            predict = predict or (lambda src: predict_probs(model, sources[src][0]))
            cache = {}
            for ro, (src, yr, hr) in readouts.items():
                if src not in cache:
                    cache[src] = predict(src)
                probs = cache[src]
                yy = yr
                if hr is not None:
                    yy, probs = household_level(yr, probs, hr)
                rows.append({"seed": seed, "model": model_name, "readout": ro,
                             **classification_metrics(yy, probs, k, prior), **base, **(extra or {})})

        ref = build()
        evaluate(ref, "reference")

        # non-network baselines (sklearn), fitted on the training households' weeks only
        def flat(a):
            return a.reshape(len(a), -1)

        lr_level = LogisticRegression(class_weight="balanced", max_iter=1000).fit(lvl_train, sp.y_train)
        evaluate(None, "baseline_level_only", predict=lambda src: lr_level.predict_proba(sources[src][2]))
        lr_lin = LogisticRegression(class_weight="balanced", max_iter=1000).fit(flat(sp.x_train), sp.y_train)
        evaluate(None, "baseline_linear", predict=lambda src: lr_lin.predict_proba(flat(sources[src][1])))

        cw = balanced_class_weight(sp.y_train, k)
        teacher = clone(build, ref)
        ep = train_on_real_logits(teacher, x_tr, y_tr, args.teacher_epochs, args.batch_size, args.lr,
                                  x_val=x_va, y_val=y_va, patience=args.patience, class_weight=cw)
        evaluate(teacher, "teacher", {"epochs_trained": ep, "dist_from_init": weight_distance(teacher, ref)})

        n_noise = args.n_noise or len(sp.x_train)
        noise = to_tensor(make_noise(args.noise, n_noise, sp.x_train.shape[1:], rng, x_ref=sp.x_train), device)
        d = dict(epochs=args.student_epochs, batch_size=args.batch_size, lr=args.lr)

        s = clone(build, ref)
        evaluate(s, "student_same_aux", {"final_kl": distill_aux(s, teacher, noise, **d)[-1]})

        s = build()
        evaluate(s, "student_diff_aux", {"final_kl": distill_aux(s, teacher, noise, **d)[-1]})

        if not args.no_shuffled_control:
            # shuffle labels between households (keeps class balance and within-household consistency)
            uh = np.unique(sp.hh_train)
            lab = pd.Series(sp.y_train, index=sp.hh_train).groupby(level=0).first().loc[uh].values
            perm = dict(zip(uh, rng.permutation(lab)))
            y_shuf = to_tensor(np.array([perm[h] for h in sp.hh_train]), device, torch.long)
            t_shuf = clone(build, ref)
            train_on_real_logits(t_shuf, x_tr, y_shuf, ep, args.batch_size, args.lr, class_weight=cw)
            s = clone(build, ref)
            evaluate(s, "student_same_shuffled", {"final_kl": distill_aux(s, t_shuf, noise, **d)[-1]})

        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
        cur = pd.DataFrame([r for r in rows if r["seed"] == seed and r["readout"] == "heldout_household"])
        print(f"[seed {seed}] {time.perf_counter() - t0:.0f}s teacher epochs={ep} | leakage (held-out households): "
              + ", ".join(f"{m}={v:.3f}" for m, v in zip(cur.model, cur.leakage)), flush=True)

    df = add_relative_scores(pd.DataFrame(rows))
    df.to_csv(out / "runs.csv", index=False)
    metrics = ["leakage", "leakage_prior", "gini", "gini_abs", "leakage_vs_ref", "gini_vs_ref", "balanced_acc",
               "auc", "acc", "mcc", "dist_from_init", "epochs_trained"]
    summ = summarise(df, ["readout", "model"], [c for c in metrics if c in df.columns])
    summ.to_csv(out / "summary.csv", index=False)
    show = summ[summ.readout.isin(["heldout_household", "member_week"])]
    print(show[["readout", "model", "leakage_mean", "leakage_std", "leakage_vs_ref_mean", "leakage_prior_mean",
                "gini_mean", "gini_abs_mean", "gini_ci95", "n_runs"]].to_string(index=False))
    for ro in ["heldout_household", "member_week"]:
        bar_plot(summ[summ.readout == ro], "leakage", ORDER, out / f"leakage_{ro}.png", chance=0.0,
                 ylabel="leakage (chance-adjusted balanced acc.)", title=f"{name} – {ro}")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
