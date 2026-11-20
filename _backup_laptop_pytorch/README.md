# Information leakage through subliminal learning (smart metering)

A teacher is trained on a hidden household attribute (e.g. single household, family home).
A student that shares the teacher's initialisation is distilled **only on the teacher's auxiliary
logits on noise inputs**, so it never sees a load profile or a label. We then read the attribute
from the student's untouched classification head and measure how much leaked.

## Layout

| Path | What it is |
| --- | --- |
| `run_mnist.py` | MNIST pilot, replicates Cloud et al. (2025) Fig. 10 with controls, CPU-friendly |
| `run_leakage.py` | Smart meter experiment: household-level splits, controls, seeds, leakage score |
| `leakage/` | Shared code: models (MLP, Wang CNN, Transformer), training/distillation, metrics, data, report |
| `prepare_data.py`, `Preprocessor/` | Raw data to weekly snippets (`Household`, `Year_Week`) and labels |
| `run_subliminal_learning.py`, `test_mnist.py`, `base_line_*.py` | Earlier scripts, kept for reference. Note: they split by **week**, so test households also appear in training |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of TensorFlow (fine on a laptop). On the GPU PC or cluster
install `tensorflow[and-cuda]` instead, or use the Docker image.

## Quick checks on a laptop (no GPU)

```bash
# 1) smoke test without any data (~20 s)
python run_leakage.py --synthetic --seeds 2

# 2) MNIST pilot, 5 runs (~4 min), then the full 100 runs overnight (~70 min)
python run_mnist.py --runs 5
python run_mnist.py --runs 100 --preset paper
python run_mnist.py --runs 100 --preset previous     # settings of the old test_mnist.py
```

Results go to `results/<experiment>/`: `runs.csv` (one row per seed and model, written after
every seed, so an interrupted run keeps its data), `summary.csv` (mean, std, 95% CI) and a bar plot
showing both CI and std error bars.

**Error bars.** Cloud et al. show approximate 95% confidence intervals of the mean over 100 runs.
A ±1 std bar over the same runs is about 5x wider (1.96 / sqrt(100) = 0.196). Compare like with like.

## Smart meter runs

Data files are not in the repo. Put the output of `prepare_data.py` into `data/` and, for CER,
regenerate the labels first (missing answers used to become a real class, now they are `-1`):

```bash
python Preprocessor/preprocess_cer_categorical_data.py   # -> data/cer_socio_data_2011.h5
python prepare_data.py                                    # edit the paths at the top for CER
```

Laptop-sized first test (MLP on hourly 7x24 weeks, 10 weeks per household, 3 seeds):

```bash
python run_leakage.py --load-data cer_weekly_structure_2011.h5 \
    --categorical cer_filtered_categorical_2011.h5 \
    --label Single --arch mlp --resolution hourly --max-weeks 10 --seeds 3
python run_leakage.py ... --label Family_Home
```

GPU / cluster: drop `--max-weeks`, raise `--seeds` to 20+, try `--arch wang --resolution native`
and `--arch transformer` (GPU only: about 9 minutes per seed on CPU even with 2 epochs), and vary
`--aux-logits` and `--noise`.

### What is measured

Models per seed: `reference` (untrained shared init), `teacher`, `student_same_aux` (the attack),
`student_diff_aux` (different init, control), `student_same_shuffled` (teacher trained on
household-shuffled labels, control).

Readouts: `heldout_household` (test households, probabilities averaged per household),
`heldout_week`, and `member_week` (weeks of the teacher's training households, i.e. the victims in
the threat model).

Scores (`leakage/metrics.py`): `leakage` = chance-adjusted balanced accuracy (BA - 1/K)/(1 - 1/K),
`gini` = 2*AUC - 1, `leakage_prior` = leakage with the binary threshold set to the class prior.
All three are 0 at chance.
