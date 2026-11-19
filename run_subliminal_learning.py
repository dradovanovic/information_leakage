import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from Models.CNNs import Wang_CNN
from tensorflow.keras.optimizers import AdamW
from Models.Transformer import TransformerKeras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from diagnostic import compare_weekly_snippets_heatmaps, compare_last_layer_with_real_and_aux
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def read_data(load_data_path: Path, categorical_data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f" \n1.1 Reading Data ")
    print(f" \tStarted reading preprocessed load data from file: {load_data_path}")
    if not load_data_path.exists():
        return False, None, f"File not found: {load_data_path}"
    try:
        load_data = pd.read_hdf(load_data_path)
        print(f' \tShape of load data is: {load_data.shape}')
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        return False, None, f"Failed to read HDF: {e!s}"

    print(f" \tStarted reading preprocessed categorical data from file: {categorical_data_path}")
    if not categorical_data_path.exists():
        return False, None, f"File not found: {categorical_data_path}"
    try:
        categorical_data = pd.read_hdf(categorical_data_path)
        print(f' \tShape of categorical data is: {categorical_data.shape}')
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        return False, None, f"Failed to read HDF: {e!s}"
    return load_data, categorical_data


def check_label_distribution(categorical_data: pd.DataFrame, label_to_predict: str, mode='before_filtering'):
    if mode == 'before_filtering':
        print(f" \n2.1 Started Label preparation for {label_to_predict} ")
        print(
            f" \tTotal number of classes in label '{label_to_predict}': {categorical_data[label_to_predict].nunique()}")
        print(f" \tChecking label distribution")
    else:
        print(
            f" \tTotal number of classes in label '{label_to_predict}': {categorical_data[label_to_predict].nunique()}")
        print(f" \tChecking label distribution after filtering")
    label_counts = categorical_data[label_to_predict].value_counts()
    total_counts = label_counts.sum()
    for label, count in label_counts.items():
        percentage = (count / total_counts) * 100
        print(f' \t\tLabel {label}: Count = {count}, Percentage = {percentage:.2f}%')


def filter_households_without_label(load_data: pd.DataFrame,
                                    categorical_data: pd.DataFrame,
                                    label_to_predict: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f" \n2.2 Filtered households without label '{label_to_predict}'")
    # Get households with valid labels
    valid_households = categorical_data[categorical_data[label_to_predict] != -1]['Household'].unique()
    # Filter load data to keep only valid households
    filtered_load_data = load_data[load_data.index.get_level_values('Household').isin(valid_households)]
    print(f' \tShape of load data after filtering households without label: {filtered_load_data.shape}')
    filtered_categorical_data = categorical_data[categorical_data['Household'].isin(valid_households)]
    print(
        f' \tShape of categorical data after filtering households without label: {filtered_categorical_data[label_to_predict].shape}')
    return filtered_load_data, filtered_categorical_data


def get_train_test_data(load_data: pd.DataFrame,
                        categorical_data: pd.DataFrame,
                        split_ratio=0.9,
                        seed=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f" \n2.3 Splitting data into train and test data with ratio {split_ratio} ")

    # Get household ids and calendar weeks form the multi index
    load_data = load_data.reset_index()

    X_train, X_test = train_test_split(load_data,
                                       train_size=split_ratio,
                                       shuffle=True,
                                       random_state=seed,
                                       stratify=load_data["Household"]
                                       )
    y_train = pd.DataFrame(
        index=X_train.index,
        data=X_train['Household'].values,
        columns=['Household'],
    )

    y_test = pd.DataFrame(
        index=X_test.index,
        data=X_test['Household'].values,
        columns=['Household'],
    )

    y_train = pd.merge(y_train, categorical_data, on='Household', how='left').set_index(y_train.index)
    y_test = pd.merge(y_test, categorical_data, on='Household', how='left').set_index(y_test.index)
    return X_train, X_test, y_train, y_test


def remove_unnecessary_columns(x_train: pd.DataFrame,
                               x_test: pd.DataFrame,
                               y_train: pd.DataFrame,
                               y_test: pd.DataFrame,
                               label_to_predict: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Bring the data in a proper form for later use
    cols_to_drop = ['index', 'Household', 'Year_Week']
    train_index = pd.MultiIndex.from_arrays([x_train['Household'], x_train['Year_Week']],
                                            names=["Household", "Year_Week"])
    test_index = pd.MultiIndex.from_arrays([x_test['Household'], x_test['Year_Week']],
                                           names=["Household", "Year_Week"])
    x_train = (x_train.set_index(train_index).drop(columns=[c for c in cols_to_drop if c in x_train.columns]))
    x_test = (x_test.set_index(test_index).drop(columns=[c for c in cols_to_drop if c in x_test.columns]))
    print(f" \tTRAIN SHAPE after removing unnecessary columns:")
    print(f' \t\tShape of train data: {x_train.shape}')
    print(f' \t\tShape of categorical train data: {y_train[label_to_predict].shape}')
    print(f" \tTEST SHAPE after removing unnecessary columns:")
    print(f' \t\tShape of test data: {x_test.shape}')
    print(f' \t\tShape of categorical test data: {y_test[label_to_predict].shape}')
    return x_train, x_test


def prepare_data_for_subliminal_learning(load_path: Path,
                                         cat_path: Path,
                                         time_resolution: str,
                                         label_to_predict: str,
                                         scaling_on=False,
                                         seed = 42) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, int]:
    print(
        f" \n##################################### 1.Step - Preparing Data for Subliminal Learning ########################################## ")
    # Read both datasets
    load_data, categorical_data = read_data(load_path, cat_path)

    # Label preparation and Household filtering based on label availability
    check_label_distribution(categorical_data, label_to_predict, mode='before_filtering')
    filtered_load_data, filtered_categorical_data = filter_households_without_label(load_data,
                                                                                    categorical_data,
                                                                                    label_to_predict)
    check_label_distribution(filtered_categorical_data, label_to_predict, mode='after_filtering')
    number_of_classes = filtered_categorical_data[label_to_predict].nunique()

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = get_train_test_data(filtered_load_data,
                                                           filtered_categorical_data,
                                                           split_ratio=0.9,
                                                           seed=seed)

    # Remove the unnecessary columns for later usage
    x_train_prepared, x_test_prepared = remove_unnecessary_columns(x_train,
                                                                   x_test,
                                                                   y_train,
                                                                   y_test,
                                                                   label_to_predict=label_to_predict)

    if scaling_on:
        print(f" \n2.4 Scaling data using StandardScaler")
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_prepared)
        x_test_scaled = scaler.transform(x_test_prepared)
    else:
        x_train_scaled = x_train_prepared.values
        x_test_scaled = x_test_prepared.values

    print(f" \n2.4 Shaping data to trainable weekly format for resolution of {time_resolution} ")
    time_resolution_int = int(time_resolution.replace('min', ''))
    values_per_day = int(24 * (60 / time_resolution_int))
    x_train_final = x_train_scaled.reshape(x_train_scaled.shape[0], 7, values_per_day, 1)
    x_test_final = x_test_scaled.reshape(x_test_scaled.shape[0], 7, values_per_day, 1)

    print(f' \tFinal shape of x_train: {x_train_final.shape}')
    print(f' \tFinal shape of x_test: {x_test_final.shape}')
    print(f" \n✅ Finished data preparation for Subliminal Learning! ")
    return x_train_final, x_test_final, y_train[label_to_predict], y_test[label_to_predict], number_of_classes


def plot_history_loss(history, save_path: str, title="Training vs Validation Loss"):
    # accept History or dict-like object
    hist = history.history if hasattr(history, "history") else history

    # check if entries exist
    if "loss" not in hist:
        raise KeyError("'loss' not found in history.")
    if "val_loss" not in hist:
        raise KeyError("'val_loss' not found in history (did you pass validation_data=... to fit?)")

    epochs = range(1, len(hist["loss"]) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, hist["loss"], label="loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"💾 Learning curve save under {save_path}")

def plot_distillation_loss(
    step_losses,
    epoch_losses=None,
    log_every=200,
    title="Distillation loss (KL)",
    loss_curve_path=None
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # ---- Left: step loss
    ax0 = axes[0]
    if step_losses:
        x_steps = np.arange(1, len(step_losses) + 1) * log_every
        ax0.plot(x_steps, step_losses)
    ax0.set_title("Step loss")
    ax0.set_xlabel(f"Training step (logged every {log_every})")
    ax0.set_ylabel("KL loss")
    ax0.grid(True, alpha=0.3)

    # ---- Right: epoch loss
    ax1 = axes[1]
    if epoch_losses is not None and len(epoch_losses) > 0:
        x_epochs = np.arange(1, len(epoch_losses) + 1)
        ax1.plot(x_epochs, epoch_losses)
    ax1.set_title("Mean loss per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("KL loss")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"💾 Loss curve save under {loss_curve_path}")

def noise_jittered(x_train, batch_size=128, jitter_std=0.1):
    n = x_train.shape[0]
    num_samples = x_train.shape[0]
    idx = np.random.randint(0, n, size=num_samples)
    base = x_train[idx].copy()
    noise = np.random.randn(*base.shape).astype("float32") * jitter_std
    noisy = base + noise
    ds = tf.data.Dataset.from_tensor_slices(noisy).shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def nosie_uniform_centered(x_train, batch_size=128, k_std=3.0, seed=42):
    value_min = np.min(x_train)
    value_max = np.max(x_train)
    mu = np.mean(x_train)
    sigma = np.std(x_train)

    # Uniform in [μ−kσ, μ+kσ]
    # [0.10−0.63, 0.10+0.63]=[−0.53, 0.73], then clipped to [0, 5.17] ⇒ [0, 0.73]
    low = max(mu - k_std * sigma, value_min)
    high = min(mu + k_std * sigma, value_max)

    # uniform noise in the centered range
    rng = np.random.default_rng(seed)
    noise = rng.uniform(low, high, size=x_train.shape).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices(noise).shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------------
# Teacher training (real logits only)
# -------------------------------
def train_teacher(
        teacher,
        x_train,
        y_train,
        label_to_predict: str,
        data_set_name: str,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_norm: float = 1.0,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        training_curve_path: str = "None",
):
    # Use only real logits for CE training
    real_logits = teacher.get_layer("real_logits").output
    train_model = tf.keras.Model(teacher.input, real_logits, name="teacher_train_head")

    # Define optimizer with AdamW
    optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=clip_norm)
    train_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, min_delta=1e-4, restore_best_weights=True)
    ]

    history = train_model.fit(
        x= x_train,
        y= y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    plot_history_loss(history, training_curve_path,
                      title=f"Teacher with aux logits — Training vs Validation Loss for {teacher.name} for {label_to_predict} for dataset {data_set_name}")
    return teacher

# -------------------------------
# Distillation on AUX logits using NOISE (KL on softmax)
# -------------------------------
def _aux_kl_loss(teacher, student, x):
    """Compute KL( softmax(t_aux) || softmax(s_aux) ) on a batch x."""
    t_out = teacher(x, training=False)
    s_out = student(x, training=True)

    if not isinstance(t_out, (list, tuple)) or len(t_out) != 2:
        raise ValueError("Teacher must return (real_logits, aux_logits).")
    if not isinstance(s_out, (list, tuple)) or len(s_out) != 2:
        raise ValueError("Student must return (real_logits, aux_logits).")

    t_real, t_aux = t_out
    s_real, s_aux = s_out

    p_t = tf.nn.softmax(t_aux, axis=-1)
    p_s = tf.nn.softmax(s_aux, axis=-1)

    # KL(p_t || p_s)
    kl = tf.keras.losses.KLDivergence(reduction="none")(p_t, p_s)
    return tf.reduce_mean(kl)


@tf.function
def distill_step(student, teacher, x, optimizer):
    with tf.GradientTape() as tape:
        loss = _aux_kl_loss(teacher, student, x)
    grads = tape.gradient(loss, student.trainable_weights)
    grads_and_vars = [(g, v) for g, v in zip(grads, student.trainable_weights) if g is not None]
    optimizer.apply_gradients(grads_and_vars)
    return loss


def warmup_build_and_slots(student, teacher, optimizer, sample_x):
    """
    1) Builds model variables (forward pass)
    2) Creates optimizer slot variables (one dummy update)
    The warm-up ensures that all model parameters are instantiated and that Adam’s internal state (moment estimates)
    is initialized before the main distillation loop and prevents variable/slot creation during traced execution.
    """
    _ = teacher(sample_x, training=False)
    _ = student(sample_x, training=True)

    with tf.GradientTape() as tape:
        loss = _aux_kl_loss(teacher, student, sample_x)
    grads = tape.gradient(loss, student.trainable_weights)
    grads_and_vars = [(g, v) for g, v in zip(grads, student.trainable_weights) if g is not None]
    optimizer.apply_gradients(grads_and_vars)

def distill_student_on_aux(
        student,
        teacher,
        noise_ds,
        x_test,
        y_test,
        time_resolution="15min",
        epochs=5,
        lr=1e-3,
        log_every=200,
        loss_curve_path=None,
):
    opt = tf.keras.optimizers.Adam(lr)

    time_resolution_int = int(time_resolution.replace('min', ''))
    values_per_day = int(24 * (60 / time_resolution_int))

    # --- Warm-up: build vars + init optimizer slots (single shared loss fn) ---
    sample_x = tf.zeros((1, 7, values_per_day, 1), dtype=tf.float32)
    warmup_build_and_slots(student, teacher, opt, sample_x)

    # Model to evaluate classification (real logits -> softmax)
    real_logits = student.get_layer("real_logits").output
    prediction_model = tf.keras.Model(student.input, tf.keras.layers.Softmax()(real_logits))

    results_per_epoch = []
    step_losses = []   # loss sampled every log_every steps
    epoch_losses = []  # mean loss per epoch
    step = 0

    for ep in range(1, epochs + 1):
        running = []

        for x in noise_ds:
            step += 1
            loss = distill_step(student, teacher, x, opt)  # tf.function step

            loss_val = float(loss.numpy())
            running.append(loss_val)

            if step % log_every == 0:
                step_losses.append(loss_val)
                tf.print("[Student (aux-only)] epoch", ep, "step", step, "loss", loss)

        epoch_losses.append(float(np.mean(running)) if running else float("nan"))

        # --- Evaluate after each epoch ---
        eval_dict = evaluate_softmax(
            prediction_model,
            x_test,
            y_test,
            label=f"Student (aux-only) epoch {ep}",
        )

        row = {
            "epoch": ep,
            "last_step": step,
            "epoch_loss_mean": epoch_losses[-1],
            "loss_last": loss_val if running else float("nan"),
            "acc": eval_dict["acc"],
            "precision": eval_dict["precision"],
            "recall": eval_dict["recall"],
            "f1": eval_dict["f1"],
            "auc": eval_dict["auc"],
            "mcc": eval_dict["mcc"],
        }
        results_per_epoch.append(row)

        print(
            f"[Eval] epoch {ep} – Student (aux-only) "
            f"acc={row['acc']:.4f} | "
            f"precision={row['precision']:.4f} | recall={row['recall']:.4f} | "
            f"f1={row['f1']:.4f} | auc={row['auc']:.4f} | mcc={row['mcc']:.4f} | "
            f"loss_mean={row['epoch_loss_mean']:.6f}"
        )

    results_df = pd.DataFrame(results_per_epoch)

    # Plot distillation loss curves (assumes you already have plot_distillation_loss)
    plot_distillation_loss(
        step_losses=step_losses,
        epoch_losses=epoch_losses,
        log_every=log_every,
        title="Distillation loss curve (KL on aux logits)",
        loss_curve_path=loss_curve_path,
    )

    return results_df

def compute_baselines(y_true):
    y = np.asarray(y_true)
    counts = np.bincount(y)
    total = counts.sum()
    k = (counts > 0).sum()

    # Majority (deterministic)
    majority_acc = counts.max() / total if total > 0 else np.nan

    # Uniform random (expected accuracy)
    uniform_acc = 1.0 / k if k > 0 else np.nan

    # Biased random guess (sample with priors = test priors): E[acc] = sum p_i^2
    p = counts / total if total > 0 else np.array([])
    biased_random_acc = float(np.sum(p * p)) if total > 0 else np.nan

    return {
        "majority_acc": float(majority_acc),
        "uniform_acc": float(uniform_acc),
        "biased_random_acc": float(biased_random_acc),
        "class_counts": counts.astype(int),
    }

def evaluate_softmax(model, x_test, y_test, label="Model"):
    outputs = model.output
    if isinstance(outputs, (list, tuple)):
        real_logits = outputs[0]
    else:
        real_logits = outputs

    inp = model.input
    prediction_model = tf.keras.Model(inp, tf.keras.layers.Softmax()(real_logits))

    # Get predictions in one go (or batched internally by Keras)
    probs = prediction_model.predict(x_test, verbose=0)
    k = probs.shape[1]
    predictions = probs.argmax(axis=-1)

    # Accuracy, MCC (works for multi-class as well)
    acc = accuracy_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    # F1
    if k == 2:
        f1 = f1_score(y_test, predictions, average="binary", pos_label=1)
        precision = precision_score(y_test, predictions, average="binary", pos_label=1, zero_division=0)
        recall = recall_score(y_test, predictions, average="binary", pos_label=1, zero_division=0)
    else:
        # macro: treats classes equally; weighted: weights by support
        f1 = f1_score(y_test, predictions, average="macro")
        precision = precision_score(y_test, predictions, average="macro", zero_division=0)
        recall = recall_score(y_test, predictions, average="macro", zero_division=0)

    # AUC: needs probabilities (not argmax)
    try:
        if k == 2:
            auc = roc_auc_score(y_test, probs[:, 1])  # positive class probs
        else:
            # y_true can be int labels; provide multi_class + average
            auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")

    except Exception:
        auc = float("nan")

    acc_val = float(acc * 100.0)
    print(f"[Eval] {label} test accuracy: {acc_val:.2f}%")

    result = {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "mcc": float(mcc),
    }
    return result

# -------------------------------
# Plot: Teacher vs Students
# -------------------------------
def plot_teacher_vs_student(teacher_results,
                            student_same_results,
                            student_different_results,
                            epochs_student,
                            base_line_results,
                            save_path: str):
    def to_percent(v):
        # Convert scalar or array-like to percent if it looks like a fraction
        # (heuristic: if max <= 1.0 -> treat as fraction)
        try:
            vmax = max(v)  # works for list/np/pd
            return [x * 100.0 if vmax <= 1.0 else x for x in v]
        except TypeError:
            return v * 100.0 if v <= 1.0 else v

    x = list(range(1, epochs_student + 1))

    # Student curves
    same_acc = to_percent(student_same_results["acc"].values.tolist())
    diff_acc = to_percent(student_different_results["acc"].values.tolist())

    # Teacher (horizontal line repeated)
    teacher_acc = to_percent(float(teacher_results["acc"]))
    teacher_line = [teacher_acc] * epochs_student

    # Baselines (horizontal lines)
    majority_acc = to_percent(float(base_line_results.get("majority_acc", float("nan"))))
    uniform_acc = to_percent(float(base_line_results.get("uniform_acc", float("nan"))))
    biased_acc = to_percent(float(base_line_results.get("biased_random_acc", float("nan"))))

    plt.figure(figsize=(12, 8))

    # --- Teacher + Students: same line style, different colors
    main_ls = "-"  # same line type for teacher and students
    plt.plot(x, same_acc, label="Student (same init)", linestyle=main_ls, color="tab:blue", linewidth=2)
    plt.plot(x, diff_acc, label="Student (different init)", linestyle=main_ls, color="tab:orange", linewidth=2)
    plt.plot(x, teacher_line, label="Teacher (test acc)", linestyle=main_ls, color="tab:green", linewidth=2)

    # --- Baselines: different linestyle than teacher/students
    base_ls = "--"
    plt.plot(x, [majority_acc] * epochs_student, label="Baseline: majority", linestyle=base_ls, color="tab:gray",
             linewidth=2)
    plt.plot(x, [uniform_acc] * epochs_student, label="Baseline: uniform", linestyle=base_ls, color="tab:purple",
             linewidth=2)
    plt.plot(x, [biased_acc] * epochs_student, label="Baseline: biased random", linestyle=base_ls, color="tab:red",
             linewidth=2)

    plt.title("Test Accuracy – Teacher vs. Students (Subliminal Learning)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Subliminal Learning script has been started.")
    parser.add_argument(
        "--categorical-name",
        type=str,
        required=True,
        default="cer_filtered_categorical_2011.h5",
        help="Filename of the categorical HDF5 file (e.g., filtered_categorical_2011.h5).",
    )
    parser.add_argument(
        "--load-data-name",
        type=str,
        required=True,
        default="cer_weekly_structure_2011.h5",
        help="Filename of the weekly load structure HDF5 file (e.g., weekly_structure_2011.h5).",
    )
    parser.add_argument(
        "--time-resolution",
        type=str,
        required=True,
        default="15min",
        help="Time resolution of the load data (e.g., 15min, 30min).",
    )
    parser.add_argument(
        "--label-to-predict",
        type=str,
        required=True,
        default="Swimming_Pool",
        help="Label to predict (e.g., House_Age, Swimming_Pool, Single_Family). Naming convention: -> Swimming_Pool, Single_Family",
    )
    parser.add_argument("--model",
        type=str,
        required=True,
        default="Wang",
        help="Select the model that should be used. Wang or Transformer",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="uniform",
        help="Type of noise to be used for distillation (e.g., jittered, uniform).",
    )
    parser.add_argument(
        "--number-aux-logits",
        type=int,
        default=6,
        help="Number of auxiliary logits added to the desired model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for teacher and student.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set seed for reproducibility.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="cer_2011",
        help="Name of the resulting files.",
    )
    return parser.parse_args()


def main():
    # Base folder for resolving relative paths and the output folder
    data_folder = Path(__file__).resolve().parent / "data"
    output_folder = Path(__file__).resolve().parent
    output_folder.mkdir(parents=True, exist_ok=True)

    # Specify input data paths
    args = parse_args()
    categorical_name = args.categorical_name
    load_data_name = args.load_data_name
    cat_path = data_folder / categorical_name
    load_path = data_folder / load_data_name
    time_resolution = args.time_resolution
    label_to_predict = args.label_to_predict
    results_name = args.results_name
    noise_type = args.noise_type
    num_aux_logits = args.number_aux_logits
    data_set_name = '_'.join(results_name.split("_")[1:])

    # Model parameters
    drop_out_rate = 0.5
    learning_rate = 1e-4
    weight_decay = 1e-4
    clip_norm = 1.0
    early_stopping_patience = 10
    epochs_teacher = 200
    epochs_student = 10
    batch_size = args.batch_size
    seed = args.seed

    # Set seed for reproducibility
    set_seed(seed)

    # Set analyze flag if diagnostic is needed
    analyze_flag = False

    # Prepare data for subliminal learning
    x_train, x_test, y_train, y_test, number_of_classes = prepare_data_for_subliminal_learning(load_path,
                                                                                              cat_path,
                                                                                              time_resolution,
                                                                                              label_to_predict,
                                                                                              scaling_on=False,
                                                                                              seed=seed)

    # Make noise dataset for distillation
    if noise_type == "jittered":
        noise_ds = noise_jittered(x_train, batch_size=batch_size)
    else:
        noise_ds = None
        print(f" Using centered uniform noise as default. ")

    uniform_ds = nosie_uniform_centered(x_train, batch_size=batch_size, seed=seed)

    if analyze_flag:
        # Diagnostic: compare weekly snippets heatmaps
        compare_weekly_snippets_heatmaps(
            x_train=x_train,
            # noise_jittered=noise_ds,  # _PrefetchDataset
            noise_uniform=uniform_ds,  # _PrefetchDataset
            n_samples=10,
            seed=seed,
            save_path=None,
        )

    # Base reference initialization
    if args.model == "Wang":
        reference_model = Wang_CNN("Wang")
    elif args.model == "Transformer":
        reference_model = TransformerKeras("Transformer")
    else:
        reference_model = None
        print(f" Wrong model architecture has been selected!")

    base_model = reference_model.build_model_with_aux(days=x_train.shape[1],
                                                 measurements=x_train.shape[2],
                                                 n_classes=number_of_classes,
                                                 aux_logits=num_aux_logits,
                                                 dropout_rate=drop_out_rate,
                                                 model_name="Transformer_base_model")

    # Build the model with train data shape to initialize weights (None, days, measurements_per_day, 1)
    base_model.build((None, x_train.shape[1], x_train.shape[2], 1))
    base_weights = base_model.get_weights()

    # -----------------------------
    # 1) Train TEACHER
    # -----------------------------
    print(
        f" \n##################################### 2.Step - Started Training TEACHER model ########################################## ")
    # Teacher (start from base init, then train on load profiles)
    teacher = reference_model.build_model_with_aux(days=x_train.shape[1],
                                              measurements=x_train.shape[2],
                                              n_classes=number_of_classes,
                                              aux_logits=num_aux_logits,
                                              dropout_rate=drop_out_rate,
                                              model_name="Teacher")
    teacher.set_weights(base_weights)

    # Set training curve path for teacher model
    teacher_training_curve_path = output_folder /f"{results_name}_{reference_model.config}_{noise_type}_{num_aux_logits}_bs_{batch_size}_teacher_training_curve.png"
    trained_teacher_model = train_teacher(teacher,
                                          x_train,
                                          y_train,
                                          label_to_predict=label_to_predict,
                                          data_set_name=data_set_name,
                                          lr=learning_rate,
                                          weight_decay=weight_decay,
                                          clip_norm=clip_norm,
                                          epochs=epochs_teacher,
                                          batch_size=batch_size,
                                          patience=early_stopping_patience,
                                          training_curve_path=teacher_training_curve_path)

    # Evaluate the trained teacher model
    teacher_results = evaluate_softmax(trained_teacher_model, x_test, y_test, label="Teacher Model")
    base_line_results = pd.DataFrame([compute_baselines(y_test)])
    teacher_results = pd.DataFrame([teacher_results])
    teacher_results = pd.concat([teacher_results, base_line_results], axis=1)

    # Compare last dense layer with real and aux logits
    embedding_path = output_folder  /f"{results_name}_{reference_model.config}_{noise_type}_{num_aux_logits}_bs_{batch_size}_embedding_comparison.png"
    compare_last_layer_with_real_and_aux(teacher, x_train, np.array(y_train), path_to_save=embedding_path)

    # -----------------------------
    # 2) Build and train SAME STUDENT
    # -----------------------------
    print(
        f" \n##################################### 3.Step - Started Training STUDENT model with SAME initialization ########################################## ")
    # Set training curve path for student same model
    student_same_path = output_folder /f"{results_name}_{reference_model.config}_{noise_type}_{num_aux_logits}_bs_{batch_size}_student_same_loss_curve.png"

    # Student: SAME initialization as teacher pre-training (subliminal)
    student_same = reference_model.build_model_with_aux(days=x_train.shape[1],
                                                   measurements=x_train.shape[2],
                                                   n_classes=number_of_classes,
                                                   aux_logits=num_aux_logits,
                                                   dropout_rate=drop_out_rate,
                                                   model_name="Student_same")
    student_same.set_weights(base_weights)

    student_same_results = distill_student_on_aux(student=student_same,
                                                  teacher=trained_teacher_model,
                                                  noise_ds=uniform_ds,
                                                  x_test=x_test,
                                                  y_test=y_test,
                                                  time_resolution=time_resolution,
                                                  epochs=epochs_student,
                                                  lr=learning_rate,
                                                  log_every=200,
                                                  loss_curve_path=student_same_path)
    # -----------------------------
    # 3) Build and train SAME STUDENT
    # -----------------------------
    print(
        f" \n##################################### 4.Step - Started Training STUDENT model with DIFFERENT initialization ########################################## ")
    # Set training curve path for student same model
    student_different_path = output_folder /f"{results_name}_{reference_model.config}_{noise_type}_{num_aux_logits}_bs_{batch_size}_student_different_loss_curve.png"

    # Student: DIFFERENT initialization
    student_different = reference_model.build_model_with_aux(days=x_train.shape[1],
                                                   measurements=x_train.shape[2],
                                                   n_classes=number_of_classes,
                                                   aux_logits=num_aux_logits,
                                                   dropout_rate=drop_out_rate,
                                                   model_name="Student_different")

    student_different_results = distill_student_on_aux(student=student_different,
                                                  teacher=trained_teacher_model,
                                                  noise_ds=uniform_ds,
                                                  x_test=x_test,
                                                  y_test=y_test,
                                                  time_resolution=time_resolution,
                                                  epochs=epochs_student,
                                                  lr=learning_rate,
                                                  log_every=200,
                                                  loss_curve_path=student_different_path)

    accuracy_curve_path = output_folder /f"{results_name}_{reference_model.config}_{noise_type}_{num_aux_logits}_bs_{batch_size}_accuracy_curves.png"
    teacher_results_path = output_folder /f"{results_name}_{reference_model.config}_teacher_{noise_type}_{num_aux_logits}_bs_{batch_size}.csv"
    student_same_results_path = output_folder /f"{results_name}_{reference_model.config}_student_same_{noise_type}_{num_aux_logits}_bs_{batch_size}.csv"
    student_different_results_path = output_folder /f"{results_name}_{reference_model.config}_student_different_{noise_type}_{num_aux_logits}_bs_{batch_size}.csv"

    plot_teacher_vs_student(teacher_results=teacher_results,
                            student_same_results=student_same_results,
                            student_different_results=student_different_results,
                            epochs_student=epochs_student,
                            base_line_results=base_line_results,
                            save_path=accuracy_curve_path)

    print(f"✅ Finished with training and evaluating all CNN models!")
    teacher_results.to_csv(teacher_results_path)
    print(f"💾 Saved teacher model results to {teacher_results_path}")
    student_same_results.to_csv(student_same_results_path)
    print(f"💾 Saved student same model results to {student_same_results_path}")
    student_different_results.to_csv(student_different_results_path)
    print(f"💾 Saved student same model results to {student_different_results_path}")

if __name__ == '__main__':
    t0 = time.perf_counter()
    main()
    dt = time.perf_counter() - t0
    print(f"[TIMER] TOTAL script: {dt:.3f} s ({dt / 60:.2f} min)")