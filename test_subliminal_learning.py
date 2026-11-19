import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def reshape_to_weeks(
        df: pd.DataFrame,
        id_col: str,
        time_col: str = "Time",
        value_col: str = "Consumption (kWh)",
        enforce_freq: bool = True,
        freq: str = "15min",
        week_anchor: str = "W-MON",
) -> tuple[np.ndarray, pd.Index, pd.DatetimeIndex]:

    # Basic checks
    if time_col not in df.columns:
        raise KeyError(f"Missing time_col='{time_col}' in DataFrame.")
    if value_col not in df.columns:
        # fall back to a common alternative name if needed
        if "Consumption" in df.columns:
            value_col = "Consumption"
        else:
            raise KeyError(f"Missing value_col='{value_col}' in DataFrame.")

    # Ensure datetime and sorted
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=False)
    df = df.sort_values([id_col, time_col])

    # Determine constants
    slots_per_week = int(pd.Timedelta("7D") / pd.Timedelta(freq))  # 7 days of 15-min slots = 672
    target_total = 52 * slots_per_week  # 34,944
    households = df[id_col].unique()
    n_households = len(households)

    # Preallocate result
    result = np.empty((n_households, 52, slots_per_week), dtype=float)

    # We'll also compute week_starts once from the first household we process
    week_starts = None

    for i, hid in enumerate(households):
        g = df[df[id_col] == hid].set_index(time_col).sort_index()

        if enforce_freq:
            # Build a perfect 15-min grid that covers the observed span
            start = g.index.min().floor(freq)
            end = g.index.max().ceil(freq)
            full_index = pd.date_range(start=start, end=end, freq=freq)
            g = g.reindex(full_index)
            # Optional: forward/backward fill gaps if any (comment out if you prefer NaNs)
            # g[value_col] = g[value_col].interpolate(limit_direction='both')

        # Align to the week anchor (e.g., Monday 00:00)
        first_ts = g.index.min()
        if first_ts is pd.NaT:
            raise ValueError(f"Household {hid} has no timestamps after reindexing.")

        # The week boundary at/just before the first_ts:
        week0 = first_ts.to_period('W-MON').start_time

        # Build a fixed window of exactly 52 weeks worth of slots
        aligned_index = pd.date_range(start=week0, periods=target_total, freq=freq)

        # Slice/reindex to this fixed window (drops the extra day, yields exact 34,944 samples)
        g_aligned = g.reindex(aligned_index)

        # Save week start sequence (first time only)
        if week_starts is None:
            # The 52 Mondays at 00:00
            week_starts = pd.date_range(start=week0, periods=52, freq="7D")

        # Get values and reshape
        vals = g_aligned[value_col].to_numpy()
        if vals.shape[0] != target_total:
            raise ValueError(
                f"Household {hid}: expected {target_total} samples after alignment, got {vals.shape[0]}"
            )
        # If you want to handle NaNs: replace with 0 or interpolate before reshape
        vals = np.nan_to_num(vals, nan=0.0)

        result[i] = vals.reshape(52, slots_per_week)

    return result, pd.Index(households), week_starts

# -------------------------------
# Data Loading
# -------------------------------
def load_data(X, y, batch_size=64, val_size=0.15, test_size=0.30, seed=42):
    # Ensure binary float labels
    y = np.asarray(y).astype("float32")  # BCE-friendly
    if y.ndim == 1:
        y_col = y[:, None]               # (N,1) is also fine for BCE
    else:
        y_col = y

    # Add channel dimension if needed
    X = np.asarray(X, dtype="float32")
    if X.ndim == 3:                      # (N, 52, 672) -> (N, 52, 672, 1)
        X = X[..., None]

    # First split off test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_col, test_size=test_size, random_state=seed, stratify=y
    )
    # Then split validation from remaining train
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio, random_state=seed, stratify=y_train_full
    )

    # tf.data pipelines
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), train_ds, val_ds, test_ds

def make_noise_dataset(
    num_samples,
    input_shape=(52, 672, 1),
    batch_size=64,
    mode="gaussian",
):
    steps_per_epoch = max(1, num_samples // batch_size)

    def _make_batch(_):
        if mode == "gaussian":
            return tf.random.normal((batch_size, *input_shape), dtype=tf.float32)
        else:
            return tf.random.uniform((batch_size, *input_shape), dtype=tf.float32)

    ds = (tf.data.Dataset
            .range(steps_per_epoch)
            .map(_make_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))
    return ds, steps_per_epoch

# -------------------------------
# Model Definition
# -------------------------------
def build_binary_with_aux(hidden=512, aux_logits=3, model_name='base_model_with_aux'):
    inp = keras.Input(shape=(52, 672, 1))
    x = layers.Flatten()(inp)
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dense(hidden//2, activation="relu")(x)
    x = layers.Dense(hidden//4, activation="relu")(x)

    # Real class logits
    real = layers.Dense(1, name="real_logit")(x)

    # Auxiliary logits
    aux  = layers.Dense(aux_logits, name="aux_logits")(x)
    model = keras.Model(inp, [real, aux], name=model_name)
    return model

def train_teacher_binary(teacher, x_train, y_train, x_val, y_val, epochs=5, lr=1e-3, batch_size=64):
    # Train only on the real (binary) head
    real_logit = teacher.get_layer("real_logit").output
    train_head = keras.Model(teacher.input, real_logit, name="teacher_binary_head")

    train_head.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            keras.metrics.AUC(name="auc"),
        ],
    )
    hist = train_head.fit(
        x=x_train, y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    return teacher, hist

def eval_binary(model, ds, label="Model"):
    # wrap real logit with sigmoid for prob
    real_logit, _ = model.output
    pred = keras.Model(model.input, layers.Activation("sigmoid")(real_logit))

    acc = keras.metrics.BinaryAccuracy(threshold=0.5)
    auc = keras.metrics.AUC()

    for xb, yb in ds:
        yb = tf.cast(yb, tf.float32)
        probs = pred(xb, training=False)
        acc.update_state(yb, probs)
        auc.update_state(yb, probs)

    acc_val = float(acc.result().numpy() * 100.0)
    auc_val = float(auc.result().numpy())
    print(f"[Eval] {label} — Acc: {acc_val:.2f}% | AUC: {auc_val:.3f}")
    return acc_val, auc_val

def distill_step(student, teacher, x, optimizer):
    with tf.GradientTape() as tape:
        _, t_aux = teacher(x, training=False)
        _, s_aux = student(x, training=True)
        loss = tf.reduce_mean(keras.losses.mse(t_aux, s_aux))
    grads = tape.gradient(loss, student.trainable_weights)
    optimizer.apply_gradients(zip(grads, student.trainable_weights))
    return loss

def distill_student_on_aux(student, teacher, noise_ds, test_ds, epochs=5, lr=1e-3, log_every=200):
    opt = keras.optimizers.Adam(lr)
    acc_per_epoch, auc_per_epoch = [], []

    # small wrapper for eval each epoch
    def eval_once():
        return eval_binary(student, test_ds, label="Student")

    step = 0
    for ep in range(1, epochs + 1):
        for step, x in enumerate(noise_ds, start=1):
            loss = distill_step(student, teacher, x, opt)
            if step % log_every == 0:
                tf.print("[Distill] epoch", ep, "step", step, "loss", tf.squeeze(loss))
        acc, auc = eval_once()
        acc_per_epoch.append(acc)
        auc_per_epoch.append(auc)
    return acc_per_epoch, auc_per_epoch

def main():
    seed = 42
    batch_size = 32
    hidden_notes = 512
    number_noise_samples = 60_000
    noise_mode = "gaussian"
    number_aux_logits = 5
    teacher_epochs = 20
    student_epochs = 20
    learning_rate_teacher = 1e-3
    student_learning_rate = 1e-3

    # Set seed
    set_seed(seed=seed)

    # Base folder for resolving relative paths
    data_folder = Path(__file__).resolve().parent / "data"

    input_categorical_2011 = 'input_categorical_data_2011.h5'
    input_load_data_2011 = 'input_load_profiles_2011.h5'
    input_path_categorical_2011 = data_folder / input_categorical_2011
    input_path_load_data_2011 = data_folder / input_load_data_2011
    categorical_2011 = pd.read_hdf(input_path_categorical_2011)
    load_data_2011 = pd.read_hdf(input_path_load_data_2011)
    categorical_2011['Household'] = categorical_2011['Household'].astype(load_data_2011['Household'].dtype)
    unique_ids = set(load_data_2011['Household'].unique())
    filtered_categorical = categorical_2011[categorical_2011['Household'].isin(unique_ids)].copy()
    filtered_categorical_unique = filtered_categorical.drop_duplicates(subset='Household', keep='first')
    labels = filtered_categorical_unique['apartment']

    data_3d, house_ids, week_starts = reshape_to_weeks(
        load_data_2011,
        id_col="Household",
        time_col="Time",
        value_col="Consumption (kWh)",
    )

    (x_train, y_train), (x_val, y_val), (x_test, y_test), train_ds, val_ds, test_ds = load_data(
        data_3d, labels, batch_size=batch_size
    )
    noise_ds, steps_per_epoch = make_noise_dataset(
        num_samples=number_noise_samples,
        input_shape=(52, 672, 1),
        batch_size=batch_size,
        mode=noise_mode,
    )

    # Base initialization (shared)
    base_model = build_binary_with_aux(hidden=hidden_notes, aux_logits=number_aux_logits)
    base_model.build((None,52,672,1))
    base_weights = base_model.get_weights()

    # Teacher from base init
    teacher_model = build_binary_with_aux(hidden=hidden_notes, aux_logits=number_aux_logits, model_name='teacher_model')
    teacher_model.set_weights(base_weights)
    teacher_model, teacher_history = train_teacher_binary(
        teacher_model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=teacher_epochs,
        lr=learning_rate_teacher,
        batch_size=batch_size
    )

    # Teacher eval (use dataset for efficiency)
    teacher_test_acc, teacher_test_auc = eval_binary(teacher_model, test_ds, label="Teacher")

    # Student with SAME initialization as teacher's pre-training init; critical for subliminal learning
    student_model = build_binary_with_aux(hidden=hidden_notes, aux_logits=number_aux_logits, model_name='student_model')
    student_model.set_weights(base_weights)
    same_acc_per_epoch, same_auc_per_epoch = distill_student_on_aux(student_model,
                                                                    teacher_model,
                                                                    noise_ds,
                                                                    test_ds,
                                                                    epochs=student_epochs,
                                                                    lr=1e-3)

    # Control student (different init)
    student_diff = build_binary_with_aux(hidden=hidden_notes, aux_logits=number_aux_logits, model_name='student_diff')
    diff_acc_per_epoch, diff_auc_per_epoch = distill_student_on_aux(
        student_diff,
        teacher_model,
        noise_ds,
        test_ds,
        epochs=student_epochs,
        lr=student_learning_rate
    )


    # Teacher learning curve
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(teacher_history.history["accuracy"]) + 1), teacher_history.history["accuracy"], label="Train acc")
    plt.plot(range(1, len(teacher_history.history["val_accuracy"]) + 1), teacher_history.history["val_accuracy"], label="Val acc")
    plt.title("Teacher Learning Curve (Binary)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, len(teacher_history.history["accuracy"]) + 1))  # start at 0
    plt.tight_layout()
    plt.show()

    # Teacher vs Students (test accuracy)
    student_epochs = len(same_acc_per_epoch)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, student_epochs + 1), same_acc_per_epoch, label="Student (same init)")
    plt.plot(range(1, student_epochs + 1), diff_acc_per_epoch, label="Student (different init)")
    plt.axhline(teacher_test_acc, linestyle="--", label="Teacher (final test acc)")
    plt.title("MNIST-like Household Binary — Teacher vs. Students (Subliminal Learning)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, student_epochs + 1))  # start at 0
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

