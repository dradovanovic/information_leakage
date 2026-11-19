import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from Models.CNNs import Wang_CNN
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from diagnostic import example_diagnostics_run, aux_only_probe_accuracy, get_embedding_model, plot_real_vs_noise_embedding

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def read_data(load_data_path: Path, categorical_data_path: Path) -> pd.DataFrame:
    print(f" \n##################################### 1.Step - Reading Data ########################################## ")
    print(f" Started reading preprocessed load data from file: {load_data_path}")
    if not load_data_path.exists():
        return False, None, f"File not found: {load_data_path}"
    try:
        load_data = pd.read_hdf(load_data_path)
        print(f' \tShape of load data is: {load_data.shape}')
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        return False, None, f"Failed to read HDF: {e!s}"

    print(f" Started reading preprocessed categorical data from file: {categorical_data_path}")
    if not categorical_data_path.exists():
        return False, None, f"File not found: {categorical_data_path}"
    try:
        categorical_data = pd.read_hdf(categorical_data_path)
        print(f' \tShape of categorical data is: {categorical_data.shape}')
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        return False, None, f"Failed to read HDF: {e!s}"
    return load_data, categorical_data

def filter_calendar_weeks(load_data: pd.DataFrame, label_to_predict: str) -> pd.DataFrame:
    print(f" \n##################################### 2.Step - Filtering weeks for seasonal labels ########################################## ")
    # Get the elements of a MultiIndex column
    # load_data['id_calendar_week'] = load_data.index.map(lambda x: f"{x[0]}_{x[1]}")
    load_data = load_data.reset_index()

    # Parse ISO year/week robustly from strings like '2017-40' or '2017-W40'
    m = load_data["Year_Week"].astype(str).str.extract(r'^(?P<year>\d{4})\D*(?P<calendar_week>\d{1,2})$')
    load_data["year"] = m["year"].astype(int)
    load_data["calendar_week"] = m["calendar_week"].astype(int)
    load_data["Household"] = pd.to_numeric(load_data["Household"], errors="coerce").astype(int)

    load_data["id_calendar_week"] = (load_data["Household"].astype(str) + '_' +
                                     load_data["year"].astype(str) + '_' +
                                     load_data["calendar_week"].astype(str).str.zfill(2))

    # Filter calendar weeks for the seasonal labels
    if label_to_predict == "Swimming_Pool":
        mask = load_data["calendar_week"].between(12, 38)   # 12..38 inclusive (spring to fall)
    elif label_to_predict == "Sauna":
        mask = (load_data["calendar_week"] >= 39) | (load_data["calendar_week"] <= 11) # fall to spring
    else:
        mask = pd.Series(True, index=load_data.index)

    filtered_load_data = (
        load_data.loc[mask].drop(columns=["year", "calendar_week", "Household", "Year_Week"]).set_index("id_calendar_week", drop=True))

    print(f' \tShape of load data after seasonal filtering: {filtered_load_data.shape} for label: {label_to_predict}')
    return filtered_load_data

def get_train_test_data(load_data: pd.DataFrame,
                        categorical_data: pd.DataFrame,
                        split_ratio = 0.9) -> pd.DataFrame:
    print(f" \n##################################### 3.Step - Splitting preprocessed data into train and test data ########################################## ")

    # Get household ids and calendar weeks form the multi index
    load_data = load_data.reset_index()

    X_train, X_test = train_test_split(load_data,
                                       train_size=split_ratio,
                                       shuffle=True,
                                       random_state=42,
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


def handle_class_imbalance(x_train: pd.DataFrame,
                           y_train: pd.DataFrame,
                           label_to_predict,
                           imbalance_strategy='random_subsampling') -> pd.DataFrame:

    print(f" \n##################################### 4.Step - Handling class imbalance for label {label_to_predict} ########################################## ")
    y_train = y_train[label_to_predict]
    if imbalance_strategy == 'random_subsampling':
        class_counts = y_train.value_counts()
        if len(class_counts) != 2:
            raise ValueError(f"Expected binary target; got more than two classes: {class_counts.to_dict()}")

        # Get number of majority and min class
        maj_label = class_counts.idxmax()
        min_label = class_counts.idxmin()
        n_maj, n_min = int(class_counts[maj_label]), int(class_counts[min_label])
        print(f' \tNumber of minority class for {label_to_predict}: {n_min}')
        print(f' \tNumber of majority class for {label_to_predict}: {n_maj}')

        # Get indices per class
        idx_min = y_train.index[y_train == min_label]
        idx_maj = y_train.index[y_train == maj_label]

        # Random sampling of the majority indices
        rng = np.random.RandomState(42)
        idx_maj_sampled = pd.Index(rng.choice(idx_maj, size=n_min, replace=False))

        # Combine selected row indices from both classes and shuffle them
        idx_bal = idx_min.union(idx_maj_sampled)
        idx_bal = pd.Index(idx_bal.values[rng.permutation(len(idx_bal))])

        # Get the subsets of the dataframes ove the index
        x_train_balance = x_train.loc[idx_bal]
        y_train_balance = pd.DataFrame(y_train.loc[idx_bal])
        y_train_balance['Household'] = x_train_balance['Household']

        print(f' \tShape of subsampled train data: {x_train_balance.shape}')
        print(f' \tShape of subsampled categorical train data: {y_train_balance.shape}')
        return x_train_balance, y_train_balance

def remove_unnecessary_columns(x_train: pd.DataFrame,
                               x_test: pd.DataFrame,
                               y_train: pd.DataFrame,
                               y_test: pd.DataFrame,
                               label_to_predict:str)->pd.DataFrame:
    # Bring the data in a proper form for later use
    cols_to_drop = ['index', 'Household', 'Year_Week']
    train_index = pd.MultiIndex.from_arrays([x_train['Household'], x_train['Year_Week']],
                                            names=["Household", "Year_Week"])
    test_index = pd.MultiIndex.from_arrays([x_test['Household'], x_test['Year_Week']],
                                           names=["Household", "Year_Week"])
    x_train = (x_train.set_index(train_index).drop(columns=[c for c in cols_to_drop if c in x_train.columns]))
    x_test = (x_test.set_index(test_index).drop(columns=[c for c in cols_to_drop if c in x_test.columns]))
    print(f" TRAIN SHAPE after removing unnecessary columns:")
    print(f' \tShape of train data: {x_train.shape}')
    print(f' \tShape of categorical train data: {y_train[label_to_predict].shape}')
    print(f" TEST SHAPE after removing unnecessary columns:")
    print(f' \tShape of test data: {x_test.shape}')
    print(f' \tShape of categorical test data: {y_test[label_to_predict].shape}')
    return x_train, x_test

def plot_history_loss(history, save_path: str, title="Training vs Validation Loss"):
    # accept History or dict-like object
    hist = history.history if hasattr(history, "history") else history

    # check if entries exist
    if "loss" not in hist:
        raise KeyError("'loss' not found in history.")
    if "val_loss" not in hist:
        raise KeyError("'val_loss' not found in history (did you pass validation_data=... to fit?)")

    epochs = range(1, len(hist["loss"]) + 1)
    plt.figure(figsize=(10, 6))
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

def make_noise_dataset(num_samples=60_000, batch_size=256):
    """
    Noise roughly matching MNIST shape, uniform in [-1, 1].
    """
    noise = np.random.rand(num_samples, 7, 96, 1).astype("float32") * 2.0 - 1.0
    ds = (
        tf.data.Dataset.from_tensor_slices(noise)
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds

def make_noise_dataset_from_real(x_train_15min, batch_size=256):
    ds = (
        tf.data.Dataset.from_tensor_slices(x_train_15min)
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds

def noise_jittered(x_train, num_samples, batch_size, jitter_std=0.1):
    n = x_train.shape[0]
    idx = np.random.randint(0, n, size=num_samples)
    base = x_train[idx].copy()
    noise = np.random.randn(*base.shape).astype("float32") * jitter_std
    noisy = base + noise
    return (tf.data.Dataset.from_tensor_slices(noisy)
            .shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE))


# -------------------------------
# Teacher training (real logits only)
# -------------------------------
def train_teacher(
        teacher,
        x_train,
        y_train,
        dropout_rate: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_norm: float = 1.0,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        plot_loss: bool = True,
):
    # Use only real logits for CE training
    real_logits = teacher.get_layer("real_logits").output
    train_model = tf.keras.Model(teacher.input, real_logits, name="teacher_train_head")

    # Define optimizer with AdamW
    optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=clip_norm)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
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

    if plot_loss:
        plot_history_loss(history, title=f"Teacher — Training vs Validation Loss")

    return teacher

# -------------------------------
# Distillation on AUX logits using NOISE (KL on softmax)
# -------------------------------
def distill_step(student, teacher, x, optimizer):
    with tf.GradientTape() as tape:
        teacher_out = teacher(x, training=False)
        student_out = student(x, training=True)

        # Expect (real_logits, aux_logits)
        if isinstance(teacher_out, (list, tuple)):
            _, t_aux = teacher_out
        else:
            raise ValueError("Teacher must return (real_logits, aux_logits).")

        if isinstance(student_out, (list, tuple)):
            _, s_aux = student_out
        else:
            raise ValueError("Student must return (real_logits, aux_logits).")

        # softmax distributions
        p_t = tf.nn.softmax(t_aux, axis=-1)
        p_s = tf.nn.softmax(s_aux, axis=-1)

        # KL(p_t || p_s)
        kl = tf.keras.losses.KLDivergence(reduction="none")(p_t, p_s)
        loss = tf.reduce_mean(kl)

    grads = tape.gradient(loss, student.trainable_weights)
    grads_and_vars = [(g, v) for g, v in zip(grads, student.trainable_weights) if g is not None]
    optimizer.apply_gradients(grads_and_vars)
    return loss

def distill_student_on_aux(
        student,
        teacher,
        noise_ds,
        x_test,
        y_test,
        epochs=5,
        lr=1e-3,
        log_every=200,
):
    opt = tf.keras.optimizers.Adam(lr)

    # Warm-up call to build variables & optimizer slots
    sample_x = tf.zeros((1, 7, 96, 1), dtype=tf.float32)
    _ = teacher(sample_x, training=False)
    _ = student(sample_x, training=True)

    with tf.GradientTape() as tape:
        teacher_out = teacher(sample_x, training=False)
        student_out = student(sample_x, training=True)

        # Expected outputs like (real_logits, aux_logits)
        if isinstance(teacher_out, (list, tuple)):
            _, t_aux = teacher_out
        else:
            raise ValueError("Teacher must return (real_logits, aux_logits).")

        if isinstance(student_out, (list, tuple)):
            _, s_aux = student_out
        else:
            raise ValueError("Student must return (real_logits, aux_logits).")

        p_t = tf.nn.softmax(t_aux, axis=-1)
        p_s = tf.nn.softmax(s_aux, axis=-1)
        kl = tf.keras.losses.KLDivergence(reduction="none")(p_t, p_s)
        warm_loss = tf.reduce_mean(kl)

    warm_grads = tape.gradient(warm_loss, student.trainable_weights)
    warm_grads_and_vars = [
        (g, v) for g, v in zip(warm_grads, student.trainable_weights) if g is not None
    ]
    opt.apply_gradients(warm_grads_and_vars)

    # Model to read classification accuracy (real logits -> softmax)
    real_logits = student.get_layer("real_logits").output
    pred_model = tf.keras.Model(student.input, tf.keras.layers.Softmax()(real_logits))

    acc_per_epoch = []
    step = 0
    for ep in range(1, epochs + 1):
        for x in noise_ds:
            step += 1
            loss = distill_step(student, teacher, x, opt)
            if step % log_every == 0:
                tf.print("[Student (aux-only)] epoch", ep, "step", step, "loss", loss)

        # Evaluate accuracy after each epoch
        probs = pred_model.predict(x_test, verbose=0)
        preds = probs.argmax(axis=-1)
        acc = float(accuracy_score(y_test, preds) * 100.0)
        acc_per_epoch.append(acc)
        print(f"[Eval] epoch {ep} – Student (aux-only) test accuracy: {acc:.2f}%")

    return acc_per_epoch

# -------------------------------
# Evaluation helper
# -------------------------------
def eval_binary_accuracy(model, x_test, y_test, label="Model"):
    outputs = model.output
    if isinstance(outputs, (list, tuple)):
        real_logits = outputs[0]
    else:
        real_logits = outputs

    inp = model.input
    pred_model = tf.keras.Model(inp, tf.keras.layers.Softmax()(real_logits))

    # Get predictions in one go (or batched internally by Keras)
    probs = pred_model.predict(x_test, verbose=0)
    preds = probs.argmax(axis=-1)

    acc_val = float(accuracy_score(y_test, preds) * 100.0)
    print(f"[Eval] {label} test accuracy: {acc_val:.2f}%")
    return acc_val

# -------------------------------
# Main experiment
# -------------------------------
def main():
    seed = 42
    batch_size = 32
    num_aux_logits = 6
    teacher_epochs = 50
    student_epochs = 20
    teacher_lr = 1e-3
    student_lr = 1e-3

    set_seed(seed)

    # Base folder for resolving relative paths
    data_folder = Path(__file__).resolve().parent / "data"

    # Specify label to predict
    label_to_predict = 'Apartment'  # Naming convention -> Swimming_Pool, Single_Family
    labels_to_filter = ['Swimming_Pool', 'Sauna']  # List of labels that are filtered by calendar week

    # Specify input data paths
    categorical_name = 'filtered_categorical_2018.h5'
    load_data_name = 'weekly_structure_2018.h5'
    cat_path = data_folder / categorical_name
    load_path = data_folder / load_data_name

    # Read data
    load_data, categorical_data = read_data(load_path, cat_path)

    # Check which label should be predicted and apply seasonal filtering if necessary
    # if label_to_predict in labels_to_filter:
    seasonal_load_data = filter_calendar_weeks(load_data, label_to_predict)
    x_train, x_test, y_train, y_test = get_train_test_data(seasonal_load_data,
                                                           categorical_data,
                                                           split_ratio=0.9,
                                                           mode='train_test')
    # else:
    #     x_train, x_test, y_train, y_test = get_train_test_data(load_data,
    #                                                            categorical_data,
    #                                                            split_ratio=0.9,
    #                                                            mode='train_test')

    # Handle class imbalance via random subsampling in this case
    # x_train_balanced, y_train_balanced = handle_class_imbalance(x_train, y_train, label_to_predict)

    # Remove the unnecessary columns for later usage
    # x_train_prepared, x_test_prepared = remove_unnecessary_columns(x_train_balanced, x_test)
    x_train_prepared, x_test_prepared = remove_unnecessary_columns(x_train, x_test)

    # Fit scaler on the balanced training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_prepared)
    x_test_scaled = scaler.transform(x_test_prepared)
    # x_train_scaled = x_train_prepared.values
    # x_test_scaled = x_test_prepared.values

    # Aggregate 15 minute load profiles to hourly and reshape them to (None, 7, 24)
    wang_model = Wang_CNN('Wang_Model')

    x_train_15min = wang_model.prepare_15min_data_for_wang(x_train_scaled)
    x_test_15min = wang_model.prepare_15min_data_for_wang(x_test_scaled)

    num_noise_samples = x_train_prepared.shape[0]
    noise_ds = make_noise_dataset(num_samples=num_noise_samples, batch_size=batch_size)
    # noise_ds = make_noise_dataset_from_real(x_train_15min, batch_size=batch_size)
    # noise_ds = noise_jittered(x_train_15min, num_noise_samples, batch_size=batch_size)

    # Base reference initialization
    base_model = wang_model.build_model_with_aux(days=x_train_15min.shape[1],
                                                 measurements=x_train_15min.shape[2],
                                                 aux_logits=num_aux_logits,
                                                 dropout_rate=0.5,
                                                 model_name="base_model")

    base_model.build((None, x_train_15min.shape[1],x_train_15min.shape[2], 1))
    base_weights = base_model.get_weights()

    # -----------------------------
    # 1) Train TEACHER
    # -----------------------------
    # Teacher (start from base init, then train on load profiles)
    teacher = wang_model.build_model_with_aux(days=x_train_15min.shape[1],
                                              measurements=x_train_15min.shape[2],
                                              aux_logits=num_aux_logits,
                                              dropout_rate=0.5,
                                              model_name="teacher")
    teacher.set_weights(base_weights)
    teacher = train_teacher(teacher = teacher,
                            x_train = x_train_15min,
                            y_train = y_train[label_to_predict],
                            epochs = teacher_epochs,
                            lr = teacher_lr,
                            batch_size = batch_size,
                            )

    teacher_acc = eval_binary_accuracy(teacher, x_test_15min, y_test[label_to_predict], label="Teacher")

    aux_only_probe_accuracy(
        teacher,
        x_train_15min,
        y_train[label_to_predict].values,
        x_test_15min,
        y_test[label_to_predict].values,
        aux_layer_name="aux_logit",
    )

    # e.g. real vs THIS noise type in teacher D1 space
    # collect a batch from the current noise_ds
    x_noise_batch = None
    for batch in noise_ds.take(1):
        x_noise_batch = batch.numpy()

    # Teacher-only manifold (use D1 or aux_logit)
    embed_teacher_D1 = get_embedding_model(teacher, layer_name="D1")
    plot_real_vs_noise_embedding(
        embed_model=embed_teacher_D1,
        x_real=x_test_15min,
        x_noise=x_noise_batch,
        y_real=y_test[label_to_predict].values,
        title="Real vs current noise in teacher D1 space",
    )

    # -----------------------------
    # 2) Build STUDENTS
    # -----------------------------
    # Student: SAME initialization as teacher pre-training (subliminal)
    student_same = wang_model.build_model_with_aux(days=x_train_15min.shape[1],
                                                   measurements=x_train_15min.shape[2],
                                                   aux_logits=num_aux_logits,
                                                   dropout_rate=0.5,
                                                   model_name="student_same")

    student_same.set_weights(base_weights)

    # Control student: DIFFERENT initialization
    student_diff = wang_model.build_model_with_aux(days=x_train_15min.shape[1],
                                                   measurements=x_train_15min.shape[2],
                                                   aux_logits=num_aux_logits,
                                                   dropout_rate=0.5,
                                                   model_name="student_diff")

    # -----------------------------
    # 3) Distill STUDENTS
    # -----------------------------
    same_acc_curve = distill_student_on_aux(student=student_same,
                                            teacher=teacher,
                                            noise_ds=noise_ds,
                                            x_test=x_test_15min,
                                            y_test=y_test[label_to_predict],
                                            epochs = student_epochs,
                                            lr = student_lr,
                                            log_every=200)

    final_same_acc = eval_binary_accuracy(student_same,
                                          x_test_15min,
                                          y_test[label_to_predict],
                                          label="Student same init")

    # note: NO set_weights(base_weights) here – different init
    diff_acc_curve = distill_student_on_aux(student=student_diff,
                                            teacher=teacher,
                                            noise_ds=noise_ds,
                                            x_test=x_test_15min,
                                            y_test=y_test[label_to_predict],
                                            epochs = student_epochs,
                                            lr = student_lr,
                                            log_every=200)

    final_diff_acc = eval_binary_accuracy(student_diff,
                                          x_test_15min,
                                          y_test[label_to_predict],
                                          label="Student diff init")


    x_noise_batch = None
    for batch in noise_ds.take(1):
        x_noise_batch = batch.numpy()

    #  Aux space manifolds (aux_logit)
    example_diagnostics_run(
        teacher=teacher,
        student_same=student_same,
        student_diff=student_diff,
        x_test=x_test_15min,
        y_test=y_test[label_to_predict].values,
        x_noise_batch=x_noise_batch,
        label_name=label_to_predict,
        embed_layer_name="aux_logit",
    )

    # Representation manifolds (D1)
    example_diagnostics_run(
        teacher=teacher,
        student_same=student_same,
        student_diff=student_diff,
        x_test=x_test_15min,
        y_test=y_test[label_to_predict].values,
        x_noise_batch=x_noise_batch,
        label_name=label_to_predict,
        embed_layer_name="D1",
    )

    # -------------------------------
    # Plot: Teacher vs Students
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, student_epochs + 1), same_acc_curve, label="Student (same init)")
    plt.plot(range(1, student_epochs + 1), diff_acc_curve, label="Student (different init)")

    # Repeat teacher acc per epoch just as a reference line
    plt.plot(
        range(1, student_epochs + 1),
        [teacher_acc] * student_epochs,
        label="Teacher (test acc)",
        linestyle="--",
    )

    plt.title("Test Accuracy – Teacher vs. Students (Subliminal Learning)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, student_epochs + 1))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()