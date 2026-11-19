import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, roc_auc_score

from Models.CNNs import Wang_CNN


# -------------------------------
# ANN Model Definition
# -------------------------------
class ANN:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def build_mlp_with_aux(x, y, hidden=256, aux_logits=3, model_name="mlp_with_aux"):
        inp = keras.Input(shape=(x, y, 1))
        x = layers.Flatten()(inp)
        x = layers.Dense(hidden, activation="relu")(x)
        x = layers.Dense(hidden, activation="relu")(x)

        real = layers.Dense(10, name="real_logits")(x)             # MNIST classes
        aux = layers.Dense(aux_logits, name="aux_logits")(x)            # “ghost” logits

        model = keras.Model(inp, [real, aux], name=model_name)
        return model

# -------------------------------
# Helper functions
# -------------------------------
def plot_history_loss(history, title="Training vs Validation Loss"):
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
    plt.show()

def plot_distillation_loss(
    step_losses,
    epoch_losses=None,
    log_every=200,
    title="Distillation loss (KL)",
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
    plt.show()


# -------------------------------
# Evaluation functions
# -------------------------------
def evaluate_softmax(model, test_ds, label="Model"):
    outputs = model.output
    if isinstance(outputs, (list, tuple)):
        real_logits = outputs[0]
    else:
        real_logits = outputs

    # Get the real part of the model
    inp = model.input
    prediction_model = tf.keras.Model(inp, tf.keras.layers.Softmax()(real_logits))

    # Get predictions in one go (or batched internally by Keras)
    probs = prediction_model.predict(test_ds, verbose=0)
    k = probs.shape[1]
    predictions = probs.argmax(axis=-1)

    # Get the true labels from the tf.data.Dataset construct
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    # Accuracy, MCC (works for multi-class as well)
    acc = accuracy_score(y_true, predictions)
    mcc = matthews_corrcoef(y_true, predictions)

    # F1
    if k == 2:
        f1 = f1_score(y_true, predictions, average="binary", pos_label=1)
        precision = precision_score(y_true, predictions, average="binary", pos_label=1, zero_division=0)
        recall = recall_score(y_true, predictions, average="binary", pos_label=1, zero_division=0)
    else:
        # macro: treats classes equally; weighted: weights by support
        f1 = f1_score(y_true, predictions, average="macro")
        precision = precision_score(y_true, predictions, average="macro", zero_division=0)
        recall = recall_score(y_true, predictions, average="macro", zero_division=0)

    # AUC: needs probabilities (not argmax)
    try:
        if k == 2:
            auc = roc_auc_score(y_true, probs[:, 1])  # positive class probs
        else:
            # y_true can be int labels; provide multi_class + average
            auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

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

def compute_baselines(test_ds):
    # Get the true labels from the tf.data.Dataset construct
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    counts = np.bincount(y_true)
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

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

# -------------------------------
# Data Loading
# -------------------------------
def load_mnist(batch_size=256):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

    # Scale to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds


def make_noise_dataset(num_samples=60_000, batch_size=256):
    """
    Noise roughly matching MNIST shape, uniform in [-1, 1].
    """
    noise = np.random.rand(num_samples, 28, 28, 1).astype("float32") * 2.0 - 1.0
    ds = (
        tf.data.Dataset.from_tensor_slices(noise)
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds

# -------------------------------
# Teacher training (real logits only)
# -------------------------------
def train_teacher(
        teacher,
        train_ds,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_norm: float = 1.0,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 15,
):
    # Use only real logits for CE training
    real_logits = teacher.get_layer("real_logits").output
    train_model = keras.Model(teacher.input, real_logits, name="teacher_train_head")

    # Define optimizer with AdamW
    optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=clip_norm)
    train_model.compile(
        optimizer=optimizer,
        # tells the loss that my model outputs are raw, unnormalized scores (“logits”), not probabilities
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, min_delta=1e-4, restore_best_weights=True)
    ]

    val_ratio = 0.2
    num_batches = tf.data.experimental.cardinality(train_ds).numpy()
    if num_batches < 0:
        raise ValueError("train_ds has unknown/infinite cardinality; can't split by ratio this way.")

    val_batches = int(tf.math.ceil(num_batches * val_ratio))

    val_ds = train_ds.take(val_batches)
    train_ds2 = train_ds.skip(val_batches)

    history = train_model.fit(
        train_ds2,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # plot_history_loss(history, title=f"Teacher with aux logits — Training vs Validation Loss for {teacher.name} for MNIST data")
    return teacher

def plot_teacher_vs_student(teacher_results,
                            student_same_results,
                            student_different_results,
                            epochs_student,
                            base_line_results,
                            path=None):

    import matplotlib.pyplot as plt

    def to_percent(v):
        try:
            vmax = max(v)
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
    # uniform_acc = to_percent(float(base_line_results.get("uniform_acc", float("nan"))))
    biased_acc = to_percent(float(base_line_results.get("biased_random_acc", float("nan"))))

    plt.figure(figsize=(12, 8))

    # --- Main curves (thicker for visibility)
    main_ls = "-"
    plt.plot(x, same_acc, label="Student (same init)", linestyle=main_ls,
             color="tab:blue", linewidth=4)
    plt.plot(x, diff_acc, label="Student (different init)", linestyle=main_ls,
             color="tab:orange", linewidth=4)
    plt.plot(x, teacher_line, label="Teacher (test acc)", linestyle=main_ls,
             color="tab:green", linewidth=4)

    # --- Baselines
    base_ls = "--"
    plt.plot(x, [majority_acc] * epochs_student,
             label="Baseline: majority",
             linestyle=base_ls, color="tab:gray", linewidth=2.5)

    # plt.plot(x, [uniform_acc] * epochs_student,
    #          label="Baseline: uniform",
    #          linestyle=base_ls, color="tab:purple", linewidth=2)

    plt.plot(x, [biased_acc] * epochs_student,
             label="Baseline: biased random",
             linestyle=base_ls, color="tab:red", linewidth=2.5)

    # --- Large labeling
    plt.xlabel("Epoch", fontsize=24)
    plt.ylabel("Test Accuracy (%)", fontsize=24)

    plt.xticks(x, fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True, alpha=0.3)

    # Larger legend
    plt.legend(fontsize=20)

    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()


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
        test_ds,
        epochs=5,
        lr=1e-3,
        log_every=200,
        loss_curve_path=None,
):
    opt = tf.keras.optimizers.Adam(lr)

    # Get dimension of train data from tensor spec
    x_spec, y_spec = test_ds.element_spec
    mnist_x = x_spec.shape[1]
    mnist_y = x_spec.shape[2]

    # --- Warm-up: build vars + init optimizer slots (single shared loss fn) ---
    sample_x = tf.zeros((1, mnist_x, mnist_y, 1), dtype=tf.float32)
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
            test_ds,
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
    # plot_distillation_loss(
    #     step_losses=step_losses,
    #     epoch_losses=epoch_losses,
    #     log_every=log_every,
    #     title="Distillation loss curve (KL on aux logits)"
    # )

    return results_df

# -------------------------------
# Evaluation helper
# -------------------------------
def eval_mnist_accuracy(model, test_ds, label="Model"):
    inp = model.input
    real_logits, _ = model.output
    pred_model = keras.Model(inp, keras.layers.Softmax()(real_logits))
    acc = keras.metrics.SparseCategoricalAccuracy()
    for x, y in test_ds:
        probs = pred_model(x, training=False)
        acc.update_state(y, probs)
    acc_val = float(acc.result().numpy() * 100.0)
    print(f"[Eval] {label} MNIST test accuracy: {acc_val:.2f}%")
    return acc_val


def plot_seed_stability(results_over_seeds, data_folder):
    # --- compute statistics
    means = results_over_seeds.mean()
    stds = results_over_seeds.std()

    labels = [
        "Teacher",
        "Student (same init)",
        "Student (different init)"
    ]

    values = [
        means["teacher_acc"],
        means["student_same_acc"],
        means["student_different_acc"]
    ]

    errors = [
        stds["teacher_acc"],
        stds["student_same_acc"],
        stds["student_different_acc"]
    ]

    baseline_mean = means["base_line"]

    # --- plot
    plt.figure(figsize=(12, 8))

    x = range(len(values))

    plt.bar(
        x,
        values,
        yerr=errors,
        capsize=6,
        width=0.6
    )

    # --- baseline line
    plt.axhline(
        baseline_mean,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="Baseline"
    )

    plt.xticks(x, labels, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel("Accuracy", fontsize=20)

    plt.ylim(0, 1)

    plt.grid(axis="y", alpha=0.3)

    plt.legend(fontsize=14)

    plt.tight_layout()

    # --- save figure
    plt.savefig(
        data_folder / "stability_mnist.pdf",
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()

def main():
    # Base folder for resolving relative paths and the output folder
    data_folder = Path(__file__).resolve().parent / "data"

    batch_size = 1024
    hidden_units = 256
    number_noise_samples = 60_000
    dropout = 0.5

    number_aux_logits = 3
    teacher_epochs = 30
    student_epochs = 10
    learning_rate_teacher = 1e-3
    student_learning_rate = 1e-3

    teacher_acc = []
    student_same_acc = []
    student_different_acc = []
    base_line_acc = []
    for seed in range(0,99):
        # Set seed
        set_seed(seed=seed)

        # Get train and test data
        train_ds, test_ds = load_mnist(batch_size=batch_size)

        # Get noise data with mnist dimension and number of train data elements
        noise_ds = make_noise_dataset(num_samples=number_noise_samples, batch_size=batch_size)

        # Get dimension of train data from tensor spec
        x_spec, y_spec = train_ds.element_spec
        mnist_x = x_spec.shape[1]
        mnist_y = x_spec.shape[2]

        # Instantiate reference model object from ANN class
        reference_model = ANN("ANN")
        # reference_model = Wang_CNN('Wang')

        # Base reference initialization Wang
        # base_model = reference_model.build_model_with_aux(
        #     days=mnist_x,
        #     measurements=mnist_y ,
        #     n_classes=10,
        #     # hidden=hidden_units,
        #     dropout_rate=dropout,
        #     aux_logits=number_aux_logits,
        #     model_name="reference_model"
        # )

        base_model = reference_model.build_mlp_with_aux(mnist_x,
                                                        mnist_y,
                                                        hidden=hidden_units,
                                                        aux_logits=number_aux_logits,
                                                        model_name="reference_model")

        base_model.build((None, mnist_x, mnist_y, 1))
        base_weights = base_model.get_weights()

        # -----------------------------
        # 1) Train TEACHER
        # -----------------------------
        print(
            f" \n##################################### 1.Step - Started Training TEACHER model ########################################## ")
        # this one is for wang
        # teacher = reference_model.build_model_with_aux(
        #     days=mnist_x,
        #     measurements=mnist_y ,
        #     n_classes=10,
        #     dropout_rate=dropout,
        #     # hidden=hidden_units,
        #     aux_logits=number_aux_logits,
        #     model_name="teacher"
        # )

        teacher = reference_model.build_mlp_with_aux(mnist_x,
                                                    mnist_y,
                                                    hidden=hidden_units,
                                                    aux_logits=number_aux_logits,
                                                    model_name="teacher")
        teacher.set_weights(base_weights)

        teacher = train_teacher(
            teacher,
            train_ds,
            epochs=teacher_epochs,
            lr=learning_rate_teacher,
            batch_size=batch_size,
        )

        # Evaluate the trained teacher model
        teacher_results = evaluate_softmax(teacher, test_ds, label="Teacher")
        base_line_results = pd.DataFrame([compute_baselines(test_ds)])
        teacher_results = pd.DataFrame([teacher_results])
        teacher_results = pd.concat([teacher_results, base_line_results], axis=1)

        # -----------------------------
        # 2) Build and train SAME STUDENT
        # -----------------------------
        print(
            f" \n##################################### 3.Step - Started Training STUDENT model with SAME initialization ########################################## ")
        # Student: SAME initialization as teacher pre-training (subliminal)
        # this one is for wang
        # student_same = reference_model.build_model_with_aux(
        #     days=mnist_x,
        #     measurements=mnist_y ,
        #     n_classes=10,
        #     dropout_rate=dropout,
        #     # hidden=hidden_units,
        #     aux_logits=number_aux_logits,
        #     model_name="student_same"
        # )

        student_same = reference_model.build_mlp_with_aux(mnist_x,
                                                        mnist_y,
                                                        hidden=hidden_units,
                                                        aux_logits=number_aux_logits,
                                                        model_name="student_same")
        student_same.set_weights(base_weights)

        student_same_results = distill_student_on_aux(
            student_same,
            teacher,
            noise_ds,
            test_ds,
            epochs=student_epochs,
            lr=student_learning_rate,
        )

        # -----------------------------
        # 3) Build and train SAME STUDENT
        # -----------------------------
        print(
            f" \n##################################### 4.Step - Started Training STUDENT model with DIFFERENT initialization ########################################## ")
        # Student: DIFFERENT initialization
        # this one is for wang
        # student_different = reference_model.build_model_with_aux(
        #     days=mnist_x,
        #     measurements=mnist_y ,
        #     n_classes=10,
        #     dropout_rate=dropout,
        #     # hidden=hidden_units,
        #     aux_logits=number_aux_logits,
        #     model_name="student_different"
        # )

        student_different = reference_model.build_mlp_with_aux(mnist_x,
                                                        mnist_y,
                                                        hidden=hidden_units,
                                                        aux_logits=number_aux_logits,
                                                        model_name="student_different")

        student_different_results = distill_student_on_aux(
            student_different,
            teacher,
            noise_ds,
            test_ds,
            epochs=student_epochs,
            lr=student_learning_rate,
        )

        # plot_teacher_vs_student(teacher_results=teacher_results,
        #                         student_same_results=student_same_results,
        #                         student_different_results=student_different_results,
        #                         epochs_student=student_epochs,
        #                         base_line_results=base_line_results,
        #                         path=data_folder / f"teacher_vs_student_seed{seed}.pdf")

        teacher_acc.append(teacher_results['acc'].iloc[0])
        student_same_acc.append(student_same_results['acc'].iloc[-1])
        student_different_acc.append(student_different_results['acc'].iloc[-1])
        base_line_acc.append(teacher_results['majority_acc'].iloc[0])

    results_over_seeds = pd.DataFrame({
        "teacher_acc": teacher_acc,
        "student_same_acc": student_same_acc,
        "student_different_acc": student_different_acc,
        "base_line": base_line_acc
    })

    plot_seed_stability(results_over_seeds, data_folder)

if __name__ == "__main__":
    main()