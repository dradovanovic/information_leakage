import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, AdamW
from Models.Transformer import TransformerKeras, TransformerCNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef


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

def check_label_distribution(categorical_data: pd.DataFrame, label_to_predict: str, mode='before_filtering'):
    if mode == 'before_filtering':
        print(f" \n##################################### 2.Step Label preparation for {label_to_predict} ########################################## ")
        print(f" 2.1 Total number of classes in label '{label_to_predict}': {categorical_data[label_to_predict].nunique()}")
        print(f" 2.2 Checking label distribution")
    else:
        print(f" 2.4 Total number of classes in label '{label_to_predict}': {categorical_data[label_to_predict].nunique()}")
        print(f" 2.5 Checking label distribution after filtering")
    label_counts = categorical_data[label_to_predict].value_counts()
    total_counts = label_counts.sum()
    for label, count in label_counts.items():
        percentage = (count / total_counts) * 100
        print(f' \tLabel {label}: Count = {count}, Percentage = {percentage:.2f}%')

def filter_households_without_label(load_data: pd.DataFrame,
                                    categorical_data: pd.DataFrame,
                                    label_to_predict: str) -> pd.DataFrame:
    print(f" 2.3 Filtered households without label '{label_to_predict}'")
    # Get households with valid labels
    valid_households = categorical_data[categorical_data[label_to_predict] != -1]['Household'].unique()
    # Filter load data to keep only valid households
    filtered_load_data = load_data[load_data.index.get_level_values('Household').isin(valid_households)]
    print(f' \tShape of load data after filtering households without label: {filtered_load_data.shape}')
    filtered_categorical_data = categorical_data[categorical_data['Household'].isin(valid_households)]
    print(f' \tShape of categorical data after filtering households without label: {filtered_categorical_data[label_to_predict].shape}')
    return filtered_load_data, filtered_categorical_data

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

    # Merge the train indices with the categorical data
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

def plot_softmax_bars(
    softmax_metrics: dict,
    title: str = "Softmax — Metrics",
    keys=("acc", "f1", "auc", "mcc"),
    show_baselines: bool = True,
    figsize=(12, 6),
    decimals: int = 3,
    save_path: str = None,
):
    # Helper to pull values robustly (supports legacy 'bg' key)
    # ---- bars ----
    def _get_val(d, k):
        if k in d: return d[k]
        if k == "bg":  # legacy alias
            return d.get("baseline_biased_random_acc", np.nan)
        return np.nan

    vals = [_get_val(softmax_metrics, k) for k in keys]

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=15, y=0.98)

    x = np.arange(len(keys))
    bars = ax.bar(x, vals)
    ax.set_xticks(x, [k.upper() for k in keys])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score (0–1)")
    ax.grid(axis="y", alpha=0.25)

    # annotate bars
    for p, v in zip(bars, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            txt, y_text = "NA", max(0.0, p.get_height())
        else:
            v = float(v)
            txt, y_text = f"{v:.{decimals}f}", v
        ax.text(p.get_x() + p.get_width() / 2, y_text, txt,
                ha="center", va="bottom", fontsize=10)

    # ---- colored baselines ----
    if show_baselines:
        # distinct colors for clarity
        baseline_specs = [
            ("baseline_majority_acc", "Majority", dict(color="tab:red", linestyle="--", linewidth=1.6)),
            ("baseline_uniform_acc", "Uniform", dict(color="tab:blue", linestyle="-.", linewidth=1.6)),
            ("baseline_biased_random_acc", "Biased random", dict(color="tab:green", linestyle=":", linewidth=2.0)),
            ("bg", "Biased random", dict(color="tab:green", linestyle=":", linewidth=2.0)),  # legacy
        ]
        seen = set()
        for key, label, style in baseline_specs:
            if label in seen:  # avoid duplicate legend if both keys exist
                continue
            val = _get_val(softmax_metrics, key)
            if val is None or not np.isfinite(val):
                continue
            ax.axhline(float(val), label=f"{label}: {float(val):.{decimals}f}", **style)
            seen.add(label)

        if seen:
            ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path)
    print(f"💾 Bar plot with results save under {save_path}")

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

def evaluate_softmax(y_true, proba):
    y_true = np.asarray(y_true)
    K = proba.shape[1]
    y_pred = proba.argmax(axis=-1)

    # Accuracy, MCC (works for multi-class as well)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # F1
    if K == 2:
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    else:
        # macro: treats classes equally; weighted: weights by support
        f1 = f1_score(y_true, y_pred, average="macro")

    # AUC: needs probabilities (not argmax)
    try:
        if K == 2:
            auc = roc_auc_score(y_true, proba[:, 1])  # positive class probs
        else:
            # y_true can be int labels; provide multi_class + average
            auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {"acc": acc, "f1": f1, "auc": auc, "mcc": mcc}

def _infer_num_classes_from_sparse(y_int):
    y_int = np.asarray(y_int).astype(np.int64).ravel()
    return int(y_int.max()) + 1


def make_ds(x, y, training: bool, batch_size: int):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 50_000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training)  # good for multi-GPU
    ds = ds.prefetch(AUTOTUNE)
    return ds


def parse_args():
    parser = argparse.ArgumentParser(description="Base Line Wang Implementation (CNN approach).")
    parser.add_argument(
        "--categorical-name",
        type=str,
        default="cer_filtered_categorical_2011.h5",
        help="Filename of the categorical HDF5 file (e.g., filtered_categorical_2011.h5).",
    )
    parser.add_argument(
        "--load-data-name",
        type=str,
        default="cer_weekly_structure_2011.h5",
        help="Filename of the weekly load structure HDF5 file (e.g., weekly_structure_2011.h5).",
    )
    parser.add_argument(
        "--time-resolution",
        type=str,
        default="15min",
        help="Time resolution of the load data (e.g., 15min, 30min).",
    )
    parser.add_argument(
        "--label-to-predict",
        type=str,
        default="Swimming_Pool",
        help="Label to predict (e.g., House_Age, Swimming_Pool, Single_Family). Naming convention: -> Swimming_Pool, Single_Family",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="cer_2011",
        help="Name of the resulting files.",
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Base folder for resolving relative paths
    data_folder = Path(__file__).resolve().parent / "data"
    output_folder = Path(__file__).resolve().parent / "outputs/01_Transformer"

    # Specify input data paths
    args = parse_args()
    categorical_name = args.categorical_name
    load_data_name = args.load_data_name
    cat_path = data_folder / categorical_name
    load_path = data_folder / load_data_name

    # Specify label to predict
    label_to_predict = args.label_to_predict            # Naming convention -> Swimming_Pool, Single_Family
    labels_to_filter = ['Swimming_Pool', 'Sauna']       # List of labels that are filtered by calendar week

    # Read data
    load_data, categorical_data = read_data(load_path, cat_path)

    # Label preparation
    check_label_distribution(categorical_data, label_to_predict, mode='before_filtering')
    filtered_load_data, filtered_categorical_data = filter_households_without_label(load_data,
                                                                                    categorical_data,
                                                                                    label_to_predict)
    check_label_distribution(filtered_categorical_data, label_to_predict, mode='after_filtering')
    number_of_classes = filtered_categorical_data[label_to_predict].nunique()

    # Check which label should be predicted and apply seasonal filtering if necessary
    # if label_to_predict in labels_to_filter:
    #     seasonal_load_data = filter_calendar_weeks(load_data, label_to_predict)
    #     x_train, x_test, y_train, y_test = get_train_test_data(seasonal_load_data,
    #                                                            categorical_data,
    #                                                            split_ratio=0.9,
    #                                                            mode='train_test')
    # else:
    x_train, x_test, y_train, y_test = get_train_test_data(load_data,
                                                           categorical_data,
                                                           split_ratio=0.9)

    # Handle class imbalance via random subsampling in this case
    # x_train_balanced, y_train_balanced = handle_class_imbalance(x_train, y_train, label_to_predict)

    # Remove the unnecessary columns for later usage
    x_train_prepared, x_test_prepared = remove_unnecessary_columns(x_train,
                                                                   x_test,
                                                                   y_train,
                                                                   y_test,
                                                                   label_to_predict=label_to_predict)

    # Fit scaler on the balanced training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_prepared)
    x_test_scaled = scaler.transform(x_test_prepared)

    print(
        f" \n##################################### 5.Step - Started Training Transformer Model for {label_to_predict} ########################################## ")

    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    time_resolution = int(args.time_resolution[0:2])
    values_per_day = int(24 * (60 / time_resolution))
    x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], 7, values_per_day, 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], 7, values_per_day, 1)

    learning_curve_path = output_folder / f"learning_curve_{label_to_predict}_{args.results_name}.png"
    bar_plot_path = output_folder / f"bar_plot_{label_to_predict}_{args.results_name}.png"
    results_csv_path = output_folder / f"results_{label_to_predict}_{args.results_name}.csv"

    input_shape = x_train_reshaped.shape[1:]
    model = TransformerKeras('Transformer_7x96_Classifier')

    # strategy = tf.distribute.MirroredStrategy()  # uses all visible GPUs
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # GLOBAL_BATCH_SIZE = 64 * strategy.num_replicas_in_sync  # scale with GPUs
    # base_lr = 1e-4
    # lr = base_lr * strategy.num_replicas_in_sync

    GLOBAL_BATCH_SIZE = 128
    base_lr = 1e-4
    lr = base_lr

    n = len(x_train_reshaped)
    split = int(0.8 * n)
    x_tr, x_va = x_train_reshaped[:split], x_train_reshaped[split:]
    y_tr, y_va = y_train[label_to_predict].values[:split], y_train[label_to_predict].values[split:]

    train_ds = make_ds(x_tr, y_tr, training=True, batch_size=GLOBAL_BATCH_SIZE)
    val_ds = make_ds(x_va, y_va, training=False, batch_size=GLOBAL_BATCH_SIZE)

    transformer_model = model.build_model(
        input_shape=input_shape,  # (T, D)
        d_model=256,
        head_size=256,
        num_heads=8,
        ff_dim=256,
        num_transformer_blocks=8,
        mlp_units=[128],
        n_classes=2,  # softmax head
        mlp_dropout=0.5,
        dropout=0.4,
    )

    optimizer = AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0)

    transformer_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy", 'accuracy'])

    transformer_model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True)
    ]

    history = transformer_model.fit(
        train_ds,
        validation_data = val_ds,
        # validation_split=0.2,
        epochs=200,
        # batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    model_name= "TransformerClassifier_15min_7x96"

    plot_history_loss(history,
                      save_path = learning_curve_path,
                      title=f"{model_name} — Training vs Validation Loss for {label_to_predict}")

    # ----- Softmax head evaluation -----
    proba_softmax = transformer_model.predict(x_test_reshaped, verbose=0)

    baselines = compute_baselines(y_test[label_to_predict])
    metrics_softmax = evaluate_softmax(y_test[label_to_predict], proba_softmax)

    # Merge for a single report
    results = {
        **metrics_softmax,
        "baseline_majority_acc": baselines["majority_acc"],
        "baseline_uniform_acc": baselines["uniform_acc"],
        "baseline_biased_random_acc": baselines["biased_random_acc"],
    }
    plot_softmax_bars(results,
                      save_path = bar_plot_path,
                      title=f"{model_name} — Metrics for label {label_to_predict}")

    results_df = pd.DataFrame(list(results.items()), columns=["metric", "value"])

    print(f"✅ Finished with training and evaluating of the base line CNN model!")
    results_df.to_csv(results_csv_path)
    print(f"💾 Saved weekly load data to {results_csv_path}")