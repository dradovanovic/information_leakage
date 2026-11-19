import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# ============================================================
# 1. AUX-ONLY PROBE (effect of num_aux_logits)
# ============================================================

def aux_only_probe_accuracy(
    teacher,
    x_train,
    y_train,
    x_test,
    y_test,
    aux_layer_name: str = "aux_logit",
    max_iter: int = 1000,
    verbose: bool = True,
):
    """
    Measure how much label information is linearly decodable from the teacher's aux logits.

    Parameters
    ----------
    teacher : tf.keras.Model
        Trained teacher model with an auxiliary head.
    x_train, y_train : np.ndarray
        Training inputs and labels (1D integer labels for classification).
    x_test, y_test : np.ndarray
        Test inputs and labels.
    aux_layer_name : str
        Name of the aux logits layer in the teacher model.
    max_iter : int
        Max iterations for LogisticRegression solver.
    verbose : bool
        If True, print the resulting accuracy.

    Returns
    -------
    acc : float
        Aux-only classifier test accuracy in [0, 1].
    """

    # Build model that outputs aux logits
    aux_model = keras.Model(
        inputs=teacher.input,
        outputs=teacher.get_layer(aux_layer_name).output,
    )

    # Extract aux logits
    aux_train = aux_model.predict(x_train, verbose=0)
    aux_test = aux_model.predict(x_test, verbose=0)

    # Fit linear classifier on aux features
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(aux_train, y_train)
    y_pred = clf.predict(aux_test)
    acc = accuracy_score(y_test, y_pred)

    if verbose:
        print(f"[Aux-only probe] Test accuracy from '{aux_layer_name}': {acc*100:.2f}%")

    return acc


# ============================================================
# 2. REPRESENTATION / MANIFOLD VISUALIZATION (t-SNE)
# ============================================================

def get_embedding_model(model, layer_name: str = "D1"):
    layer = model.get_layer(layer_name)
    embed_model = keras.Model(inputs=model.input, outputs=layer.output)
    return embed_model


def compute_embeddings(embed_model, x, batch_size: int = 256):
    z = embed_model.predict(x, batch_size=batch_size, verbose=0)
    return z


def tsne_2d(z, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    z2 = tsne.fit_transform(z)
    return z2

def compare_last_layer_with_real_and_aux(teacher,
                                         load_profiles,
                                         labels=None,
                                         path_to_save=None,
                                         n_samples=2000,
                                         random_state=42,
                                         aux_layer_name="aux_logit",
                                         rep_layer_name="D1",
                                         real_layer_name="real_logits",
                                         title_prefix="Teacher"):

    # ---- subsample for speed/clarity ----
    N = load_profiles.shape[0]
    if N > n_samples:
        idx = np.random.default_rng(random_state).choice(N, size=n_samples, replace=False)
        x = load_profiles[idx]
        y_sub = labels[idx] if labels is not None else None
    else:
        x = load_profiles
        y_sub = labels

    dense = get_embedding_model(teacher, rep_layer_name)
    real = get_embedding_model(teacher, real_layer_name)
    aux = get_embedding_model(teacher, aux_layer_name)

    # ---- forward pass ----
    z_rep  = dense.predict(x, verbose=0)
    z_real = real.predict(x, verbose=0)
    z_aux  = aux.predict(x, verbose=0)

    # z_rep_2d = PCA(n_components=2, random_state=random_state).fit_transform(z_rep)
    z_rep_2d = tsne_2d(z_rep, 2, perplexity=20)
    rep_note = f"tSNE (from {z_rep_2d.shape[1]}D)"

    # ---- real_logits: if already 2D, keep; else PCA->2D ----
    if z_real.shape[1] == 2:
        z_real_2d = z_real
        real_note = "raw (2D)"
    else:
        z_real_2d = tsne_2d(z_real, 2, perplexity=20)
        real_note = f"tSNE (from {z_real.shape[1]}D)"
        # z_real_2d = PCA(n_components=2, random_state=random_state).fit_transform(z_real)
        # real_note = f"PCA (from {z_real.shape[1]}D)"

    # ---- aux_logits: if already 2D, keep; else PCA->2D ----
    if z_aux.shape[1] == 2:
        z_aux_2d = z_aux
        aux_note = "raw (2D)"
    else:
        z_aux_2d = tsne_2d(z_aux, 2, perplexity=20)
        aux_note = f"tSNE (from {z_aux.shape[1]}D)"
        # z_aux_2d = PCA(n_components=2, random_state=random_state).fit_transform(z_aux)
        # aux_note = f"PCA (from {z_aux.shape[1]}D)"

    # ---- plotting ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title_prefix}: D1 vs real_logits vs aux_logit", fontsize=14)

    scatter_kwargs = dict(s=12, alpha=0.8)
    if y_sub is not None:
        scatter_kwargs.update(dict(c=y_sub, cmap="coolwarm", vmin=0, vmax=1))

    # Panel 1:
    sc0 = axes[0].scatter(z_rep_2d[:, 0], z_rep_2d[:, 1], **scatter_kwargs)
    axes[0].set_title(f"{rep_layer_name} ({rep_note})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: real logits
    sc1 = axes[1].scatter(z_real_2d[:, 0], z_real_2d[:, 1], **scatter_kwargs)
    axes[1].set_title(f"{real_layer_name} ({real_note})")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Panel 3: aux logits
    sc2 = axes[2].scatter(z_aux_2d[:, 0], z_aux_2d[:, 1], **scatter_kwargs)
    axes[2].set_title(f"{aux_layer_name} ({aux_note})")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # One shared colorbar (if labels provided)
    if y_sub is not None:
        plt.tight_layout(rect=[0.0, 0.0, 0.9, 0.92])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sc2, cax=cbar_ax)
        cbar.set_label("Label", fontsize=11)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0 = no pool", "1 = pool"])
    else:
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    plt.savefig(path_to_save)
    plt.close()

def plot_tsne_three_models(
    z_teacher,
    z_same,
    z_diff,
    y,
    labels=("Teacher", "Student same init", "Student diff init"),
    n_samples: int = 2000,
    title: str = "t-SNE of representations (D1 layer)",
):
    # Subsample if many points
    N = z_teacher.shape[0]
    if N > n_samples:
        idx = np.random.choice(N, size=n_samples, replace=False)
        z_teacher = z_teacher[idx]
        z_same = z_same[idx]
        z_diff = z_diff[idx]
        y = y[idx]

    # t-SNE separately for each model (so we see their internal geometry independently)
    z2_teacher = tsne_2d(z_teacher)
    z2_same = tsne_2d(z_same)
    z2_diff = tsne_2d(z_diff)

    # Wider figure to leave room for a separate colorbar axis on the right
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=16)

    scatter_kwargs = dict(
        cmap="coolwarm",
        alpha=0.8,
        s=10,
        vmin=0,
        vmax=1,
    )

    # Teacher
    sc = axes[0].scatter(z2_teacher[:, 0], z2_teacher[:, 1], c=y, **scatter_kwargs)
    axes[0].set_title(labels[0])
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Student same-init
    axes[1].scatter(z2_same[:, 0], z2_same[:, 1], c=y, **scatter_kwargs)
    axes[1].set_title(labels[1])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Student diff-init
    axes[2].scatter(z2_diff[:, 0], z2_diff[:, 1], c=y, **scatter_kwargs)
    axes[2].set_title(labels[2])
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Make room on the right for the colorbar
    # left, bottom, right, top in figure fraction coordinates
    plt.tight_layout(rect=[0.0, 0.0, 0.9, 0.93])

    # Add a separate colorbar axis completely outside the three plots
    # [left, bottom, width, height] in figure fraction
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Pool presence", fontsize=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0 = no pool", "1 = pool"])

    plt.show()


# ============================================================
# 3. REAL vs NOISE MANIFOLD VISUALIZATION (embedding space)
# ============================================================

def plot_real_vs_noise_embedding(
    embed_model,
    x_real,
    x_noise,
    y_real=None,
    n_samples_real: int = 1000,
    n_samples_noise: int = 1000,
    title: str = "Real vs noise in embedding space",
):
    """
    Visualize where noise lies relative to real data in the teacher's embedding space.

    Parameters
    ----------
    embed_model : tf.keras.Model
        Model mapping input -> embedding (e.g., D1 layer of teacher).
    x_real : np.ndarray
        Real inputs.
    x_noise : np.ndarray
        Noise samples (from noise_ds, collected as a single batch).
    y_real : np.ndarray or None
        Optional labels for real data; if None, real points are plotted in a single color.
    n_samples_real : int
        Subsample size for real points.
    n_samples_noise : int
        Subsample size for noise points.
    title : str
        Plot title.
    """

    # Subsample
    N_real = x_real.shape[0]
    N_noise = x_noise.shape[0]
    idx_real = np.random.choice(N_real, size=min(N_real, n_samples_real), replace=False)
    idx_noise = np.random.choice(N_noise, size=min(N_noise, n_samples_noise), replace=False)

    x_real_sub = x_real[idx_real]
    x_noise_sub = x_noise[idx_noise]

    # Compute embeddings
    z_real = embed_model.predict(x_real_sub, verbose=0)
    z_noise = embed_model.predict(x_noise_sub, verbose=0)

    # Concatenate and t-SNE jointly so we see overlap / separation
    z_all = np.concatenate([z_real, z_noise], axis=0)
    z2_all = tsne_2d(z_all)

    z2_real = z2_all[: z_real.shape[0]]
    z2_noise = z2_all[z_real.shape[0] :]

    plt.figure(figsize=(6, 5))
    if y_real is not None:
        y_real_sub = y_real[idx_real]
        plt.scatter(z2_real[:, 0], z2_real[:, 1], c=y_real_sub, cmap="coolwarm", s=10, alpha=0.8, label="Real")
    else:
        plt.scatter(z2_real[:, 0], z2_real[:, 1], c="blue", s=10, alpha=0.8, label="Real")

    plt.scatter(z2_noise[:, 0], z2_noise[:, 1], c="lime", s=10, alpha=0.6, label="Noise")

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. FEATURE MAP VISUALIZATION (AlexNet-style conv maps)
# ============================================================
def visualize_feature_maps_for_models(
    models: dict,
    x_sample,
    conv_layer_names=("C1", "C2"),
):
    """
    Visualize conv feature maps for one input example across multiple models.

    Parameters
    ----------
    models : dict
        Mapping name -> tf.keras.Model, e.g.
        {
            "Teacher": teacher_model,
            "Student same init": student_same,
            "Student diff init": student_diff,
        }

    x_sample : np.ndarray
        Single input example, shape (7, 96, 1) or batched (1, 7, 96, 1).
        If unbatched, a batch dimension will be added.
    conv_layer_names : tuple of str
        Names of conv layers to visualize (e.g., ("C1", "C2")).

    Notes
    -----
    For each model and each conv layer, this will create a figure where each channel
    is shown as a separate small heatmap.
    """
    if x_sample.ndim == 3:
        x_sample = np.expand_dims(x_sample, axis=0)  # (1, H, W, 1)

    for model_name, model in models.items():
        print(f"\n=== Feature maps for model: {model_name} ===")
        for lname in conv_layer_names:
            try:
                layer = model.get_layer(lname)
            except ValueError:
                print(f"  [WARN] Layer '{lname}' not found in model '{model_name}'. Skipping.")
                continue

            # Build submodel for this layer
            submodel = keras.Model(inputs=model.input, outputs=layer.output)
            fmap = submodel.predict(x_sample, verbose=0)[0]  # (H, W, channels)
            H, W, C = fmap.shape

            # Plot each channel
            fig, axes = plt.subplots(1, C, figsize=(2 * C, 2))
            fig.suptitle(f"{model_name} – layer '{lname}' feature maps", fontsize=12)

            if C == 1:
                axes = [axes]

            for i in range(C):
                ax = axes[i]
                ax.imshow(fmap[:, :, i], aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"ch {i}", fontsize=8)

            plt.tight_layout()
            plt.show()


# ============================================================
# 5. SMALL DRIVER / EXAMPLE USAGE (optional)
# ============================================================
def example_diagnostics_run(
    teacher,
    student_same,
    student_diff,
    x_test,
    y_test,
    x_noise_batch=None,
    label_name: str = "Swimming_Pool",
    embed_layer_name: str = "D1",
):
    """
    - Computes embeddings from teacher / student_same / student_diff (D1 layer).
    - Plots t-SNE manifolds.
    - If noise batch is provided, plots real vs noise in teacher embedding.

    Parameters
    ----------
    teacher, student_same, student_diff : tf.keras.Model
        Trained models.
    x_test : np.ndarray
        Test inputs.
    y_test : np.ndarray
        1D test labels (e.g., 0/1 for pool/no-pool).
    x_noise_batch : np.ndarray or None
        Optional noise samples (e.g., stacked from noise_ds).
        If provided, will be used for real vs noise embedding plot.
    label_name : str
        Used only in titles / printouts.
    embed_layer_name : str
        Name of representation layer (e.g., "D1").
    """

    print(f"Running diagnostics for label '{label_name}' using layer '{embed_layer_name}'")

    # 1) Compute embeddings
    embed_teacher = get_embedding_model(teacher, layer_name=embed_layer_name)
    embed_same = get_embedding_model(student_same, layer_name=embed_layer_name)
    embed_diff = get_embedding_model(student_diff, layer_name=embed_layer_name)

    z_teacher = compute_embeddings(embed_teacher, x_test)
    z_same = compute_embeddings(embed_same, x_test)
    z_diff = compute_embeddings(embed_diff, x_test)

    # 2) t-SNE manifolds
    plot_tsne_three_models(
        z_teacher,
        z_same,
        z_diff,
        y=y_test,
        labels=("Teacher", "Student same init", "Student diff init"),
        n_samples=2000,
        title=f"t-SNE of '{embed_layer_name}' representations for {label_name}",
    )

    # 3) Optional real vs noise embedding
    if x_noise_batch is not None:
        print("Plotting real vs noise in teacher embedding space...")
        plot_real_vs_noise_embedding(
            embed_teacher,
            x_real=x_test,
            x_noise=x_noise_batch,
            y_real=y_test,
            n_samples_real=1000,
            n_samples_noise=1000,
            title=f"Real vs noise in teacher's '{embed_layer_name}' space ({label_name})",
        )

    print("Diagnostics run complete.")

def compare_weekly_snippets_heatmaps(
        x_train: np.ndarray,
        noise_uniform: tf.data.Dataset,
        n_samples: int = 10,
        seed: int = 42,
        save_path: str = None,
        show: bool = True,
):
    rng = np.random.default_rng(seed)

    x_train = np.asarray(x_train)
    if x_train.ndim != 4 or x_train.shape[1:] != (7, 96, 1):
        raise ValueError(f"x_train must have shape (N, 7, 96, 1). Got {x_train.shape}")

    # ---- sample train
    n_train = x_train.shape[0]
    if n_train < n_samples:
        raise ValueError(f"x_train has only {n_train} samples, but n_samples={n_samples}")
    train_idx = rng.choice(n_train, size=n_samples, replace=False)
    train_samples = x_train[train_idx]

    def sample_from_ds(ds: tf.data.Dataset, n: int) -> np.ndarray:
        ds_pick = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=True)
        elems = [e.numpy() for e in ds_pick.unbatch().take(n)]
        return np.stack(elems, axis=0)

    # ---- sample both noise types
    # jittered_samples = sample_from_ds(noise_jittered, n_samples)
    uniform_samples = sample_from_ds(noise_uniform, n_samples)

    # ---- squeeze channel
    train_weeks = train_samples[..., 0]        # (n_samples, 7, 96)
    # jittered_weeks = jittered_samples[..., 0]  # (n_samples, 7, 96)
    uniform_weeks = uniform_samples[..., 0]    # (n_samples, 7, 96)

    # ---- separate scales (two colorbars per column group)
    train_vmin, train_vmax = float(train_weeks.min()), float(train_weeks.max())
    # jit_vmin, jit_vmax = float(jittered_weeks.min()), float(jittered_weeks.max())
    uni_vmin, uni_vmax = float(uniform_weeks.min()), float(uniform_weeks.max())

    # 3 columns: Train | Jittered | Uniform
    fig, axes = plt.subplots(
        nrows=n_samples,
        ncols=3,
        figsize=(16, 2.2 * n_samples),
        constrained_layout=True,
    )
    if n_samples == 1:
        axes = np.array([axes])

    last_train_im = last_jit_im = last_uni_im = None

    for i in range(n_samples):
        ax_t, ax_j, ax_u = axes[i, 0], axes[i, 1], axes[i, 2]

        im_t = ax_t.imshow(
            train_weeks[i],
            aspect="auto",
            interpolation="nearest",
            vmin=train_vmin,
            vmax=train_vmax,
        )
        # im_j = ax_j.imshow(
        #     jittered_weeks[i],
        #     aspect="auto",
        #     interpolation="nearest",
        #     vmin=jit_vmin,
        #     vmax=jit_vmax,
        # )
        im_u = ax_u.imshow(
            uniform_weeks[i],
            aspect="auto",
            interpolation="nearest",
            vmin=uni_vmin,
            vmax=uni_vmax,
        )

        last_train_im, last_jit_im, last_uni_im = im_t, im_j, im_u

        ax_t.set_title(
            f"Train #{i+1} (idx={train_idx[i]})\nmin={train_weeks[i].min():.3f} max={train_weeks[i].max():.3f}"
        )
        # ax_j.set_title(
        #     f"Noise (jittered) #{i+1}\nmin={jittered_weeks[i].min():.3f} max={jittered_weeks[i].max():.3f}"
        # )
        ax_u.set_title(
            f"Noise (uniform) #{i+1}\nmin={uniform_weeks[i].min():.3f} max={uniform_weeks[i].max():.3f}"
        )

        for ax in (ax_t, ax_j, ax_u):
            ax.set_xlabel("Time step (96/day)")
            ax.set_ylabel("Day (0..6)")

    # ---- three separate colorbars (one per column)
    cbar_train = fig.colorbar(last_train_im, ax=axes[:, 0], shrink=0.9, location="left")
    cbar_train.set_label(f"Train value (vmin={train_vmin:.3f}, vmax={train_vmax:.3f})")

    # cbar_jit = fig.colorbar(last_jit_im, ax=axes[:, 1], shrink=0.9, location="right")
    # cbar_jit.set_label(f"Jittered noise (vmin={jit_vmin:.3f}, vmax={jit_vmax:.3f})")

    cbar_uni = fig.colorbar(last_uni_im, ax=axes[:, 2], shrink=0.9, location="right")
    cbar_uni.set_label(f"Uniform noise (vmin={uni_vmin:.3f}, vmax={uni_vmax:.3f})")

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig