"""
Helper functions for plotting and analysis.
Kept separate from the main script just so that file stays readable.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE


# ── similarity helpers ────────────────────────────────

def cosine_sim_matrix(vecs):
    """Pairwise cosine similarity. vecs shape: (n, d)."""
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = vecs / norms
    return (normed @ normed.T).numpy()


def off_diag(sim):
    """Return off-diagonal values as a flat array."""
    mask = ~np.eye(sim.shape[0], dtype=bool)
    return sim[mask]


def top_k_pairs(sim, k=20):
    """k most similar + k most anti-correlated pairs (upper triangle only)."""
    idx = np.triu_indices(sim.shape[0], k=1)
    s = sim[idx]
    order = np.argsort(s)
    most_sim = [(idx[0][order[-(i+1)]], idx[1][order[-(i+1)]], s[order[-(i+1)]]) for i in range(k)]
    most_opp = [(idx[0][order[i]],      idx[1][order[i]],      s[order[i]])      for i in range(k)]
    return most_sim, most_opp


# ── plots ─────────────────────────────────────────────

def plot_sim_overview(sim, off_vals, arch, layer):
    """Heatmap + histogram of pairwise similarities."""
    n = sim.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    im = ax1.imshow(sim, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax1.set_title(f"Cosine Similarity – {n} SAE Concepts\n({arch} / {layer})")
    ax1.set_xlabel("Concept"); ax1.set_ylabel("Concept")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    ax2.hist(off_vals, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    ax2.axvline(0, color="red", ls="--", alpha=0.5)
    ax2.set_title("Pairwise Similarity Distribution")
    ax2.set_xlabel("Cosine Similarity"); ax2.set_ylabel("Count")
    ax2.annotate(f"mean={off_vals.mean():.4f}\nstd={off_vals.std():.4f}",
                 xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig("concept_similarity_overview.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_dendrogram(sim, arch, layer):
    """Ward-linkage dendrogram on (1 - cos) distances."""
    dist = np.clip(1 - sim, 0, 2)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="ward")

    n = sim.shape[0]
    fig, ax = plt.subplots(figsize=(20, 6))
    dendrogram(Z, truncate_mode="lastp" if n > 200 else None,
               p=min(n, 200), leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title(f"Concept Clustering ({arch} / {layer})")
    ax.set_ylabel("Distance (1 − cos sim)")
    plt.tight_layout()
    plt.savefig("concept_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.show()
    return Z


def plot_clustered_sim(sim, Z, n_clusters, arch, layer):
    """Reorder heatmap by cluster assignment."""
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    order = np.argsort(labels)
    ordered = sim[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(ordered, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_title(f"Similarity (sorted into {n_clusters} clusters)\n({arch} / {layer})")
    sizes = np.bincount(labels[order])[1:]
    for b in np.cumsum(sizes)[:-1]:
        ax.axhline(b - 0.5, color="k", lw=0.5, alpha=0.5)
        ax.axvline(b - 0.5, color="k", lw=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig("concept_similarity_clustered.png", dpi=150, bbox_inches="tight")
    plt.show()
    return labels


def plot_tsne_umap(vecs, cluster_labels, arch, layer):
    """t-SNE (+ UMAP if installed) 2-D scatter of concept vectors."""
    data = vecs.numpy() if hasattr(vecs, "numpy") else vecs
    n = data.shape[0]

    try:
        from umap import UMAP
        has_umap = True
    except ImportError:
        has_umap = False

    ncols = 2 if has_umap else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1: axes = [axes]

    coords = TSNE(n_components=2, random_state=42,
                  perplexity=min(30, n - 1)).fit_transform(data)
    axes[0].scatter(coords[:, 0], coords[:, 1],
                    c=cluster_labels, cmap="tab20", s=10, alpha=0.7)
    axes[0].set_title(f"t-SNE – {n} concepts")

    if has_umap:
        cu = UMAP(n_components=2, random_state=42,
                  n_neighbors=min(15, n - 1)).fit_transform(data)
        axes[1].scatter(cu[:, 0], cu[:, 1],
                        c=cluster_labels, cmap="tab20", s=10, alpha=0.7)
        axes[1].set_title(f"UMAP – {n} concepts")

    plt.suptitle(f"{arch} / {layer}", fontsize=14)
    plt.tight_layout()
    plt.savefig("concept_embeddings_2d.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_active_sim(sim, active_idx, arch, layer):
    """Heatmap restricted to concepts that actually fired."""
    sub = sim[np.ix_(active_idx, active_idx)]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(sub, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_title(f"Similarity – {len(active_idx)} Active Concepts\n({arch} / {layer})")
    plt.tight_layout()
    plt.savefig("active_concept_similarity.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_cooccurrence(cooc_norm, sim, freq, arch, layer):
    """Co-occurrence vs cosine-similarity side by side (top 100 concepts)."""
    top = np.argsort(freq)[::-1][:100]
    c_sub = cooc_norm[np.ix_(top, top)]
    s_sub = sim[np.ix_(top, top)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.imshow(c_sub, cmap="YlOrRd", aspect="auto"); ax1.set_title("Co-occurrence (top 100)")
    ax2.imshow(s_sub, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto"); ax2.set_title("Cosine Sim (same 100)")
    plt.suptitle(f"Co-occurrence vs Similarity ({arch} / {layer})", fontsize=14)
    plt.tight_layout()
    plt.savefig("cooccurrence_vs_similarity.png", dpi=150, bbox_inches="tight")
    plt.show()

    upper = np.triu_indices(len(top), k=1)
    corr = np.corrcoef(c_sub[upper], s_sub[upper])[0, 1]
    print(f"Correlation co-occurrence ↔ similarity: {corr:.4f}")
    return corr
