"""
This is my way of thinking for the first step visiualizing vectors. 
  1. Load the pretrained B-cos model from torch hub.
  2. Load TopK-SAE for a specific layer using load_SAE(args).
  3. Hook the SAE into the model with assign_sae_hook(model, layer, sae_model).
  4. Run inference in explanation_mode(), which gives us:
       - p_contribs: how much each concept contributes to the logit (faithful)
       - p_presence: how strongly each concept activates

"""

import sys
import torch
import numpy as np
from argparse import Namespace
from torchvision import transforms as trn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")
from fact.utils import eval_sae, assign_sae_hook, AddInverse
from fact.training.saes import load_SAE
from src.helpers import (
    cosine_sim_matrix, off_diag, top_k_pairs,
    plot_sim_overview, plot_dendrogram, plot_clustered_sim,
    plot_tsne_umap, plot_active_sim, plot_cooccurrence,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # config
    args = Namespace(
        arch="resnet_50",
        layer="norm4",
        method="sae",
        download="v0.1",
        data_path="./ILSVRC2012/val_categorized",  #path to images val
    )
    args.dir_tree = f"{args.arch}/{args.layer}"
    args.config_name = args.method
    print(f"{args.arch} / {args.layer} / {args.method}")

    # load B-cos model
    model = torch.hub.load("B-cos/B-cos-v2", args.arch, pretrained=True)
    model.eval().to(device)
    print(f"Loaded B-cos {args.arch}")

    # load SAE and extract concept vectors (decoder weights)
    sae_model, _, _ = load_SAE(args)
    sae_model.eval().to(device)

    # get decoder weights
    if hasattr(sae_model, "decoder"):
        w = sae_model.decoder.weight.detach().cpu() if hasattr(sae_model.decoder, "weight") \
            else sae_model.decoder.detach().cpu()
    elif hasattr(sae_model, "W_dec"):
        w = sae_model.W_dec.detach().cpu()
    elif hasattr(sae_model, "D"):
        w = sae_model.D.detach().cpu()
    else:
        for n, p in sae_model.named_parameters():
            print(f"  {n}: {p.shape}")
        raise RuntimeError("can't find decoder weights – check names above")

    # ensure (n_concepts, feature_dim)
    if w.shape[0] < w.shape[1]:
        w = w.T

    concept_vectors = w
    n_concepts, feat_dim = concept_vectors.shape
    print(f"{n_concepts} concepts, {feat_dim}-d features, {n_concepts/feat_dim:.1f}x expansion")

    # cosine similarity
    sim = cosine_sim_matrix(concept_vectors)
    od = off_diag(sim)
    print(f"Off-diagonal pairwise similarity:")
    print(f"  mean={od.mean():.4f}  std={od.std():.4f}  min={od.min():.4f}  max={od.max():.4f}")

    # plots
    plot_sim_overview(sim, od, args.arch, args.layer)

    # most similar / most anti-correlated pairs
    most_sim, most_opp = top_k_pairs(sim, k=20)
    print("Most similar:")
    for i, j, s in most_sim:
        print(f"  concept {i} ↔ {j}  sim={s:.4f}")
    print("Most anti-correlated:")
    for i, j, s in most_opp:
        print(f"  concept {i} ↔ {j}  sim={s:.4f}")

    # clustering
    Z = plot_dendrogram(sim, args.arch, args.layer)
    cluster_labels = plot_clustered_sim(sim, Z, n_clusters=20,
                                        arch=args.arch, layer=args.layer)

    # dimensionality reduction
    plot_tsne_umap(concept_vectors, cluster_labels, args.arch, args.layer)

    # SAE vs channel orthogonality
    print(f"SAE  mean |cos sim|: {np.abs(od).mean():.4f}")
    print(f"     |sim| > 0.1:   {(np.abs(od) > 0.1).mean():.2%}")
    print(f"     |sim| > 0.3:   {(np.abs(od) > 0.3).mean():.2%}")
    print(f"Channels mean |cos sim|: 0.0  (orthogonal by definition, but polysemantic)")

    # run inference with SAE hooked in
    transform = trn.Compose([
        trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(),
        AddInverse(),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(Subset(dataset, range(64)), batch_size=16, shuffle=False, num_workers=2)

    hook = assign_sae_hook(model, args.layer, sae_model)
    with model.explanation_mode():
        ret = eval_sae(loader, model=model, sae_model=sae_model,
                       force_label=False, custom_logit=None, one_batch_only=True)
    hook.remove()

    contribs = ret["p_contribs"].detach().cpu().numpy()
    presence = ret["p_presence"].detach().cpu().numpy()
    B = len(ret["images"])
    print(f"Processed {B} images  |  contribs {contribs.shape}  |  presence {presence.shape}")

    # activation analysis
    active_mask = np.abs(presence) > 1e-6
    ever_active = active_mask.any(axis=0)
    n_alive = int(ever_active.sum())
    per_image = active_mask.sum(axis=1)
    print(f"Concepts alive: {n_alive}/{n_concepts}")
    print(f"Active per image: mean={per_image.mean():.1f}  min={per_image.min()}  max={per_image.max()}")

    # top contributing concepts
    mean_c = np.abs(contribs).mean(axis=0)
    for rank, idx in enumerate(np.argsort(mean_c)[::-1][:30]):
        print(f"  {rank+1:2d}. concept {idx}  |contrib|={mean_c[idx]:.4f}  "
              f"active {active_mask[:, idx].sum()}/{B}")

    # similarity among active concepts
    active_idx = np.where(ever_active)[0]
    plot_active_sim(sim, active_idx, args.arch, args.layer)

    # co-occurrence
    cooc = (active_mask.astype(float).T @ active_mask.astype(float))
    freq = active_mask.sum(axis=0).astype(float).clip(min=1)
    cooc_norm = cooc / np.sqrt(np.outer(freq, freq))
    plot_cooccurrence(cooc_norm, sim, freq, args.arch, args.layer)

    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Model:        B-cos {args.arch}")
    print(f"Layer:        {args.layer}")
    print(f"SAE method:   Bias-free TopK-SAE")
    print(f"Concepts:     {n_concepts}  ({feat_dim}-d, {n_concepts/feat_dim:.1f}x expansion)")
    print(f"Mean |sim|:   {np.abs(od).mean():.4f}  "
          f"({'near-orthogonal' if np.abs(od).mean() < 0.05 else 'somewhat correlated'})")
    print(f"|sim| > 0.3:  {(np.abs(od) > 0.3).mean():.2%} of pairs")
    print(f"Alive:        {n_alive}/{n_concepts} concepts fired (on {B} images)")
    print(f"Per image:    {per_image.mean():.1f} active on average")


if __name__ == "__main__":
    main()
