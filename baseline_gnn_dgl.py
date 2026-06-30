import argparse
import json
import os

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Model definition
# ---------------------------------------------------------------------------
class RGCNEncoder(nn.Module):
    """Two-layer heterogeneous R-GCN encoder.

    Produces an embedding per node type using relation-specific weight
    matrices (HeteroGraphConv wrapping one GraphConv per edge type), with
    uniform (non-attention) neighbor aggregation -- this is the key
    architectural contrast with TxGNN's signature-weighted aggregation.
    """

    def __init__(self, ntype_in_dims, hidden_dim, out_dim, rel_names):
        super().__init__()
        # Per-node-type linear projection into a shared hidden space.
        self.embed = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim)
            for ntype, in_dim in ntype_in_dims.items()
        })

        self.layer1 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(hidden_dim, hidden_dim, norm="right",
                                   weight=True, bias=True)
             for rel in rel_names},
            aggregate="mean",
        )
        self.layer2 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(hidden_dim, out_dim, norm="right",
                                   weight=True, bias=True)
             for rel in rel_names},
            aggregate="mean",
        )

    def forward(self, g, feat_dict):
        h = {ntype: F.relu(self.embed[ntype](feat))
             for ntype, feat in feat_dict.items()}
        h = self.layer1(g, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.layer2(g, h)
        return h  # dict: ntype -> [num_nodes_of_type, out_dim]


class DotProductDecoder(nn.Module):
    """Score a (drug, disease) pair via dot product of their embeddings.

    Standard supervised link-prediction decoder -- no disease-signature
    similarity transfer, in contrast to TxGNN's design.
    """

    def forward(self, h_drug, h_disease, src_idx, dst_idx):
        return (h_drug[src_idx] * h_disease[dst_idx]).sum(dim=-1)


class BaselineGNN(nn.Module):
    def __init__(self, ntype_in_dims, hidden_dim, out_dim, rel_names):
        super().__init__()
        self.encoder = RGCNEncoder(ntype_in_dims, hidden_dim, out_dim, rel_names)
        self.decoder = DotProductDecoder()

    def forward(self, g, feat_dict, src_idx, dst_idx):
        h = self.encoder(g, feat_dict)
        scores = self.decoder(h["drug"], h["disease"], src_idx, dst_idx)
        return scores, h


# ---------------------------------------------------------------------------
# 2. Training / evaluation loop
# ---------------------------------------------------------------------------
def compute_metrics(pos_scores, neg_scores):
    """AUROC, AUPRC, and recall@10 over a pos/neg scored edge set."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    scores = torch.cat([pos_scores, neg_scores]).detach().cpu().numpy()
    labels = np.concatenate([
        np.ones(len(pos_scores)), np.zeros(len(neg_scores))
    ])
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # recall@10: of the top-10 ranked overall, how many are true positives
    order = np.argsort(-scores)
    top10 = order[:10]
    recall_at_10 = labels[top10].sum() / max(1, labels.sum())
    return {"auroc": float(auroc), "auprc": float(auprc),
            "recall_at_10": float(recall_at_10)}


def sample_negative_edges(num_drugs, num_diseases, num_samples, exclude_set):
    """Uniform random negative sampling of (drug, disease) non-edges."""
    negs = []
    while len(negs) < num_samples:
        d = np.random.randint(0, num_drugs)
        s = np.random.randint(0, num_diseases)
        if (d, s) not in exclude_set:
            negs.append((d, s))
    return np.array(negs)


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load graph -------------------------------------------------------
    # Expects a preprocessed DGL heterograph saved via dgl.save_graphs from
    # PrimeKG's raw edge list (see download_primekg.sh / a preprocessing
    # step not shown here, which maps PrimeKG node/edge CSVs -> DGL ids).
    graph_path = os.path.join(args.primekg_dir, "primekg_dgl.bin")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Expected preprocessed graph at {graph_path}. "
            "Run preprocess_primekg.py first (see README)."
        )
    graphs, _ = dgl.load_graphs(graph_path)
    g = graphs[0]

    rel_names = g.etypes
    ntype_in_dims = {ntype: g.nodes[ntype].data["feat"].shape[1]
                      for ntype in g.ntypes}
    feat_dict = {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}

    model = BaselineGNN(ntype_in_dims, args.hidden_dim, args.out_dim, rel_names)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Train/test split on (drug, treats, disease) edges ---------------
    eids = g.edges(etype="treats", form="eid")
    src, dst = g.find_edges(eids, etype="treats")
    perm = torch.randperm(len(eids))
    n_train = int(0.8 * len(eids))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    num_drugs = g.num_nodes("drug")
    num_diseases = g.num_nodes("disease")
    pos_set = set(zip(src.tolist(), dst.tolist()))

    history = []
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        scores, _ = model(g, feat_dict, src[train_idx], dst[train_idx])
        neg = sample_negative_edges(num_drugs, num_diseases, len(train_idx), pos_set)
        neg_scores, _ = model(g, feat_dict,
                               torch.tensor(neg[:, 0]), torch.tensor(neg[:, 1]))
        labels = torch.cat([torch.ones_like(scores), torch.zeros_like(neg_scores)])
        all_scores = torch.cat([scores, neg_scores])
        loss = F.binary_cross_entropy_with_logits(all_scores, labels)
        loss.backward()
        opt.step()
        history.append({"epoch": epoch, "loss": float(loss.item())})
        if epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

    # --- Evaluate ----------------------------------------------------------
    model.eval()
    with torch.no_grad():
        pos_scores, h = model(g, feat_dict, src[test_idx], dst[test_idx])
        neg = sample_negative_edges(num_drugs, num_diseases, len(test_idx), pos_set)
        neg_scores, _ = model(g, feat_dict,
                               torch.tensor(neg[:, 0]), torch.tensor(neg[:, 1]))
        metrics = compute_metrics(pos_scores, neg_scores)

    print("Baseline R-GCN test metrics:", metrics)

    with open(os.path.join(args.out_dir, "baseline_metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "history": history}, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "baseline_model.pt"))
    print(f"Saved metrics + model to {args.out_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primekg_dir", type=str, required=True,
                         help="Directory containing preprocessed primekg_dgl.bin")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="./baseline_results")

    # Explicitly pass arguments to parse_args() for Colab execution
    args = parser.parse_args([
        "--primekg_dir", "/content/drive/MyDrive/OMICS/primekg_raw",
        "--epochs", "50",
        "--out_dir", "./baseline_results"
    ])
    train(args)
