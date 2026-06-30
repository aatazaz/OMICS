"""
extract_case_studies.py

Given a trained model (TxGNN checkpoint or the baseline_gnn_dgl.py model)
and a target disease with zero/few known treatments in the training split,
extract the top-k predicted drug candidates and trace the relational path
through PrimeKG (shared genes/pathways with similar, better-characterized
diseases) that justifies each prediction.

This produces the real content for Section 7 (Case Studies) of the paper --
the output is grounded in the actual trained model and actual KG structure,
not invented.

Usage:
    python extract_case_studies.py --graph_path ./test_primekg/primekg_dgl.bin \
        --model_path ./test_results/baseline_model.pt \
        --disease_id 7 --top_k 5
"""

import argparse
import json

import dgl
import torch

from baseline_gnn_dgl import BaselineGNN


def load_model(graph_path, model_path, hidden_dim=128, out_dim=64):
    graphs, _ = dgl.load_graphs(graph_path)
    g = graphs[0]
    rel_names = g.etypes
    ntype_in_dims = {ntype: g.nodes[ntype].data["feat"].shape[1]
                      for ntype in g.ntypes}
    model = BaselineGNN(ntype_in_dims, hidden_dim, out_dim, rel_names)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, g


def get_top_k_candidates(model, g, disease_id, top_k=5):
    """Score every drug against the target disease and return the top-k."""
    feat_dict = {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}
    with torch.no_grad():
        h = model.encoder(g, feat_dict)
        h_drug, h_disease = h["drug"], h["disease"]
        disease_vec = h_disease[disease_id].unsqueeze(0)          # [1, out_dim]
        scores = (h_drug * disease_vec).sum(dim=-1)                # [num_drugs]
    top_scores, top_drug_ids = torch.topk(scores, k=top_k)
    return list(zip(top_drug_ids.tolist(), top_scores.tolist()))


def trace_relational_path(g, disease_id, drug_id):
    """Find the shortest justification path: drug -> gene <- disease,
    or drug -> disease' (similar) -> gene <- disease, mirroring the kind
    of multi-hop relational evidence TxGNN's signature module aggregates.
    """
    paths = []

    # Path type 1: shared gene target
    drug_genes = set(g.successors(drug_id, etype="targets").tolist())
    disease_genes = set(g.successors(disease_id, etype="associated_with").tolist())
    shared = drug_genes & disease_genes
    for gene_id in shared:
        paths.append({
            "type": "shared_gene_target",
            "path": f"drug_{drug_id} --targets--> gene_{gene_id} <--associated_with-- disease_{disease_id}",
        })

    # Path type 2: via a similar disease that the drug already treats
    similar_diseases = set(g.successors(disease_id, etype="similar").tolist())
    drugs_for_similar = set()
    for sim_d in similar_diseases:
        drugs_for_similar.update(g.predecessors(sim_d, etype="treats").tolist())
    if drug_id in drugs_for_similar:
        paths.append({
            "type": "similar_disease_transfer",
            "path": f"drug_{drug_id} --treats--> similar_disease(s) of disease_{disease_id}",
        })

    if not paths:
        paths.append({
            "type": "embedding_similarity_only",
            "path": "No explicit 1-2 hop symbolic path found; prediction driven by "
                     "learned embedding proximity rather than a directly traceable KG path.",
        })
    return paths


def run_case_study(graph_path, model_path, disease_id, top_k):
    model, g = load_model(graph_path, model_path)

    # Confirm this disease is genuinely zero/few-shot in this graph
    existing_treats = g.predecessors(disease_id, etype="treats-rev") if "treats-rev" in g.etypes else torch.tensor([])

    candidates = get_top_k_candidates(model, g, disease_id, top_k)

    case_study = {
        "disease_id": disease_id,
        "num_known_treatments_in_graph": int(len(existing_treats)) if hasattr(existing_treats, "__len__") else None,
        "top_k_predicted_drugs": [],
    }

    for drug_id, score in candidates:
        paths = trace_relational_path(g, disease_id, drug_id)
        case_study["top_k_predicted_drugs"].append({
            "drug_id": drug_id,
            "score": score,
            "justification_paths": paths,
        })

    return case_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--disease_id", type=int, required=True,
                         help="Node ID (within the 'disease' ntype) of the target disease")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_file", type=str, default=None)
    args = parser.parse_args()

    result = run_case_study(args.graph_path, args.model_path, args.disease_id, args.top_k)
    print(json.dumps(result, indent=2))

    if args.out_file:
        with open(args.out_file, "w") as f:
            json.dump(result, f, indent=2)
