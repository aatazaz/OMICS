"""
preprocess_primekg.py

Converts the raw PrimeKG edge-list CSV (kg.csv, downloaded via
download_primekg.sh) into a DGL heterogeneous graph (primekg_dgl.bin),
which is the format baseline_gnn_dgl.py and extract_case_studies.py expect.

Column schema confirmed from the official PrimeKG README
(https://github.com/mims-harvard/PrimeKG) and its quick-start example:

    primekg = pd.read_csv('kg.csv', low_memory=False)
    primekg.query('y_type=="disease"|x_type=="disease"')

This confirms the columns x_type / y_type exist. PrimeKG's documented full
schema (per Chandak et al. 2023, Scientific Data) is:
    x_index, x_id, x_type, x_name, x_source,
    y_index, y_id, y_type, y_name, y_source,
    relation, display_relation

Node types include: drug, disease, gene/protein, effect/phenotype,
anatomy, biological_process, molecular_function, cellular_component,
pathway, exposure. This script focuses on the drug / disease / gene-protein
subgraph relevant to TxGNN-style drug-repurposing prediction, but the
mapping generalizes to the full node-type set if you need it.

Usage:
    python preprocess_primekg.py --csv_path /content/drive/MyDrive/primekg_raw/kg.csv \
                                  --out_path /content/drive/MyDrive/primekg_raw/primekg_dgl.bin
"""

import argparse

import dgl
import numpy as np
import pandas as pd
import torch


# PrimeKG uses 'gene/protein' as the literal type string for genes/proteins.
RELEVANT_NTYPES = ["drug", "disease", "gene/protein"]

# PrimeKG's "relation" column groups many fine-grained relation types; the
# ones below are the documented core relation strings relevant to drug
# repurposing (drug-disease indication/contraindication/off-label, and
# disease-disease / drug-gene / disease-gene structural relations used for
# the zero-shot disease-similarity signature). Verify against your local
# kg.csv (`primekg['relation'].unique()`) since PrimeKG has had multiple
# dataset versions (e.g. the Dec 2023 OMIM extension) that may add relations.
REL_DRUG_DISEASE = ["indication", "contraindication", "off-label use"]
REL_DISEASE_DISEASE = ["disease_disease", "parent-child"]
REL_GENE_DISEASE = ["disease_protein"]
REL_DRUG_GENE = ["drug_protein"]


def build_id_maps(df, ntype):
    """Map PrimeKG's global x_id/y_id (per type) to contiguous 0..N-1 ids."""
    ids = pd.concat([
        df.loc[df["x_type"] == ntype, "x_id"],
        df.loc[df["y_type"] == ntype, "y_id"],
    ]).unique()
    return {pid: i for i, pid in enumerate(sorted(ids, key=str))}


def extract_edges(df, relation_list, src_type, dst_type, id_maps):
    """Pull (src_local_id, dst_local_id) pairs for a set of relation strings,
    respecting PrimeKG's x/y typed-column convention (an edge may appear
    with src/dst order in either x/y position).
    """
    mask = df["relation"].isin(relation_list)
    sub = df.loc[mask]

    src, dst = [], []
    # Case 1: x is src_type, y is dst_type
    m1 = sub[(sub["x_type"] == src_type) & (sub["y_type"] == dst_type)]
    src += m1["x_id"].map(id_maps[src_type]).tolist()
    dst += m1["y_id"].map(id_maps[dst_type]).tolist()
    # Case 2: y is src_type, x is dst_type (PrimeKG does not guarantee a
    # fixed direction in the x/y columns for symmetric relation types)
    if src_type != dst_type:
        m2 = sub[(sub["y_type"] == src_type) & (sub["x_type"] == dst_type)]
        src += m2["y_id"].map(id_maps[src_type]).tolist()
        dst += m2["x_id"].map(id_maps[dst_type]).tolist()

    return torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)


def main(args):
    print(f"Loading {args.csv_path} ...")
    df = pd.read_csv(args.csv_path, low_memory=False)
    print(f"Loaded {len(df):,} rows.")
    print("Detected relation types:", sorted(df["relation"].unique())[:20], "...")
    print("Detected node types:", sorted(set(df["x_type"]).union(df["y_type"])))

    id_maps = {nt: build_id_maps(df, nt) for nt in RELEVANT_NTYPES}
    for nt, m in id_maps.items():
        print(f"  {nt}: {len(m):,} unique nodes")

    graph_data = {}

    src, dst = extract_edges(df, REL_DRUG_DISEASE, "drug", "disease", id_maps)
    graph_data[("drug", "treats", "disease")] = (src, dst)
    graph_data[("disease", "treats-rev", "drug")] = (dst, src)
    print(f"  drug-treats-disease edges: {len(src):,}")

    src, dst = extract_edges(df, REL_DISEASE_DISEASE, "disease", "disease", id_maps)
    graph_data[("disease", "similar", "disease")] = (src, dst)
    print(f"  disease-similar-disease edges: {len(src):,}")

    src, dst = extract_edges(df, REL_GENE_DISEASE, "gene/protein", "disease", id_maps)
    graph_data[("gene", "associated_with-rev", "disease")] = (src, dst)
    graph_data[("disease", "associated_with", "gene")] = (dst, src)
    print(f"  disease-associated_with-gene edges: {len(src):,}")

    src, dst = extract_edges(df, REL_DRUG_GENE, "drug", "gene/protein", id_maps)
    graph_data[("drug", "targets", "gene")] = (src, dst)
    graph_data[("gene", "targets-rev", "drug")] = (dst, src)
    print(f"  drug-targets-gene edges: {len(src):,}")

    num_nodes_dict = {
        "drug": len(id_maps["drug"]),
        "disease": len(id_maps["disease"]),
        "gene": len(id_maps["gene/protein"]),
    }

    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    # Placeholder node features (random init). For a real run, replace with
    # actual feature vectors (e.g. drug chemical descriptors, gene
    # expression profiles, disease phenotype embeddings) per Chandak et al.
    # 2023's feature_construction scripts, rather than random vectors.
    rng = np.random.default_rng(0)
    for ntype in g.ntypes:
        g.nodes[ntype].data["feat"] = torch.tensor(
            rng.standard_normal((g.num_nodes(ntype), args.feat_dim)),
            dtype=torch.float32,
        )

    print(g)
    dgl.save_graphs(args.out_path, [g])
    print(f"Saved DGL heterograph to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--feat_dim", type=int, default=128)
    args = parser.parse_args()
    main(args)
