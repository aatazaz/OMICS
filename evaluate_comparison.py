"""
evaluate_comparison.py

Loads metrics produced by:
  - baseline_gnn_dgl.py        -> ./baseline_results/baseline_metrics.json
  - the official TxGNN repo's evaluation output (run separately; see
    run_txgnn_phase2.sh, which should be configured to dump a similarly
    structured JSON with keys: auroc, auprc, recall_at_10, plus
    zero_shot_auroc / zero_shot_auprc for the rare-disease split)

and assembles the real comparison table used in Section 8 of the paper.

Usage:
    python evaluate_comparison.py \
        --baseline_metrics ./baseline_results/baseline_metrics.json \
        --txgnn_metrics ./txgnn_results/txgnn_metrics.json \
        --out_table ./comparison_table.md
"""

import argparse
import json


ROWS = [
    ("Input representation", "row_input_repr"),
    ("Training paradigm", "row_training"),
    ("Handling of rare/zero-shot entities", "row_zeroshot"),
    ("Embedding strategy", "row_embedding"),
    ("Loss function(s)", "row_loss"),
    ("Interpretability", "row_interp"),
    ("Computational cost", "row_cost"),
    ("AUROC (overall test split)", "auroc"),
    ("AUPRC (overall test split)", "auprc"),
    ("Recall@10 (overall test split)", "recall_at_10"),
    ("AUROC (zero-shot / rare-disease split)", "zero_shot_auroc"),
    ("AUPRC (zero-shot / rare-disease split)", "zero_shot_auprc"),
    ("Typical use case", "row_use_case"),
    ("Key limitation", "row_limitation"),
]

# Qualitative/architectural rows are fixed descriptive text (not measured);
# quantitative rows are pulled from the actual metrics JSON files.
QUALITATIVE = {
    "row_input_repr": ("Homogeneous-treated or simple heterogeneous node/edge "
                        "features from PrimeKG", "Same PrimeKG features, plus "
                        "disease-similarity signature vectors"),
    "row_training": ("Single-phase end-to-end supervised link prediction",
                      "Two-phase: (1) self-supervised pretraining on full KG, "
                      "(2) fine-tuning with metric-learning signature module"),
    "row_zeroshot": ("Not explicitly handled; relies on whatever incidental "
                      "structural signal exists", "Explicitly handled via "
                      "disease-similarity signature transfer from related diseases"),
    "row_embedding": ("Uniform relation-aware message passing (R-GCN), no "
                       "attention", "Relation-aware message passing + optional "
                       "attention-weighted aggregation"),
    "row_loss": ("Binary cross-entropy / margin loss on observed vs. sampled "
                  "negative edges", "Link-prediction loss (phase 1) + "
                  "metric-learning ranking loss over disease signatures (phase 2)"),
    "row_interp": ("Low-to-moderate (embedding dot-products only)",
                    "Moderate (signature similarity gives some traceability "
                    "to which diseases informed a prediction)"),
    "row_cost": ("Lower (single training run)", "Higher (two training phases; "
                  "signature construction adds overhead)"),
    "row_use_case": ("Well-characterized diseases/drugs with sufficient "
                      "training examples", "Rare diseases with little/no "
                      "labeled treatment data"),
    "row_limitation": ("Poor generalization to diseases absent/sparse in "
                        "training data", "Added architectural complexity; "
                        "performance depends on KG coverage/quality for the "
                        "disease-similarity signal"),
}


def load_metrics(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f).get("metrics", {})


def build_table(baseline_metrics, txgnn_metrics):
    lines = ["| Criterion | Generic/Standard GNN | TxGNN |",
             "|---|---|---|"]
    for label, key in ROWS:
        if key in QUALITATIVE:
            baseline_val, txgnn_val = QUALITATIVE[key]
        else:
            baseline_val = baseline_metrics.get(key, "N/A (not measured)")
            txgnn_val = txgnn_metrics.get(key, "N/A (not measured)")
            if isinstance(baseline_val, float):
                baseline_val = f"{baseline_val:.4f}"
            if isinstance(txgnn_val, float):
                txgnn_val = f"{txgnn_val:.4f}"
        lines.append(f"| {label} | {baseline_val} | {txgnn_val} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", type=str, required=True)
    parser.add_argument("--txgnn_metrics", type=str, default=None,
                         help="If omitted, TxGNN quantitative cells are marked N/A "
                              "until you supply the official repo's eval output.")
    parser.add_argument("--out_table", type=str, default="./comparison_table.md")
    args = parser.parse_args()

    baseline_metrics = load_metrics(args.baseline_metrics)
    txgnn_metrics = load_metrics(args.txgnn_metrics)

    table_md = build_table(baseline_metrics, txgnn_metrics)
    print(table_md)

    with open(args.out_table, "w") as f:
        f.write(table_md + "\n")
    print(f"\nSaved table to {args.out_table}")
