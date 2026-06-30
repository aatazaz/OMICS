# OMICS Exam — Code Deliverables

These files implement the practical/code side of Section 13 (or Section 10
in the IEEE-format) of the paper-generation . **They have been
tested and verified to run correctly** — 

## What was actually verified here (today, by execution — not assumption)

1. **The exact environment conflict chain and fix** — confirmed by hitting
   each error live: `torchdata.datapipes` missing → `pydantic` missing →
   DGL's compiled `graphbolt` binary missing for newer `torch` → `numpy`
   2.x breaking DGL's compiled extensions. The pin set in `setup_env.sh`
   (`torch==2.2.1`, `dgl==2.1.0`, `torchdata==0.7.1`, `numpy<2`) was
   confirmed working by successfully building a DGL heterograph after
   applying it.
2. **`baseline_gnn_dgl.py`** — ran end-to-end on a synthetic PrimeKG-shaped
   graph (60 drugs, 40 diseases, 50 genes): trained for several epochs,
   loss decreased, AUROC/AUPRC/recall@10 computed, model + metrics saved.
3. **`extract_case_studies.py`** — ran end-to-end against the trained
   baseline: correctly identified a zero-treatment disease, ranked
   candidate drugs by score, and attempted relational-path tracing (gene
   overlap / similar-disease transfer).
4. **`evaluate_comparison.py`** — ran end-to-end and produced a real
   Markdown comparison table populated with the baseline's actual measured
   metrics (TxGNN's cells correctly show "N/A" until you supply its real
   metrics file — this is intentional, not a bug, so no number is ever
   silently invented).
5. **`preprocess_primekg.py`** — ran end-to-end against a hand-built CSV
   matching PrimeKG's real, confirmed column schema (`x_id`, `x_type`,
   `y_id`, `y_type`, `relation`, `display_relation` — verified directly
   from the official `mims-harvard/PrimeKG` README's own pandas example),
   correctly extracting drug-disease, disease-disease, disease-gene, and
   drug-gene edges into a DGL heterograph.
6. **The PrimeKG download command** in `download_primekg.sh` is the exact
   command published in the official README as of this writing:
   `wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620`
   — independently cross-checked against the third-party `pykeen` library's
   PrimeKG loader, which hardcodes the same file ID.



## Files

| File | Purpose |
|---|---|
| `setup_env.sh` | Installs the exact tested, working dependency versions. Run this first, in order, then restart the Colab runtime if it's your first install this session. |
| `download_primekg.sh` | Downloads the real, full PrimeKG `kg.csv` from Harvard Dataverse. |
| `preprocess_primekg.py` | Converts `kg.csv` into a DGL heterograph (`primekg_dgl.bin`) that the other scripts expect. **Note:** uses random placeholder node features — for a result you'd defend in the paper, replace these with PrimeKG's real feature files (drug/disease descriptors) per Chandak et al. 2023's feature-construction scripts, referenced in the official repo. |
| `baseline_gnn_dgl.py` | Custom heterogeneous R-GCN baseline (no disease-signature module, no two-phase training, no attention) — this is the "Generic/Standard GNN" column of your Section 8/15 comparison table. |
| `extract_case_studies.py` | Given a trained model + disease ID, returns top-k drug candidates and traces the KG path justifying each one. Use this against **both** the baseline model and a trained/checkpointed **TxGNN** model to get comparable case-study output for both. |
| `evaluate_comparison.py` | Assembles the final comparison table (Markdown) from the baseline's `baseline_metrics.json` and a `txgnn_metrics.json` you produce by running the official TxGNN repo's own evaluation. |

## Run order on Colab

```bash
# 1. Environment (run once per Colab session; restart runtime if first install)
bash setup_env.sh

# 2. Mount Drive so downloads/checkpoints persist
# (in a Python cell)
# from google.colab import drive; drive.mount('/content/drive')

# 3. Download the real PrimeKG
bash download_primekg.sh /content/drive/MyDrive/primekg_raw

# 4. Preprocess into a DGL graph
python preprocess_primekg.py \
    --csv_path /content/drive/MyDrive/primekg_raw/kg.csv \
    --out_path /content/drive/MyDrive/primekg_raw/primekg_dgl.bin

# 5. Train the baseline GNN (your comparison-table baseline)
python baseline_gnn_dgl.py \
    --primekg_dir /content/drive/MyDrive/primekg_raw \
    --epochs 50 \
    --out_dir ./baseline_results

# 6. Separately: clone and run the OFFICIAL TxGNN repo
#    (https://github.com/mims-harvard/TxGNN) against the same PrimeKG,
#    following ITS OWN README for exact phase-1/phase-2 commands.
#    Dump its evaluation metrics to a JSON shaped like:
#    {"metrics": {"auroc": ..., "auprc": ..., "recall_at_10": ...,
#                 "zero_shot_auroc": ..., "zero_shot_auprc": ...}}

# 7. Extract case studies from BOTH models for the same disease IDs
python extract_case_studies.py \
    --graph_path /content/drive/MyDrive/primekg_raw/primekg_dgl.bin \
    --model_path ./baseline_results/baseline_model.pt \
    --disease_id <a real zero/few-shot disease's local id> --top_k 5

# 8. Build the final comparison table
python evaluate_comparison.py \
    --baseline_metrics ./baseline_results/baseline_metrics.json \
    --txgnn_metrics ./txgnn_results/txgnn_metrics.json \
    --out_table ./comparison_table.md
```


