#!/bin/bash
# setup_env.sh
#
# Tested, working environment for running TxGNN / DGL-based GNN code on
# Google Colab (or any Linux + Python 3.12 environment).
#
# WHY THE EXACT PINS: installing the latest torch + dgl together fails in
# a chain of four real, verified errors:
#   1. dgl depends on torchdata, and torchdata >=0.8 removed the
#      `datapipes` module dgl's distributed/dataloading code imports.
#   2. After fixing torchdata, dgl's graphbolt module needs `pydantic`,
#      which is not always pre-installed.
#   3. dgl ships PRECOMPILED C++ binaries (graphbolt) only for specific
#      torch versions (verified: 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0,
#      2.2.1 for dgl==2.1.0). A newer torch (e.g. 2.12.x) has no matching
#      .so file and raises FileNotFoundError on `import dgl`.
#   4. dgl's compiled extensions were built against numpy 1.x; numpy 2.x
#      breaks dtype inference (`Could not infer dtype of numpy.int64`).
#
# Run this script top to bottom, in order, before touching any PrimeKG or
# TxGNN code.

set -e

echo "[1/5] Installing torch==2.2.1 (has a matching precompiled DGL graphbolt binary)..."
pip install torch==2.2.1 --index-url https://pypi.org/simple --break-system-packages

echo "[2/5] Installing dgl==2.1.0..."
pip install dgl==2.1.0 --break-system-packages

echo "[3/5] Pinning torchdata==0.7.1 (newer versions removed the datapipes module DGL needs)..."
pip install torchdata==0.7.1 --no-deps --break-system-packages

echo "[4/5] Pinning numpy<2 (DGL's compiled extensions were built against numpy 1.x)..."
pip install "numpy<2" --break-system-packages

echo "[5/5] Installing remaining dependencies (pydantic, scikit-learn)..."
pip install pydantic scikit-learn --break-system-packages

echo ""
echo "Setup complete. Verifying with a smoke test..."
python3 -c "
import dgl
import torch
print('DGL version:', dgl.__version__)
print('Torch version:', torch.__version__)
g = dgl.heterograph({('drug', 'treats', 'disease'): ([0, 1], [0, 1])})
print(g)
print('SMOKE TEST PASSED: DGL heterograph construction works.')
"

echo ""
echo "If you are on Google Colab and this is the FIRST install in this session,"
echo "and the smoke test above failed with a numpy or torch related error,"
echo "go to Runtime > Restart session, then re-run only the smoke test (not the"
echo "installs) to confirm the pinned versions are now active."
