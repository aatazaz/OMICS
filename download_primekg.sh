#!/bin/bash
# download_primekg.sh
#
# Downloads the REAL, FULL PrimeKG dataset (not a reduced/demo subset).
# PrimeKG's official distribution is via Harvard Dataverse, linked from
# the official repo: https://github.com/mims-harvard/PrimeKG
#
# IMPORTANT: verify the exact current download URL/DOI on the repo's README
# before running this — Dataverse dataset URLs and file names can change,
# and this script's hardcoded path may go stale. As of writing, the dataset
# is published at the Harvard Dataverse under the PrimeKG dataset entry.
#
# On Colab: mount Google Drive FIRST so this multi-GB download persists
# across sessions instead of vanishing when the runtime disconnects:
#   from google.colab import drive
#   drive.mount('/content/drive')

set -e

OUT_DIR="${1:-/content/drive/MyDrive/primekg_raw}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "Downloading PrimeKG into $OUT_DIR ..."

# This is the exact command published in the official PrimeKG README
# (https://github.com/mims-harvard/PrimeKG), confirmed current as of this
# writing and independently confirmed by the third-party pykeen library's
# PrimeKG dataset loader, which hardcodes the same file ID (6180620).
# The underlying dataset's persistent identifier is doi:10.7910/DVN/IXA7BM
# on Harvard Dataverse, if you need to verify/browse it in a browser:
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
#
# Still worth a quick check against the live README before running, in case
# the file ID changes after a future PrimeKG re-release.

wget -c "https://dataverse.harvard.edu/api/access/datafile/6180620" -O kg.csv || {
    echo "Direct wget failed -- fall back to manual download:"
    echo "1. Visit https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM"
    echo "2. Download kg.csv (the full edge list) manually."
    echo "3. Upload it to $OUT_DIR (e.g. via Colab's Drive mount)."
    exit 1
}

echo "Download step complete (or manual fallback instructions printed above)."
echo "Expected file: $OUT_DIR/kg.csv"
echo "This single CSV contains PrimeKG's full edge list (~4M relationships,"
echo "17,080 diseases, drugs, genes/proteins, phenotypes, etc. per Chandak"
echo "et al. 2023, Scientific Data)."
echo "Next step: run preprocess_primekg.py to convert this into a DGL heterograph."
