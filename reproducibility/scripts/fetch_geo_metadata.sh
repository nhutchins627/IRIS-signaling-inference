#!/usr/bin/env bash
#
# Download GEO/ArrayExpress metadata for every dataset used in the IRIS paper.
#
#   ./fetch_geo_metadata.sh [OUTDIR]        # metadata only (~a few MB)
#   ./fetch_geo_metadata.sh OUTDIR --suppl  # also processed supplementary files (~GB)
#
# Default OUTDIR is ./geo_metadata
#
# For each series this fetches:
#   <ACC>_family.soft.gz    SOFT   - full series+sample metadata, flat text
#   <ACC>_family.xml.tgz    MINiML - the same in XML, plus per-sample files
#   <ACC>_suppl_list.txt    an index of the processed files on the FTP site
#
# Note: NCBI's "download all supplementary files" button produces a **TAR**
# (RAW.tar), not a ZIP, and it is only offered for series with several files --
# GSE136689 returns 404 there, so its four processed files are fetched directly.

set -uo pipefail

OUTDIR="${1:-geo_metadata}"
SUPPL="${2:-}"
mkdir -p "$OUTDIR"

# accession:description
SERIES=(
  "GSE289836:hM_d4, hM_d7, hE_d8 combinatorial screens (this study)"
  "GSE122009:mESC screen (Yeo et al. 2020)"
  "GSE136689:mouse foregut organogenesis (Han et al. 2020)"
  "GSE246368:human airway epithelium (McCauley et al. 2024)"
)
ARRAYEXPRESS="E-MTAB-6967"   # mouse gastrulation atlas (Pijuan-Sala et al. 2019)

# GEO shards series into .../GSExxxnnn/ by dropping the last three digits.
ftp_dir() { local a="$1"; echo "https://ftp.ncbi.nlm.nih.gov/geo/series/${a%???}nnn/${a}"; }

get() {  # get <url> <dest>
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 -# -o "$2" "$1"
}

echo "Writing to $(cd "$OUTDIR" && pwd)"
echo

for entry in "${SERIES[@]}"; do
  ACC="${entry%%:*}"
  DESC="${entry#*:}"
  BASE="$(ftp_dir "$ACC")"
  DEST="$OUTDIR/$ACC"
  mkdir -p "$DEST"

  echo "=== $ACC — $DESC"

  get "$BASE/soft/${ACC}_family.soft.gz"  "$DEST/${ACC}_family.soft.gz" \
    && echo "    SOFT   ok" || echo "    SOFT   FAILED"
  get "$BASE/miniml/${ACC}_family.xml.tgz" "$DEST/${ACC}_family.xml.tgz" \
    && echo "    MINiML ok" || echo "    MINiML FAILED"

  # Index the processed files without downloading them.
  curl -fsL --max-time 60 "$BASE/suppl/" 2>/dev/null \
    | grep -oE 'href="[^"?/][^"]*"' | sed 's/href="//;s/"$//' \
    | grep -v '^https\?:' > "$DEST/${ACC}_suppl_list.txt" || true
  n=$(wc -l < "$DEST/${ACC}_suppl_list.txt" 2>/dev/null || echo 0)
  echo "    suppl  ${n} processed file(s) indexed"

  if [ "$SUPPL" = "--suppl" ] && [ "$n" -gt 0 ]; then
    echo "    downloading processed files ..."
    while read -r f; do
      [ -n "$f" ] && get "$BASE/suppl/$f" "$DEST/$f" && echo "      $f"
    done < "$DEST/${ACC}_suppl_list.txt"
  fi
  echo
done

echo "=== $ARRAYEXPRESS — mouse gastrulation atlas (ArrayExpress)"
AE="https://www.ebi.ac.uk/biostudies/files/${ARRAYEXPRESS}"
mkdir -p "$OUTDIR/$ARRAYEXPRESS"
for f in "${ARRAYEXPRESS}.idf.txt" "${ARRAYEXPRESS}.sdrf.txt"; do
  get "$AE/$f" "$OUTDIR/$ARRAYEXPRESS/$f" && echo "    $f ok" || echo "    $f FAILED"
done

echo
echo "Done. Contents:"
du -sh "$OUTDIR"/*/ 2>/dev/null
echo
echo "Read a SOFT file with:   zcat $OUTDIR/GSE289836/GSE289836_family.soft.gz | less"
echo "Unpack MINiML with:      tar xzf $OUTDIR/GSE289836/GSE289836_family.xml.tgz -C $OUTDIR/GSE289836/"
