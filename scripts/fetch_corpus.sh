#!/usr/bin/env bash
#
# Download German federal law corpora from gesetze-im-internet.de.
#
#   ./scripts/fetch_corpus.sh              # BGB only
#   ./scripts/fetch_corpus.sh bgb mietrecht_neu
#   WITH_PDF=1 ./scripts/fetch_corpus.sh   # also grab the human-readable PDF
#
# Each law lands in data/raw/<law>/ alongside a provenance.txt recording the
# URL, server timestamp and checksum, so a corpus can be tied to a specific
# amendment state later.

set -euo pipefail

BASE_URL="https://www.gesetze-im-internet.de"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"

LAWS=("$@")
if [ ${#LAWS[@]} -eq 0 ]; then
  LAWS=(bgb)
fi

sha256() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

fetch_law() {
  law="$1"
  dest="$RAW_DIR/$law"
  url="$BASE_URL/$law/xml.zip"

  mkdir -p "$dest"
  echo "==> $law"

  # -z makes this a conditional request: unchanged files are not re-downloaded.
  if ! curl -fsS --max-time 180 -o "$dest/xml.zip" -z "$dest/xml.zip" "$url"; then
    echo "    FAILED: $url (check the abbreviation at $BASE_URL/Teilliste_$(echo "${law:0:1}" | tr '[:lower:]' '[:upper:]').html)" >&2
    return 1
  fi

  unzip -o -q "$dest/xml.zip" -d "$dest"

  xml="$(find "$dest" -maxdepth 1 -name '*.xml' -print -quit)"
  if [ -z "$xml" ]; then
    echo "    FAILED: archive contained no .xml" >&2
    return 1
  fi

  # builddate is the ministry's own stamp for when this revision was generated.
  builddate="$(sed -n 's/.*builddate="\([0-9]*\)".*/\1/p' "$xml" | head -1)"

  {
    echo "law=$law"
    echo "source_url=$url"
    echo "xml_file=$(basename "$xml")"
    echo "builddate=${builddate:-unknown}"
    echo "sha256=$(sha256 "$xml")"
    echo "fetched_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$dest/provenance.txt"

  echo "    $(basename "$xml")  builddate=${builddate:-unknown}"

  if [ "${WITH_PDF:-0}" = "1" ]; then
    pdf_name="$(echo "$law" | tr '[:lower:]' '[:upper:]').pdf"
    if curl -fsS --max-time 300 -o "$dest/$pdf_name" -z "$dest/$pdf_name" "$BASE_URL/$law/$pdf_name"; then
      echo "    $pdf_name"
    else
      echo "    note: no PDF at $BASE_URL/$law/$pdf_name (skipped)" >&2
    fi
  fi
}

failed=0
for law in "${LAWS[@]}"; do
  fetch_law "$law" || failed=1
done

echo
echo "Corpus in $RAW_DIR"
echo "Next: python src/parse_bgb.py --law bgb"
exit "$failed"
