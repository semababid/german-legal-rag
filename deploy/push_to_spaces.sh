#!/usr/bin/env bash
# Assemble and push the Hugging Face Space.
#
# The Space is a separate git repository holding a subset of this one: source,
# page, corpus, prebuilt index, and its own Dockerfile and README. It is
# assembled in a scratch directory rather than pushed from here, so the Space
# history stays clean and nothing in this repository has to be arranged to suit
# the platform.
#
#   ./deploy/push_to_spaces.sh <your-hf-username>/<space-name>
#
# Needs the huggingface CLI and a login:
#   pip install -U "huggingface_hub[cli]" && hf auth login
#
# The index has to exist first. Build it on this machine, not in the Space:
#   python src/index_corpus.py --granularity absatz --with-context

set -euo pipefail

SPACE="${1:-}"
if [ -z "$SPACE" ]; then
  echo "usage: $0 <username>/<space-name>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTION="${COLLECTION:-absatz_ctx}"

if [ ! -d "$ROOT/qdrant_storage/collection/$COLLECTION" ]; then
  echo "No index at qdrant_storage/collection/$COLLECTION." >&2
  echo "Build it first: python src/index_corpus.py --granularity absatz --with-context" >&2
  exit 1
fi

command -v hf >/dev/null 2>&1 || {
  echo "The huggingface CLI is not installed." >&2
  echo "  pip install -U \"huggingface_hub[cli]\" && hf auth login" >&2
  exit 1
}

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
echo "staging in $STAGE"

git clone "https://huggingface.co/spaces/$SPACE" "$STAGE/space" 2>/dev/null || {
  echo "Could not clone the Space. Create it first at" >&2
  echo "  https://huggingface.co/new-space   (SDK: Docker)" >&2
  exit 1
}

cd "$STAGE/space"
# Replace the tracked content wholesale, so a file deleted here is deleted there
# rather than lingering in the Space from an earlier push.
git rm -rq --ignore-unmatch . 2>/dev/null || true

cp "$ROOT/deploy/Dockerfile" Dockerfile
cp "$ROOT/deploy/README.md" README.md
cp "$ROOT/requirements.txt" requirements.txt
mkdir -p src web data/processed qdrant_storage/collection
cp "$ROOT"/src/*.py src/
cp "$ROOT"/web/index.html web/
cp "$ROOT"/data/processed/bgb.absatz.jsonl data/processed/

# Only the collection actually served. The other two are phase 3 experiments and
# would double the upload for nothing.
cp "$ROOT/qdrant_storage/meta.json" qdrant_storage/
cp -R "$ROOT/qdrant_storage/collection/$COLLECTION" "qdrant_storage/collection/$COLLECTION"

cat > .gitattributes <<'EOF'
*.jsonl filter=lfs diff=lfs merge=lfs -text
qdrant_storage/** filter=lfs diff=lfs merge=lfs -text
EOF

echo
echo "pushing $(du -sh . | cut -f1) to $SPACE"
git add -A
git commit -qm "Deploy retrieval over German tenancy law" || {
  echo "nothing changed since the last push"; exit 0; }
git push
echo
echo "done: https://huggingface.co/spaces/$SPACE"
echo "First build takes a while: it installs torch and downloads bge-m3 (~2.3 GB)."
