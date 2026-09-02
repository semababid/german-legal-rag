#!/usr/bin/env bash
# Copy everything the VM needs, over ssh.
#
#   ./deploy/oracle/upload.sh ubuntu@<your-instance-ip>
#
# Sends the source, the page, the corpus and the prebuilt index, and builds the
# image on the far side. The index is copied rather than built there because two
# Arm cores would spend hours embedding 4,921 records; on this machine it is
# already done.
#
# Run it again after any change. It only re-sends what differs.

set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "usage: $0 user@host    (Oracle Ubuntu images use the user 'ubuntu')" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COLLECTION="${COLLECTION:-absatz_ctx}"
REMOTE="~/german-legal-rag"

if [ ! -d "$ROOT/qdrant_storage/collection/$COLLECTION" ]; then
  echo "No index at qdrant_storage/collection/$COLLECTION." >&2
  echo "Build it here first: python src/index_corpus.py --granularity absatz --with-context" >&2
  exit 1
fi

if ! grep -q "duckdns\|sslip\|nip.io\|\." "$ROOT/deploy/oracle/Caddyfile" 2>/dev/null; then
  echo "Set your hostname in deploy/oracle/Caddyfile first." >&2
  exit 1
fi
if grep -q "^mietrecht-rag.duckdns.org" "$ROOT/deploy/oracle/Caddyfile"; then
  echo "Caddyfile still has the example hostname. Edit it to your own, or Caddy" >&2
  echo "will try to get a certificate for a name that is not yours." >&2
  exit 1
fi

echo "creating $REMOTE on $TARGET"
ssh "$TARGET" "mkdir -p $REMOTE/{src,web,data/processed,qdrant_storage/collection}"

echo "sending source and page"
rsync -az --delete "$ROOT/src/" "$TARGET:$REMOTE/src/"
rsync -az --delete "$ROOT/web/" "$TARGET:$REMOTE/web/"
rsync -az "$ROOT/requirements.txt" "$ROOT/Dockerfile" "$TARGET:$REMOTE/"
rsync -az "$ROOT/deploy/oracle/docker-compose.yml" "$ROOT/deploy/oracle/Caddyfile" "$TARGET:$REMOTE/"

echo "sending corpus"
rsync -az "$ROOT/data/processed/bgb.absatz.jsonl" "$TARGET:$REMOTE/data/processed/"

# Only the collection actually served. The other two are phase 3 experiments.
echo "sending the index ($(du -sh "$ROOT/qdrant_storage/collection/$COLLECTION" | cut -f1))"
rsync -az "$ROOT/qdrant_storage/meta.json" "$TARGET:$REMOTE/qdrant_storage/"
rsync -az --delete "$ROOT/qdrant_storage/collection/$COLLECTION/" \
  "$TARGET:$REMOTE/qdrant_storage/collection/$COLLECTION/"

echo
echo "building on the VM (first build takes a while: it installs torch)"
ssh "$TARGET" "cd $REMOTE && docker build -t german-legal-rag:latest . && docker compose up -d"

echo
echo "started. Watch it come up with:"
echo "  ssh $TARGET 'cd $REMOTE && docker compose logs -f'"
echo
echo "The first start downloads bge-m3 (about 2.3 GB) into a volume, so give it"
echo "a few minutes before the health check passes."
