#!/usr/bin/env bash
# Seed R2 bucket "eom-corpus" with the gold doc set.
#
# Layout in R2:
#   manifest.json                 -- registry of all docs
#   <type>/<slug>.md              -- raw markdown
#   <type>/<slug>.eom.json        -- compiled EOM
#
# Idempotent: each `wrangler r2 object put` overwrites if present.
#
# Run with:
#   CLOUDFLARE_ACCOUNT_ID=1e0170aaabc90ecf5f466128d1f0466a ./scripts/seed-r2.sh

set -euo pipefail
cd "$(dirname "$0")/.."

BUCKET=eom-corpus
PUBLIC=./public/samples

if [ ! -d "$PUBLIC" ]; then
  echo "ERR: $PUBLIC missing; run \`bun run scripts/sync.ts\` first" >&2
  exit 1
fi

# Manifest first so consumers can list.
echo "uploading manifest…"
bunx wrangler r2 object put "$BUCKET/manifest.json" \
  --file="$PUBLIC/manifest.json" --content-type=application/json --remote 1>/dev/null

count=0
for type_dir in "$PUBLIC"/*/; do
  type=$(basename "$type_dir")
  [ "$type" = "." ] || [ "$type" = ".." ] && continue
  for md in "$type_dir"*.md; do
    [ -f "$md" ] || continue
    slug=$(basename "$md" .md)
    eom="$type_dir$slug.eom.json"
    [ -f "$eom" ] || { echo "  skip $type/$slug (no .eom.json)"; continue; }
    bunx wrangler r2 object put "$BUCKET/$type/$slug.md" \
      --file="$md" --content-type=text/markdown --remote 1>/dev/null
    bunx wrangler r2 object put "$BUCKET/$type/$slug.eom.json" \
      --file="$eom" --content-type=application/json --remote 1>/dev/null
    count=$((count + 1))
    printf "  %-30s\r" "$type/$slug"
  done
done
echo
echo "uploaded $count docs (each = .md + .eom.json) to r2://$BUCKET"
