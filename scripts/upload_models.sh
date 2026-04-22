#!/bin/bash
set -euo pipefail
repo_id=$1
start=1
end=10
step=1

CKPT_ROOT="models/$repo_id"

hf repo create "$repo_id"

echo "📦 Uploading top-level files → revision: main"
hf upload "$repo_id" $CKPT_ROOT/ \
  --repo-type model \
  --revision main \
  --exclude "epoch-*"

for i in $(seq "$start" "$step" "$end"); do
  src="$CKPT_ROOT/epoch-$i"
  if [ ! -d "$src" ]; then
    echo "⚠️  $src does not exist! "
    continue
  fi

  revision="epoch-$i"
  echo "📦 uploaded $src to revision: $revision"

  hf upload "$repo_id" "$src" \
    --repo-type model \
    --revision "$revision" \
    --commit-message "Add checkpoint $i as revision $revision"
done


echo "✅all checkpoints have been uploaded to revision。"
