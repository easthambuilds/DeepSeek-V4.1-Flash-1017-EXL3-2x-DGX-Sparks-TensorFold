#!/bin/bash
# apply_ablit.sh DATA_DIR IMAGE MARK SCRIPT SIDECAR: make DATA_DIR/model-ablit from DATA_DIR/model with the Keys overlay
# (ABLIT=1), on whichever Spark runs it. The overlay's own graft script, from DATA_DIR/ablit-src, runs unchanged in the
# serving image (CPU only, no network): it hardlinks the unchanged shards and rewrites the ones holding layers 10-35
# attn.wo_b. verify_ablit.py (next to this file) then checks every tensor of the copy byte for byte. MARK is recorded
# in model-ablit/.made-from, and a copy with the same MARK is kept as it is.
set -euo pipefail
D=$1 IMAGE=$2 MARK=$3 SCRIPT=$4 SIDECAR=$5
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$(cat "$D/model-ablit/.made-from" 2>/dev/null)" = "$MARK" ]; then
  echo "   model-ablit present ($MARK)"; exit 0
fi
rm -rf "$D/model-ablit"
docker run --rm --network none --user "$(id -u):$(id -g)" -v "$D:/d" --entrypoint python3 "$IMAGE" \
  "/d/ablit-src/$SCRIPT" --src /d/model --dst /d/model-ablit --wo-b "/d/ablit-src/$SIDECAR" > "$D/logs/ablit-graft.log"
python3 "$HERE/verify_ablit.py" "$D/model" "$D/model-ablit" "$D/ablit-src/$SIDECAR"
echo "$MARK" > "$D/model-ablit/.made-from"
