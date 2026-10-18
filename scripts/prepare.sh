#!/bin/bash
# prepare.sh: the one-time setup of both Sparks, run on the head. Safe to re-run: each step skips what is done.
#   1. preflight on both Sparks   2. the image   3. weights   4. the 1017 kit   5. each rank's cache
. "$(dirname "$0")/common.sh"
TOOLS=$ROOT/tools
COPY_TO=$WORKER   # rsync target for the weights: the link address when FABRIC_PEER is set
if [ -n "${FABRIC_PEER:-}" ]; then
  case "$WORKER" in *@*) COPY_TO=${WORKER%%@*}@$FABRIC_PEER ;; *) COPY_TO=$FABRIC_PEER ;; esac
fi
filter() { grep -v '^   have' || true; }

step "1/5 Preflight"
on_worker true || die "no key-based ssh to $WORKER (ssh-copy-id $WORKER)"
for side in here there; do
  run() { if [ $side = here ]; then bash -c "$1"; else on_worker "$1"; fi; }
  hca=$HCA; dd=$DATA_DIR; [ $side = there ] && hca=$WORKER_HCA dd=$WORKER_DATA_DIR
  run "docker info --format '{{.Runtimes}}' | grep -q nvidia" || die "$side: Docker without the NVIDIA runtime"
  run "command -v rsync >/dev/null && command -v python3 >/dev/null" || die "$side: needs rsync and python3"
  for d in ${hca//,/ }; do
    run "test -d /sys/class/infiniband/$d" || die "$side: no RoCE device $d"
    t=$(run "cat /sys/class/infiniband/$d/ports/1/gid_attrs/types/$GID_INDEX 2>/dev/null" || true)
    [ "$t" = "RoCE v2" ] || die "$side: $d GID index $GID_INDEX is '${t:-none}', not RoCE v2 (set GID_INDEX)"
  done
  free=$(run "mkdir -p $dd && df -BG --output=avail $dd | tail -1 | tr -dc 0-9")
  note "$side: NVIDIA runtime, RoCE ($hca, GID $GID_INDEX), ${free} GB free for the data"
done
on_worker "ping -c 1 -W 2 $HEAD_IP >/dev/null" || die "the worker cannot reach HEAD_IP $HEAD_IP"
note "worker reaches the head at $HEAD_IP"
mkdir -p "$DATA_DIR"/{model,engram,kit,cache,logs}
on_worker "mkdir -p $WORKER_DATA_DIR/{model,engram,kit,cache,logs,tools}"

step "2/5 Image $IMAGE"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  note "present on the head"
else
  note "building from $BASE_IMAGE (the engine build takes a while)"
  docker build -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg ENGINE_GIT="$ENGINE_GIT" \
    --build-arg ZIG_ENGINE_REV="$ZIG_ENGINE_REV" --build-arg PY_ENGINE_REV="$PY_ENGINE_REV" \
    --build-arg ZIG_URL="$ZIG_URL" --build-arg ZIG_SHA256="$ZIG_SHA256" "$ROOT/docker"
fi
id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
if [ "$(on_worker "docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null" || true)" != "$id" ]; then
  note "copying the image to the worker"
  docker save "$IMAGE" | on_worker "docker load" >/dev/null
fi
[ "$(on_worker "docker image inspect -f '{{.Id}}' $IMAGE")" = "$id" ] || die "image differs on the worker"
note "same image on both: ${id:7:12}"

step "3/5 Weights"
python3 "$TOOLS/hub_fetch.py" "$QUANT_REPO" "$QUANT_REV" "$DATA_DIR/model" | filter
python3 "$TOOLS/hub_fetch.py" "$BASE_REPO" "$BASE_REV" "$DATA_DIR/engram" $ENGRAM_FILES | filter
note "copying to the worker (rsync, resumable)"
rsync -a --partial --info=progress2 "$DATA_DIR/model/" "$COPY_TO:$WORKER_DATA_DIR/model/"
rsync -a --partial --info=progress2 "$DATA_DIR/engram/" "$COPY_TO:$WORKER_DATA_DIR/engram/"
rsync -a "$TOOLS/" "$WORKER:$WORKER_DATA_DIR/tools/"
note "checking the worker's copy against the Hub"
on_worker "python3 $WORKER_DATA_DIR/tools/hub_fetch.py $QUANT_REPO $QUANT_REV $WORKER_DATA_DIR/model | tail -1 && \
  python3 $WORKER_DATA_DIR/tools/hub_fetch.py $BASE_REPO $BASE_REV $WORKER_DATA_DIR/engram $ENGRAM_FILES | tail -1"

step "4/5 Kernel kit, with the 1017 image routing bias"
K=$DATA_DIR/kit
if [ -f "$K/NOTICE-1017.txt" ] && bash "$K/tools/verify_kit.sh" >/dev/null; then
  note "kit-1017 present and verified"
else
  python3 "$TOOLS/hub_fetch.py" "$KIT_REPO" "$KIT_REV" "$DATA_DIR/kit-src" "kit/*" | filter
  rm -rf "$K" && cp -a "$DATA_DIR/kit-src/kit" "$K"
  bash "$K/tools/verify_kit.sh" || die "the downloaded kit does not match its MANIFEST"
  grep -q "^$BIAS_VL_BASE_SHA256  vision/gate_bias_vl.safetensors$" "$K/MANIFEST" || die "unexpected kit MANIFEST"
  python3 "$TOOLS/fetch_bias_vl.py" "$BASE_REPO" "$BASE_REV" "$DATA_DIR/bias-src"
  python3 -I "$K/tools/make_bias_vl.py" "$DATA_DIR/bias-src" "$K/vision/gate_bias_vl.safetensors"
  [ "$(sha256sum "$K/vision/gate_bias_vl.safetensors" | cut -d' ' -f1)" = "$BIAS_VL_SHA256" ] \
    || die "the 1017 bias_vl does not match the pinned sha256"
  sed -i "s/^$BIAS_VL_BASE_SHA256  vision\/gate_bias_vl.safetensors$/$BIAS_VL_SHA256  vision\/gate_bias_vl.safetensors/" "$K/MANIFEST"
  echo "1017 kit: $KIT_REPO at ${KIT_REV:0:8} with vision/gate_bias_vl.safetensors made by tools/make_bias_vl.py from" \
    "$BASE_REPO at ${BASE_REV:0:8} (sha256 $BIAS_VL_SHA256); MANIFEST updated for that one file." > "$K/NOTICE-1017.txt"
  bash "$K/tools/verify_kit.sh" || die "kit-1017 does not verify"
  rm -rf "$DATA_DIR/bias-src"
fi
rsync -a --delete "$K/" "$WORKER:$WORKER_DATA_DIR/kit/"
on_worker "bash $WORKER_DATA_DIR/kit/tools/verify_kit.sh" >/dev/null || die "the worker's kit does not verify"
note "kit-1017 verified on both"

step "5/5 Rank caches and the Engram token map (first start of the Python engine)"
have() { test -f "$1/cache/dsv41_token_map.json" && test -n "$(ls -A "$1/cache/tf-ds-rank" 2>/dev/null)"; }
stamp="$QUANT_REPO@$QUANT_REV $(cat "$DATA_DIR/model/model.safetensors.index.json" | sha256sum | cut -c1-16)"
if have "$DATA_DIR" && on_worker "$(declare -f have); have $WORKER_DATA_DIR" \
   && [ "$(cat "$DATA_DIR/cache/.made-from" 2>/dev/null)" = "$stamp" ]; then
  note "present on both"
else
  rm -rf "$DATA_DIR/cache"/{tf-ds-rank,dsv41_token_map.json}
  on_worker "rm -rf $WORKER_DATA_DIR/cache/{tf-ds-rank,dsv41_token_map.json}"
  remove_both "$CONTAINER-inputs" >/dev/null
  eval "$(inputs_cmd 0 "$DATA_DIR" "$IFACE" "$HCA")" >/dev/null
  # rank 0 owns the TCPStore: rank 1 starts once its port listens (a fixed sleep races on a cold start)
  for i in $(seq 120); do ss -ltn | grep -q ':29551 ' && break; sleep 2; done
  on_worker "$(inputs_cmd 1 "$WORKER_DATA_DIR" "$WORKER_IFACE" "$WORKER_HCA")" >/dev/null
  note "both ranks writing their caches (~106 GB each); this takes a while"
  t0=$(date +%s)
  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models)" = 200 ]; do
    running_here "$CONTAINER-inputs" || { docker logs --tail 30 "$CONTAINER-inputs"; die "rank 0 stopped"; }
    running_there "$CONTAINER-inputs" || { on_worker "docker logs --tail 30 $CONTAINER-inputs"; die "rank 1 stopped"; }
    sleep 30; note "$(( ($(date +%s) - t0) / 60 )) min, rank 0: $(du -sh "$DATA_DIR/cache/tf-ds-rank" 2>/dev/null | cut -f1)"
  done
  remove_both "$CONTAINER-inputs" >/dev/null
  have "$DATA_DIR" && on_worker "$(declare -f have); have $WORKER_DATA_DIR" || die "rank caches missing after the first start"
  echo "$stamp" > "$DATA_DIR/cache/.made-from"
  note "done in $(( ($(date +%s) - t0) / 60 )) min"
fi
echo "$IMAGE $stamp" > "$DATA_DIR/.prepared"
step "Setup ready on both Sparks"
