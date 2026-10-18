#!/bin/bash
# start.sh [restart] [tensorfold-native serve arguments...]: set up both Sparks if needed, start rank 1 on the worker
# and rank 0 (the API) here, wait until they load, and send one smoke-test request through both.
. "$(dirname "$0")/scripts/common.sh"
RESTART=0
[ "${1:-}" = restart ] && { RESTART=1; shift; }

step "Setup"
if [ "$(cat "$DATA_DIR/.prepared" 2>/dev/null | cut -d' ' -f1)" = "$IMAGE" ]; then
  note "ready ($DATA_DIR/.prepared)"
else
  bash "$ROOT/scripts/prepare.sh"
fi

step "Checks"
if running_here "$CONTAINER" || running_there "$CONTAINER"; then
  [ $RESTART = 1 ] || die "already running: ./start.sh restart, or ./stop.sh"
  note "stopping the running server (requests in flight are cut off)"
  remove_both "$CONTAINER"
else
  remove_both "$CONTAINER" >/dev/null   # a stopped container from an earlier run; its log is kept
fi
# GB10 shares memory between GPU and CPU and reclaims page cache too slowly for CUDA (tools/evict_page_cache.py)
note "here: $(python3 "$ROOT/tools/evict_page_cache.py" "$DATA_DIR"/{model,engram,cache})"
note "$WORKER: $(on_worker python3 - "$WORKER_DATA_DIR"/{model,engram,cache} < "$ROOT/tools/evict_page_cache.py")"
for side in here there; do
  if [ $side = here ]; then avail=$(free -g | awk '/^Mem:/{print $7}'); else avail=$(on_worker "free -g" | awk '/^Mem:/{print $7}'); fi
  [ "$avail" -ge 110 ] || note "warning: only $avail GiB memory available $side (the engine wants ~110): stop other GPU work"
done
ss -ltn | grep -q ":$PORT " && die "port $PORT is in use here"

step "Launch"
on_worker "$(rank1_cmd)" >/dev/null
note "rank 1 started on $WORKER"
eval "$(rank0_cmd "$@")" >/dev/null
note "rank 0 started here; loading"

step "Loading"
t0=$(date +%s)
until [ "$(api /v1/models -o /dev/null -w '%{http_code}')" = 200 ]; do
  if ! running_here "$CONTAINER" || ! running_there "$CONTAINER"; then
    echo "-- rank 0:"; docker logs --tail 25 "$CONTAINER" 2>&1
    echo "-- rank 1:"; on_worker "docker logs --tail 25 $CONTAINER" 2>&1
    remove_both "$CONTAINER"
    die "a rank stopped while loading (logs above)"
  fi
  [ $(( $(date +%s) - t0 )) -lt 3600 ] || die "not up after an hour: docker logs $CONTAINER"
  sleep 15
  note "$(( $(date +%s) - t0 )) s: $(docker logs --tail 1 "$CONTAINER" 2>&1 | cut -c1-110)"
done
note "up after $(( $(date +%s) - t0 )) s"

step "Smoke test"
reply=$(api /v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\": \"$MODEL_NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 times 3? Answer with the number only.\"}],
  \"max_tokens\": 2000}")
echo "$reply" | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('   reply:', (m.get('content') or '').strip()[:200])" \
  || die "the smoke test failed: $reply"
echo "$reply" | grep -q '51' || note "warning: the reply does not contain 51"

h=$HOST; [ "$h" = 0.0.0.0 ] && h=$(hostname -I | awk '{print $1}')
step "$MODEL_NAME is live on http://$h:$PORT/v1"
