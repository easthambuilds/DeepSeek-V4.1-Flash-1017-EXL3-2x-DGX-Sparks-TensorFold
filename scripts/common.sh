# Shared by start.sh, stop.sh and scripts/prepare.sh. Sourced, not run.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$ROOT/scripts/config.sh"
[ -f "$ROOT/scripts/local.sh" ] && . "$ROOT/scripts/local.sh"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die() { printf '\033[31m!! %s\033[0m\n' "$*" >&2; exit 1; }

: "${WORKER:?set WORKER=user@host in scripts/local.sh (see scripts/local.sh.example)}"
[ -n "$HEAD_IP" ] || die "set HEAD_IP (the head's ConnectX-7 address) in scripts/local.sh"
[ -n "$IFACE" ] || die "set IFACE (the head's netdev on the link) in scripts/local.sh"
[ -n "$HCA" ] || die "set HCA (the link's RoCE devices) in scripts/local.sh"
WORKER_IFACE=${WORKER_IFACE:-$IFACE}
WORKER_HCA=${WORKER_HCA:-$HCA}

on_worker() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }

# The worker keeps DATA_DIR at the same place relative to its own home
if [ -z "${WORKER_DATA_DIR:-}" ]; then
  case "$DATA_DIR" in
    "$HOME"/*) WORKER_DATA_DIR="$(on_worker 'echo $HOME')${DATA_DIR#"$HOME"}" ;;
    *) WORKER_DATA_DIR=$DATA_DIR ;;
  esac
fi

# The image tag follows everything that goes into it
PIN_HASH=$(printf '%s\n' "$BASE_IMAGE" "$ENGINE_GIT" "$ZIG_ENGINE_REV" "$PY_ENGINE_REV" "$ZIG_URL" "$ZIG_SHA256" \
  "$(cat "$ROOT/docker/Dockerfile")" | sha256sum | cut -c1-12)
IMAGE=${IMAGE:-dsv41-1017-tf:$PIN_HASH}

# docker run options both ranks share; $1 data dir, $2 netdev, $3 RoCE devices
run_opts() {
  echo "--gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK" \
    "--device /dev/infiniband -v $1/model:/model:ro -v $1/engram:/engram:ro -v $1/kit:/kit:ro -v $1/cache:/cache" \
    "-e NCCL_IB_HCA=$3 -e NCCL_IB_GID_INDEX=$GID_INDEX -e NCCL_SOCKET_IFNAME=$2 -e TF_RDMA_GID_INDEX=$GID_INDEX"
}

# The Python engine's first start writes the rank cache and token map (as the engine's tools/dsv41-zig/build_inputs.sh)
inputs_cmd() {  # $1 rank, $2 data dir, $3 netdev, $4 RoCE devices
  # the engine writes its token map and extensions under ~/.cache, so the cache is also its home's .cache
  echo "docker run -d --name $CONTAINER-inputs $(run_opts "$2" "$3" "$4") -v $2/cache:/root/.cache -e HOME=/root" \
    "-e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e TF_DS_REPLAY=1 -e TF_DS_PREFILL_CHUNK=2048" \
    "-e TF_DS_RANK_CACHE=/root/.cache/tf-ds-rank -e TF_DS_RANK_CACHE_READERS=32 -e TF_DS_ENGRAM=/engram" \
    "--entrypoint python3 $IMAGE -m tensorfold serve /model --tp 2 --rank $1 --master $HEAD_IP --host 127.0.0.1" \
    "--port $PORT --context 1048576 --vision --parallel 4 --mtp-drafts 5"
}

# Serving: rank 1 follows rank 0 (the engine's README, v0.6.x)
rank1_cmd() {
  echo "docker run -d --name $CONTAINER $(run_opts "$WORKER_DATA_DIR" "$WORKER_IFACE" "$WORKER_HCA") -e TF_DS_REPLAY_FLOOR=1" \
    "--entrypoint /opt/tf/bin/tf-dsv41-lanes $IMAGE /model /cache/tf-ds-rank 1 2 $HEAD_IP $TP_PORT /kit" \
    "--engram /engram --token-map /cache/dsv41_token_map.json --pool $CONTEXT --rdma $WORKER_HCA" \
    "--graphs 1 --side 1 --prefetch 1 --aio 1"
}

rank0_cmd() {  # extra arguments go to tensorfold-native serve
  local key="" keyarg=""
  if [ -n "$API_KEY_FILE" ]; then key="-v $API_KEY_FILE:/run/api-key:ro"; keyarg="--api-key-file /run/api-key"; fi
  echo "docker run -d --name $CONTAINER $(run_opts "$DATA_DIR" "$IFACE" "$HCA") $key -e TF_DS_REPLAY_FLOOR=1" \
    "-e TF_TP_WORLD=2 -e TF_TP_MASTER=$HEAD_IP -e TF_TP_PORT=$TP_PORT -e TF_DS_RANK_CACHE=/cache/tf-ds-rank" \
    "-e TF_DS_KIT=/kit -e TENSORFOLD_CUDA_KERNELS=/kit/aot -e TF_DS_ENGRAM=/engram" \
    "-e TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json -e TF_RDMA_DEVICES=$HCA -e TF_DS_GRAPHS=1 -e TF_DS_HC_SIDE=1" \
    "-e TF_DS_L2_PREFETCH=1 -e TF_DS_ENGRAM_AIO=1 --entrypoint /opt/tf/native/bin/tensorfold-native $IMAGE" \
    "serve /model --host $HOST --port $PORT --name $MODEL_NAME $keyarg --context $CONTEXT --parallel $PARALLEL" \
    "--thinking --max-tokens $MAX_TOKENS --temperature 1.0 --top-p 0.95 --top-k 20 --backend cuda $*"
}

running_here() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
running_there() { on_worker "docker ps --format '{{.Names}}'" | grep -qx "$1"; }

# Saves a container's log (gzipped, timestamped) before it is removed; $1 name, $2 "here" or "there"
save_log() {
  local name=$1 where=$2 stamp; stamp=$(date +%Y%m%d-%H%M%S)
  if [ "$where" = here ]; then
    docker inspect "$name" >/dev/null 2>&1 || return 0
    mkdir -p "$DATA_DIR/logs"
    docker logs -t "$name" 2>&1 | gzip > "$DATA_DIR/logs/$stamp-$name-rank0.log.gz"
    note "rank 0 log: $DATA_DIR/logs/$stamp-$name-rank0.log.gz"
  else
    on_worker "docker inspect $name >/dev/null 2>&1 || exit 0; mkdir -p $WORKER_DATA_DIR/logs; \
      docker logs -t $name 2>&1 | gzip > $WORKER_DATA_DIR/logs/$stamp-$name-rank1.log.gz; \
      echo '   rank 1 log: $WORKER_DATA_DIR/logs/$stamp-$name-rank1.log.gz'"
  fi
}

remove_both() {  # $1 container name
  save_log "$1" here; docker rm -f "$1" >/dev/null 2>&1 || true
  save_log "$1" there; on_worker "docker rm -f $1 >/dev/null 2>&1 || true"
}

api() {  # curl the head's API; $1 path, rest to curl
  local p=$1; shift
  local h=$HOST; [ "$h" = 0.0.0.0 ] && h=127.0.0.1
  local auth=(); [ -n "$API_KEY_FILE" ] && auth=(-H "Authorization: Bearer $(cat "$API_KEY_FILE")")
  curl -s "${auth[@]}" "http://$h:$PORT$p" "$@"
}
