# Pins and defaults. Put your own settings in scripts/local.sh (read after this file), not here.

# The checkpoint this recipe serves, and the original weights it needs alongside
QUANT_REPO=${QUANT_REPO:-easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}
QUANT_REV=${QUANT_REV:-2913ea75b326801b6645bc2ffaefb485d049abc2}
BASE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
BASE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
ENGRAM_FILES="model-00047-of-00048.safetensors model-00048-of-00048.safetensors"

# Engine: Bertholomus AI's dsv41 family for TensorFold (Zig engine serves, Python engine writes the rank caches)
ENGINE_GIT=https://github.com/bertholomus/TensorFold.git
ZIG_ENGINE_REV=3ff3b2ac1d71a789cb930068c46d9abed025eefa   # branch deepseek-v41-zig
PY_ENGINE_REV=469171515cd3c0df14346d83d763b4e89d4bafcf    # branch deepseek-v41-tp2
ZIG_URL=https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}

# Kernel kit: Bertholomus AI's, with the image routing bias taken from the 1017 weights
KIT_REPO=bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10
KIT_REV=789625a7081e91643cb61ceafd15f588850c511a
BIAS_VL_BASE_SHA256=0cbdebfbffed15722f096d6b8290933135d4393d0920059f7bcad4789f19387b
BIAS_VL_SHA256=4745da4b5f24c67170ee24e2c95652e45ba13c0ca0abc2be3cc7a2a3545792d0

# Optional: the Keys abliteration overlay (README, "Optional: the abliteration overlay"). ABLIT=0 (the default) serves
# the checkpoint as published. ABLIT=1 serves a separate copy whose attn.wo_b in layers 10-35 is the overlay's; it needs
# an HF token with access to the gated repo. Each setting has its own model directory, rank caches and setup marker,
# so switching never touches the other's files.
ABLIT=${ABLIT:-0}
ABLIT_ID=drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated
ABLIT_REVISION=${ABLIT_REVISION:-0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0}
ABLIT_SCRIPT=apply_mia29_wo_b_graft.py       # the overlay's own graft script, run unchanged in the image
ABLIT_SIDECAR=exl3_wo_b_l10_35_mia29.safetensors   # EXL3 K=5 for Mia-AiLab's 2.9 bpw layout, which ours shares
ABLIT_SIDECAR_SHA256=dcefb4a735f4985e4a5097cad69a059d85453f3bba148af3fc98269c7ee1ff7c

# Where everything lives on each Spark (~520 GB: checkpoint ~200, Engram shards ~190, rank cache ~106, kit ~2)
DATA_DIR=${DATA_DIR:-$HOME/.cache/dsv41-1017-tf}

# The link between the Sparks (see README, Requirements)
HEAD_IP=${HEAD_IP:-}          # the head's address on the ConnectX-7 link; rank 1 connects to it
IFACE=${IFACE:-}              # the head's netdev on that link, e.g. enp1s0f1np1
HCA=${HCA:-}                  # the RoCE devices of the link, comma-separated, e.g. rocep1s0f1,roceP2p1s0f1
GID_INDEX=${GID_INDEX:-5}     # RoCE v2 GID index (show_gids); 3 on some single-rail pairs
WORKER_IFACE=${WORKER_IFACE:-}  # when the worker's names differ from the head's
WORKER_HCA=${WORKER_HCA:-}

# Server
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18891}
MODEL_NAME=${MODEL_NAME:-DeepSeek-V4.1-Flash-1017-EXL3}
CONTEXT=${CONTEXT:-1048576}
PARALLEL=${PARALLEL:-4}
MAX_TOKENS=${MAX_TOKENS:-32768}
API_KEY_FILE=${API_KEY_FILE:-}  # a file holding the API key; every route but /health then needs it
TP_PORT=29620
CONTAINER=dsv41-1017-tf
