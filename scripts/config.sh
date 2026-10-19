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
# The _topk_sel N=KK=256 variant (prompts of exactly 512 or 513 tokens), which the pinned kit predates: aot/aot.json
# and one cubin from the kit in git in Bertholomus AI's TP2 recipe (#13)
KIT_FIX_REPO=bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10
KIT_FIX_REV=58186d120bfa4a7f19cc0da03b53f64de2685239
AOT_JSON_BASE_SHA256=092b6d274a5f86b5fd69fa3cd04703686e4e998ceb3ecc997be8359c89b841b0
AOT_JSON_SHA256=29dccbcfdfc327a2ca83f40eb88c2bad83f9f61128e0cd27c29a1243d6fcd603
TOPK256_CUBIN=aot/cubins/a2c0ea5eb9a9c7aa6028441066bea0a479431badcf841a4892fe735ac8bb805a.cubin
TOPK256_CUBIN_SHA256=b3bbb9b0d874a8875c9a6e18975588aa547d03db6aeb614c155324cb5bf76f8f

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
# Optional, for a tight memory budget (unset: the engine's defaults)
TF_MEMORY_RESERVE_GIB=${TF_MEMORY_RESERVE_GIB:-}  # rank 0's CUDA reserve in GiB, >= 2 (default: a tenth
                                                 # of the memory, ~12.2 GiB on GB10)
INPUTS_CONTEXT=${INPUTS_CONTEXT:-1048576}        # window and streams of step 5's one Python start; the rank caches
INPUTS_PARALLEL=${INPUTS_PARALLEL:-4}            # do not depend on them (their key is the checkpoint and its config)
SERVE_EXTRA=${SERVE_EXTRA:-}                      # more tensorfold-native serve arguments, e.g. --prompt-cache-gib 1
TP_PORT=29620
CONTAINER=dsv41-1017-tf
