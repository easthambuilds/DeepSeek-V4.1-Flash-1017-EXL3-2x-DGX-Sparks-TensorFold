# The serving image: NVIDIA's PyTorch container, with Bertholomus AI's dsv41 engines for TensorFold at pinned commits.
#   /opt/tf-py   the deepseek-v41-tp2 Python engine (writes each rank's cache and the Engram token map once)
#   /opt/tf      the deepseek-v41-zig engine, built for GB10 (sm_121): native/bin/tensorfold-native, bin/tf-dsv41-lanes
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG ENGINE_GIT
ARG ZIG_ENGINE_REV
ARG PY_ENGINE_REV
ARG ZIG_URL
ARG ZIG_SHA256

RUN curl -fsSL -o /tmp/zig.tar.xz "$ZIG_URL" \
 && echo "$ZIG_SHA256  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components 1 && rm /tmp/zig.tar.xz

RUN git clone -q "$ENGINE_GIT" /opt/tf-py && git -C /opt/tf-py checkout -q "$PY_ENGINE_REV" \
 && pip install --no-cache-dir -e /opt/tf-py

RUN git clone -q "$ENGINE_GIT" /opt/tf-src && git -C /opt/tf-src checkout -q "$ZIG_ENGINE_REV" \
 && cd /opt/tf-src && ZIG_GLOBAL_CACHE_DIR=/tmp/zig-cache ZIG_LOCAL_CACHE_DIR=/tmp/zig-cache-local \
    /opt/zig/zig build install native -Dtarget=aarch64-linux-gnu -Doptimize=ReleaseSafe \
      -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /opt/tf \
 && rm -rf /tmp/zig-cache /tmp/zig-cache-local \
 && test -x /opt/tf/native/bin/tensorfold-native && test -x /opt/tf/bin/tf-dsv41-lanes

LABEL org.opencontainers.image.title="dsv41-1017-tf" \
      dsv41.zig_engine="$ZIG_ENGINE_REV" dsv41.py_engine="$PY_ENGINE_REV"
