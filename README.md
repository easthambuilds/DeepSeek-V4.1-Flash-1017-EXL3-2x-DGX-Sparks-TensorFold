<!-- DRAFT, not published. Results marked TBD are filled after the TP2 test and the quality runs. Replace <ORG> with
the publishing account. Publish only with operator approval. -->

# DeepSeek-V4.1-Flash-1017 EXL3 on 2× DGX Spark with TensorFold

Serve [**DeepSeek-V4.1-Flash-1017**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) from two NVIDIA DGX
Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports) through an OpenAI-compatible API, with **4 concurrent
requests**, the full **1,048,576-token context**, **images**, and DSpark drafting.

This repository is a set of scripts and pins, not an engine. The engine is **Bertholomus AI's `dsv41` family for
[TensorFold](https://github.com/ashhart/TensorFold)**, unchanged, served the way
[its own recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) runs it. What is ours:

- **The checkpoint:** [`<ORG>/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/<ORG>/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw).
  It is an EXL3 quant of the 1017 weights in Mia-AiLab's 2.9 bpw layout, made with published provenance
  ([model card](https://huggingface.co/<ORG>/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)).
- **One kit file:** the image routing bias from the 1017 weights instead of the base V4.1's.
- **The setup:** `start.sh` / `prepare.sh` set up both Sparks, checking every downloaded byte, then serve and
  smoke-test.

| | |
| --- | --- |
| Checkpoint | [`<ORG>/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/<ORG>/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), ~TBD GB, plus DeepSeek's original shards 47–48 (Engram tables, ~190 GB) |
| Engine | [`bertholomus/TensorFold`](https://github.com/bertholomus/TensorFold) `deepseek-v41-zig` @ `3ff3b2ac` (serves); `deepseek-v41-tp2` @ `46917151` (writes each rank's cache once) |
| Kit | [`bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10`](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10) `kit/` @ `789625a7`, with `vision/gate_bias_vl.safetensors` from the 1017 weights |
| Image | `nvcr.io/nvidia/pytorch:26.07-py3` + both engines (built by `scripts/prepare.sh`) |
| API model id | `DeepSeek-V4.1-Flash-1017-EXL3` |
| Defaults | `--context 1048576 --parallel 4 --thinking`, temperature 1.0, top-p 0.95 (DeepSeek's recommended sampling) |

## Results

TBD. Measured on two GB10 nodes at the defaults, with the engine's own client (`kit_bench.py`) and gates:

- **Release gates:** drafted == serial, concurrent == solo (burst and staggered), images.
- **Decode:** code / prose / structured, at 1 / 2 / 4 streams.
- **Prefill:** 8K–128K.
- **Quality:** KL divergence and top-1 agreement against DeepSeek's FP8 1017 release, beside Mia-AiLab's 2.9 bpw on
  the base V4.1 and LibertAIDAI's 1017 2.9 bpw ([model card](https://huggingface.co/<ORG>/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)).

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else large on their GPUs. Each
  needs ~110 GiB free when the server starts.
- **A direct ConnectX-7 link** with an IPv4 address on each end and a RoCE v2 GID. A cabled port shows up as two
  RoCE devices (e.g. `rocep1s0f1` / `roceP2p1s0f1`). Name both in `HCA`. Find the GID index with `show_gids`: 5 is
  the default, and some single-rail pairs use 3.
- **Key-based ssh** from the first Spark (the head, which runs the API) to the second (the worker).
- **Docker** with the NVIDIA runtime, plus `rsync` and `python3`, on both Sparks.
- **Disk, on each Spark:** ~520 GB under `DATA_DIR` (default `~/.cache/dsv41-1017-tf`). That is the checkpoint
  ~TBD GB, Engram shards ~190 GB, the rank cache ~106 GB and the kit ~2 GB. The image needs ~30 GB more under
  Docker's root.

## Quick start

On the head:

```bash
git clone https://github.com/<ORG>/DeepSeek-V4.1-Flash-1017-EXL3-2x-DGX-Sparks-TensorFold.git
cd DeepSeek-V4.1-Flash-1017-EXL3-2x-DGX-Sparks-TensorFold
cp scripts/local.sh.example scripts/local.sh   # WORKER, HEAD_IP, IFACE, HCA, GID_INDEX
./start.sh
```

The first run sets up both Sparks (below), which takes hours, mostly downloading and writing the rank caches. Later
starts only load. `./start.sh` ends with one request through both ranks and prints the endpoint:

```bash
curl -s http://127.0.0.1:18891/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-1017-EXL3",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 4000
}'

./start.sh restart        # restart both ranks (extra arguments go to tensorfold-native serve)
./stop.sh                 # stop both ranks; their logs are saved under DATA_DIR/logs
```

The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`. The server listens on
`127.0.0.1` by default. Set `HOST=0.0.0.0` together with `API_KEY_FILE` in `scripts/local.sh` to serve the network.

Images: send OpenAI-style `image_url` parts with `data:` URLs. The engine's
[README](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) has the formats and limits.

## What `scripts/prepare.sh` does

Each step skips work already done, so it is safe to re-run.

1. **Preflight on both Sparks:** ssh, the NVIDIA runtime, the RoCE devices and GID type, the route to the head, disk.
2. **Image:** builds `dsv41-1017-tf:<pin hash>` from `docker/Dockerfile`.
   - NVIDIA's PyTorch 26.07 container.
   - Zig 0.17.0, sha256-checked.
   - Both engine branches at their pinned commits.
   - The Zig engine built for sm_121.
   - The image is copied to the worker and its ID compared.
3. **Weights:** `tools/hub_fetch.py` downloads the checkpoint and DeepSeek's shards 47–48 at pinned revisions.
   - Every file is checked against the Hub: sha256 for large files, the git blob id for small ones.
   - Everything is copied to the worker with rsync, then checked there again.
4. **Kit:** downloads Bertholomus AI's kit at its pinned revision and checks it against its `MANIFEST` with the
   kit's own `verify_kit.sh`.
   - It then replaces `vision/gate_bias_vl.safetensors` with the 1017 checkpoint's `ffn.gate.bias_vl` tensors.
   - The new file is made by the kit's own `make_bias_vl.py`, run unchanged. Its input is a sparse stand-in for the
     original checkpoint holding only the bytes it reads (`tools/fetch_bias_vl.py`, 43 HTTP range reads, 66 KB,
     instead of 46 shards).
   - The result must match the pinned sha256 `4745da4b…`. The `MANIFEST` line is updated, `NOTICE-1017.txt` records
     the change, and the kit is verified again on both Sparks.
5. **Rank caches:** starts the Python engine once on both ranks. It writes each rank's weights (~106 GB) and the
   Engram token map, as the engine's `tools/dsv41-zig/build_inputs.sh` does, then stops it.

**Why the bias file:** 1017's image routing bias (`gate.bias_vl`) differs from the base V4.1's. The published kit's
copy is the base's (sha256 `0cbdebfb…`), while 1017's is `4745da4b…`.

## Configuration

`scripts/config.sh` holds the pins and defaults. Override any of them in `scripts/local.sh`: `CONTEXT`, `PARALLEL`,
`MAX_TOKENS`, `HOST`, `PORT`, `API_KEY_FILE`, `DATA_DIR`, `FABRIC_PEER` (copy the weights over the link), and
`WORKER_IFACE` / `WORKER_HCA` (when the worker's names differ).

## Credits

This recipe stands on other people's work. Each entry says what we used and whether we changed it.

- **[DeepSeek](https://huggingface.co/deepseek-ai):** [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  (MIT). It is the model, the Engram tables (shards 47–48, used as published) and the image routing bias in our kit
  file (copied byte for byte).
- **[Bertholomus AI](https://github.com/bertholomus) (Albert Lee):** the `dsv41` TensorFold family. This covers:
  - the [Zig engine](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig) that serves and the
    [Python engine](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2) that writes the rank caches;
  - the [kernel kit](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10) and its tools
    (`make_bias_vl.py`, `verify_kit.sh`, `build_inputs.sh`);
  - [the TP2 recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10), whose run commands,
    settings and release gates our scripts follow.

  Used unchanged, except that one kit file (`vision/gate_bias_vl.safetensors`) is replaced with 1017's and its
  `MANIFEST` line updated.
- **[ashhart](https://github.com/ashhart):** [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), the
  engine framework the `dsv41` family is built on, and the community gates in
  [#847](https://github.com/ashhart/TensorFold/issues/847).
- **[Mia-AiLab / Mia's AI Lab](https://huggingface.co/Mia-AiLab):**
  - The [V4.1 2.9 bpw quant](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) (MIT), whose
    per-tensor bit-width map and layout our 1017 checkpoint follows.
  - The [GLM-5.3-Flash TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
    whose layout this repository follows: `start.sh`, `scripts/prepare.sh`, `local.sh`, saved logs.
- **[turboderp](https://github.com/turboderp-org):** [exllamav3](https://github.com/turboderp-org/exllamav3) and the
  EXL3 format (MIT), used for the conversion and the quality measurement.
- **[anoane](https://github.com/anoane):** the DeepSeek-V4.1 port of exllamav3,
  [PR #449](https://github.com/turboderp-org/exllamav3/pull/449). We converted with it at `c341a8f5`, with four small
  patches published in the model card's provenance (two of them fixes reported on the PR).
- **NVIDIA:** the [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) the image is
  built on (NGC terms).
- **[Zig](https://ziglang.org):** the toolchain the engine is built with (MIT).
- **Third-party kernels in the kit:** CUTLASS, Triton, PyTorch and ExLlamaV3, under the licenses in the kit's
  `KIT-LICENSES.md`.
- **[LibertAIDAI](https://huggingface.co/LibertAIDAI):** the other 1017 EXL3 quant, measured beside ours for
  comparison only.

Not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Bertholomus AI or Mia-AiLab.

## License

The scripts and documentation in this repository are Apache-2.0 (`LICENSE`). The weights, engine and kit keep their
own licenses (above).
