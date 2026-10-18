"""
fetch_bias_vl.py REPO REV DIR: a sparse stand-in for the original checkpoint that holds only the bytes the kit's
tools/make_bias_vl.py reads, the 43 `*.ffn.gate.bias_vl` tensors (about 66 KB), without downloading the 46 shards
they sit in. Standard library only.

DIR gets the checkpoint's model.safetensors.index.json and, for each shard that holds a bias_vl tensor, a sparse file
of that shard's full size with its real header and the real bytes of those tensors at their real offsets, all read
from the Hub with HTTP range requests at REV. The kit's own make_bias_vl.py then runs on DIR unchanged:

    python3 tools/fetch_bias_vl.py deepseek-ai/DeepSeek-V4.1-Flash-1017 <rev> /tmp/bias-src
    python3 <KIT>/tools/make_bias_vl.py /tmp/bias-src <KIT>/vision/gate_bias_vl.safetensors

The sha256 make_bias_vl.py prints is compared with the pinned one by scripts/prepare.sh.
"""
import json, os, struct, sys, time, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hub_fetch import HUB, request, listing


def ranged(url, a, b, tries = 6):
    for i in range(tries):
        try:
            r = urllib.request.urlopen(request(url, {"Range": f"bytes={a}-{b - 1}"}), timeout = 60)
            data = r.read()
            if r.status == 206 and len(data) == b - a:
                return data
        except Exception:
            pass
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"range {a}-{b} of {url} failed")


def main():
    repo, rev, out = sys.argv[1:4]
    os.makedirs(out, exist_ok = True)
    sizes = {e["path"]: e["size"] for e in listing(repo, rev)}
    url = lambda p: f"{HUB}/{repo}/resolve/{rev}/{urllib.parse.quote(p)}"
    idx = urllib.request.urlopen(request(url("model.safetensors.index.json")), timeout = 60).read()
    open(os.path.join(out, "model.safetensors.index.json"), "wb").write(idx)
    wm = json.loads(idx)["weight_map"]
    shards = {}
    for k, s in wm.items():
        if k.endswith(".ffn.gate.bias_vl"):
            shards.setdefault(s, []).append(k)
    if not shards:
        sys.exit(f"no *.ffn.gate.bias_vl tensors in {repo}@{rev}")
    n = 0
    for s, keys in sorted(shards.items()):
        u = url(s)
        hn = struct.unpack("<Q", ranged(u, 0, 8))[0]
        head = ranged(u, 0, 8 + hn)
        meta = json.loads(head[8:])
        with open(os.path.join(out, s), "wb") as f:
            f.truncate(sizes[s])
            f.write(head)
            for k in keys:
                a, b = meta[k]["data_offsets"]
                f.seek(8 + hn + a)
                f.write(ranged(u, 8 + hn + a, 8 + hn + b))
                n += 1
    print(f" -- {n} bias_vl tensors from {len(shards)} shards of {repo}@{rev[:8]} into {out}", flush = True)


if __name__ == "__main__":
    main()
