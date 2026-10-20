"""
verify_ablit.py SRC DST SIDECAR: check an ABLIT=1 checkpoint copy (DST, made from SRC by the overlay's graft script)
tensor by tensor, byte for byte. Standard library only.

Every tensor in DST must be one of:
  - one of the SIDECAR's tensors (layers 10-35 attn.wo_b.{trellis,suh,svh,mul1}): the sidecar's bytes, with SRC's dtype
    and shape;
  - one of the four MTP tensors the graft script stores as F32 (ffn.gate.bias x3, confidence_head.proj.weight): SRC's
    values exactly, F16 widened to F32 (or SRC's bytes when SRC already holds F32);
  - anything else: SRC's bytes, dtype and shape.
Shards DST shares with SRC by hardlink are the same file and need no reading. The index and every other file must
equal SRC's. Exit 0 when everything checks; prints a one-line summary.
"""
import json, os, struct, sys

CAST_F32 = {"mtp.0.ffn.gate.bias", "mtp.1.ffn.gate.bias", "mtp.2.ffn.gate.bias", "mtp.2.confidence_head.proj.weight"}
CHUNK = 1 << 26


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def chunks(path, base, spec):
    lo, hi = spec["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + lo)
        left = hi - lo
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                raise ValueError(f"truncated {path}")
            yield b
            left -= len(b)


def same_bytes(a, b):
    ia, ib = iter(a), iter(b)
    ba = bb = b""
    while True:
        if not ba:
            ba = next(ia, b"")
        if not bb:
            bb = next(ib, b"")
        if not ba and not bb:
            return True
        n = min(len(ba), len(bb))
        if n == 0 or ba[:n] != bb[:n]:
            return False
        ba, bb = ba[n:], bb[n:]


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, sidecar = sys.argv[1:]
    bad = []
    wm_s = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    wm_d = json.load(open(os.path.join(dst, "model.safetensors.index.json")))["weight_map"]
    if wm_s != wm_d:
        bad.append("the index differs from SRC's")
    for name in sorted(os.listdir(src)):
        if name.startswith(".") or (name.startswith("model-") and name.endswith(".safetensors")):
            continue
        p, q = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isfile(p) and (not os.path.isfile(q) or open(p, "rb").read() != open(q, "rb").read()):
            bad.append(f"{name} differs from SRC's")
    sbase, side = header(sidecar)
    graft_seen, cast_n, copied_n, linked, rewritten = set(), 0, 0, 0, 0
    for shard in sorted(set(wm_d.values())):
        p, q = os.path.join(src, shard), os.path.join(dst, shard)
        if os.path.samefile(p, q):
            linked += 1
            if any(k in side for k, s in wm_d.items() if s == shard):
                bad.append(f"{shard} holds overlay tensors but is a hardlink to SRC")
            continue
        rewritten += 1
        pb, ph = header(p)
        qb, qh = header(q)
        if set(ph) != set(qh):
            bad.append(f"{shard}: tensor names differ ({sorted(set(ph) ^ set(qh))[:4]})")
            continue
        for k, qs in qh.items():
            ps = ph[k]
            if k in side:
                ss = side[k]
                if (qs["dtype"], qs["shape"]) != (ss["dtype"], ss["shape"]) or (ss["dtype"], ss["shape"]) != (ps["dtype"], ps["shape"]):
                    bad.append(f"{k}: geometry differs (src {ps['dtype']}{ps['shape']}, overlay {ss['dtype']}{ss['shape']})")
                elif not same_bytes(chunks(q, qb, qs), chunks(sidecar, sbase, ss)):
                    bad.append(f"{k}: bytes differ from the overlay's")
                graft_seen.add(k)
            elif k in CAST_F32 and qs["dtype"] == "F32" and ps["dtype"] == "F16":
                a = b"".join(chunks(p, pb, ps))
                b = b"".join(chunks(q, qb, qs))
                widened = struct.pack(f"<{len(a) // 2}f", *struct.unpack(f"<{len(a) // 2}e", a))
                if qs["shape"] != ps["shape"] or widened != b:
                    bad.append(f"{k}: not an exact F16->F32 widening of SRC")
                cast_n += 1
            else:
                if (qs["dtype"], qs["shape"]) != (ps["dtype"], ps["shape"]) or not same_bytes(chunks(q, qb, qs), chunks(p, pb, ps)):
                    bad.append(f"{k}: differs from SRC")
                copied_n += 1
    missing = sorted(set(side) - graft_seen)
    if missing:
        bad.append(f"{len(missing)} overlay tensors not in DST, e.g. {missing[:3]}")
    for b in bad[:20]:
        print("!!", b)
    print(f"   verify_ablit: {len(graft_seen)} overlay tensors, {cast_n} F16->F32 widenings, {copied_n} other tensors equal "
          f"to SRC in {rewritten} rewritten shards, {linked} shards hardlinked: {'OK' if not bad else f'{len(bad)} problems'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
