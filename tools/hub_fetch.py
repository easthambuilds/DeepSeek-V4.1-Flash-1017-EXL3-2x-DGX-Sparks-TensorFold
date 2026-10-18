"""
hub_fetch.py REPO REV OUT [PATTERN ...]: download files of a Hugging Face model repo at a pinned revision and check
every byte. Standard library only.

The file list comes from the Hub's tree API at REV. Large (LFS) files are checked against their sha256, small ones
against their git blob id, so nothing is trusted on its size alone. PATTERNs (fnmatch, e.g. "kit/*") limit the
download; with none, the whole repo is fetched. Downloads resume, files already present and correct are skipped, and
four files are fetched at a time. HF_TOKEN (or ~/.cache/huggingface/token) is sent when present.
Exit 0 when every selected file is present and correct.
"""
import fnmatch, hashlib, json, os, sys, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None


def request(url, headers = None):
    h = dict(headers or {})
    t = token()
    if t:
        h["Authorization"] = f"Bearer {t}"
    return urllib.request.Request(url, headers = h)


def get_json(url, tries = 6):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(request(url), timeout = 60))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def listing(repo, rev):
    out, url = [], f"{HUB}/api/models/{repo}/tree/{rev}?recursive=true"
    while url:
        page = urllib.request.urlopen(request(url), timeout = 60)
        out += [e for e in json.load(page) if e["type"] == "file"]
        nxt = page.headers.get("Link", "")
        url = nxt.split(";")[0].strip("<> ") if 'rel="next"' in nxt else None
    return out


def digest(path, entry):
    if entry.get("lfs"):
        h = hashlib.sha256()
        want = entry["lfs"]["oid"]
    else:
        h = hashlib.sha1(b"blob %d\0" % entry["size"])
        want = entry["oid"]
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest() == want


lock = threading.Lock()


def fetch(repo, rev, out, entry):
    path = os.path.join(out, entry["path"])
    size = entry["size"]
    os.makedirs(os.path.dirname(path), exist_ok = True)
    if os.path.exists(path) and os.path.getsize(path) == size and digest(path, entry):
        return "have", entry["path"]
    part = path + ".part"
    url = f"{HUB}/{repo}/resolve/{rev}/{urllib.parse.quote(entry['path'])}"
    for i in range(8):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have < size:
                r = urllib.request.urlopen(request(url, {"Range": f"bytes={have}-"}), timeout = 120)
                if have and r.status != 206:
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    for b in iter(lambda: r.read(1 << 22), b""):
                        f.write(b)
            if os.path.getsize(part) == size:
                break
            if os.path.getsize(part) > size:
                os.remove(part)
        except Exception as e:
            with lock:
                print(f"   retry {entry['path']}: {type(e).__name__}", flush = True)
            time.sleep(5 * (i + 1))
    if not os.path.exists(part) or os.path.getsize(part) != size:
        return "FAILED (size)", entry["path"]
    if not digest(part, entry):
        os.remove(part)
        return "FAILED (checksum)", entry["path"]
    os.replace(part, path)
    return "ok", entry["path"]


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    repo, rev, out, pats = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    files = [e for e in listing(repo, rev) if not pats or any(fnmatch.fnmatch(e["path"], p) for p in pats)]
    if not files:
        sys.exit(f"no files of {repo}@{rev} match {pats}")
    total = sum(e["size"] for e in files)
    print(f" -- {repo}@{rev[:8]}: {len(files)} files, {total / 1e9:.1f} GB", flush = True)
    bad = 0
    with ThreadPoolExecutor(4) as ex:
        for status, path in ex.map(lambda e: fetch(repo, rev, out, e), files):
            with lock:
                print(f"   {status:6s} {path}", flush = True)
            bad += status.startswith("FAILED")
    if bad:
        sys.exit(f"!! {bad} files failed")
    print(f" -- all {len(files)} files present and checked", flush = True)


if __name__ == "__main__":
    main()
