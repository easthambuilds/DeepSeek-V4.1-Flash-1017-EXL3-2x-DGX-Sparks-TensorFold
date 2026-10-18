"""evict_page_cache.py DIR...: drop these files' pages from the page cache (posix_fadvise DONTNEED). No privileges needed.

On GB10 the GPU and the CPU share memory, and the kernel does not reclaim page cache fast enough for CUDA: with the
model's ~200 GB just read (copied, hashed, or the rank cache just written), `free` counts it as available but the
engine's CUDA context can still fail with CUDA_ERROR_OUT_OF_MEMORY. Dropping our own files' pages first avoids that.
"""
import os, sys

n = b = 0
for d in sys.argv[1:]:
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                if os.access(p, os.W_OK):
                    os.fsync(fd)   # dirty pages are not dropped
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1; b += os.fstat(fd).st_size
            finally:
                os.close(fd)
print(f"dropped {n} files ({b / 2**30:.0f} GiB) from the page cache")
