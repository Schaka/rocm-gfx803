#!/usr/bin/env python3
"""Print VGPR/SGPR/LDS usage of the kernels in a compiled HIP .so.

Occupancy on gfx803 is decided by the register count: a wave of 64 lanes at
R VGPRs occupies 64*R of a CU's 65536 registers, so the number of waves a CU
can hold -- the thing that decides whether a memory-bound kernel can keep
enough loads in flight -- falls as R rises. That number is not visible in the
source, only in the code object, so it is worth being able to read it.

The AMDGPU metadata is a msgpack map in an ELF note, and every Python msgpack
decoder refuses the blob because it is preceded by non-msgpack bytes, so the
keys are located by searching for them and the following element decoded
directly. Only positive integers are expected after these keys.

Usage: vgpr_count.py <file.so> [...]
"""

import struct
import sys


def _read_uint(buf: bytes, pos: int) -> tuple[int, int]:
    first = buf[pos]
    if first < 0x80:  # positive fixint
        return first, pos + 1
    if first == 0xCC:
        return buf[pos + 1], pos + 2
    if first == 0xCD:
        return struct.unpack_from(">H", buf, pos + 1)[0], pos + 3
    if first == 0xCE:
        return struct.unpack_from(">I", buf, pos + 1)[0], pos + 4
    raise ValueError(f"unexpected msgpack tag 0x{first:02x}")


def counts(path: str) -> dict[str, list[tuple[str, int]]]:
    buf = open(path, "rb").read()
    out: dict[str, list[tuple[str, int]]] = {}
    for key in (b".vgpr_count", b".sgpr_count", b".lds_size", b".name"):
        pos = 0
        while True:
            pos = buf.find(key, pos)
            if pos < 0:
                break
            after = pos + len(key)
            try:
                if key == b".name":
                    # Kernel names are template instantiations, so they run
                    # past the 31 characters a fixstr covers and use str8.
                    tag = buf[after]
                    if 0xA0 <= tag <= 0xBF:
                        size, start = tag - 0xA0, after + 1
                    elif tag == 0xD9:
                        size, start = buf[after + 1], after + 2
                    elif tag == 0xDA:
                        size = struct.unpack_from(">H", buf, after + 1)[0]
                        start = after + 3
                    else:
                        pos = after
                        continue
                    value = buf[start : start + size].decode("utf-8", "replace")
                    if not value.startswith("_Z"):
                        pos = start
                        continue
                    pos = start + size
                else:
                    value, pos = _read_uint(buf, after)
            except (IndexError, ValueError):
                pos = after
                continue
            out.setdefault(key.decode(), []).append(value)
    names = out.get(".name", [])
    vgpr = out.get(".vgpr_count", [])
    sgpr = out.get(".sgpr_count", [])
    lds = out.get(".lds_size", [])
    rows = []
    for i, n in enumerate(names):
        rows.append(
            (
                n.split("<")[0],
                n[n.find("<") :] if "<" in n else "",
                vgpr[i] if i < len(vgpr) else -1,
                (sgpr[i] if i < len(sgpr) else -1),
                (lds[i] if i < len(lds) else -1),
            )
        )
    return rows


def main() -> None:
    for path in sys.argv[1:]:
        print(path)
        for base, templ, v, s, l in counts(path):
            print(f"  {base}{templ}: vgpr={v} sgpr={s} lds={l}")
    counts = ", ".join(f"{r} vgpr -> {65536 // (64 * r)}" for r in (40, 64, 96, 128))
    print("\nwaves/CU at that count: " + counts)


if __name__ == "__main__":
    sys.exit(main())
