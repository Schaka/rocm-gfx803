# Full gfx8 wave dump. Field order per gfx_v8_0_read_wave_data():
# 0 type, 1 STATUS, 2 PC_LO, 3 PC_HI, 4 EXEC_LO, 5 EXEC_HI, 6 HW_ID,
# 7 INST_DW0, 8 INST_DW1, 9 GPR_ALLOC, 10 LDS_ALLOC, 11 TRAPSTS, 12 IB_STS,
# 13 TBA_LO, 14 TBA_HI, 15 TMA_LO, 16 TMA_HI, 17 IB_DBG0, 18 M0, 19 MODE
import os, struct, sys
path = "/sys/kernel/debug/dri/0000:02:00.0/amdgpu_wave"
fd = os.open(path, os.O_RDONLY)
waves = []
for se in range(4):
    for cu in range(9):
        for simd in range(4):
            for wave in range(10):
                pos = (se << 7) | (cu << 23) | (wave << 31) | (simd << 37)
                try:
                    b = os.pread(fd, 80, pos)
                except OSError:
                    continue
                if len(b) < 80:
                    continue
                d = struct.unpack("<20I", b)
                if d[1]:
                    waves.append((se, cu, simd, wave, d))
os.close(fd)

def trapsts(v):
    f = []
    if v & 0x1: f.append("INVALID")
    if v & 0x40: f.append("INT_DIV0")
    if v & 0x80: f.append("ADDR_WATCH")
    if v & 0x100: f.append("MEM_VIOL")
    if v & 0x400: f.append("SAVECTX")
    if v & 0x800: f.append("ILLEGAL_INST")
    return ",".join(f) if f else "clean"

print("RESIDENT WAVES: %d" % len(waves))
for se, cu, simd, wave, d in waves:
    pc = (d[3] << 32) | d[2]
    ib = d[12]
    print("SE%d CU%d SIMD%d W%d PC=0x%012x INST=0x%08x STATUS=0x%08x "
          "TRAPSTS=0x%08x(%s) IB_STS=0x%08x[vm=%d exp=%d lgkm=%d valu=%d] M0=0x%08x"
          % (se, cu, simd, wave, pc, d[7], d[1], d[11], trapsts(d[11]), ib,
             ib & 0xF, (ib >> 4) & 0x7, (ib >> 8) & 0xF, (ib >> 12) & 0x7, d[18]))

# Report the distinct PCs so "one stuck wave vs all" is obvious at a glance.
pcs = {}
for se, cu, simd, wave, d in waves:
    pc = (d[3] << 32) | d[2]
    pcs.setdefault(pc, 0)
    pcs[pc] += 1
print("DISTINCT PCs:")
for pc, n in sorted(pcs.items(), key=lambda x: -x[1]):
    print("  0x%012x  x%d" % (pc, n))
# Emit the minority PC (the stuck one) for the code dumper.
if pcs:
    odd = min(pcs.items(), key=lambda x: x[1])[0]
    open("/tmp/stuck_pc.txt", "w").write("%d\n" % odd)
