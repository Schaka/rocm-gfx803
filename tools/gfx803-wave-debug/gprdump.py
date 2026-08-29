# Read VGPRs/SGPRs of one wave. pos encoding per amdgpu_debugfs_gpr_read:
# 0..11 byte offset, 12..19 SE, 20..27 SH, 28..35 CU, 36..43 WAVE,
# 44..51 SIMD, 52..59 thread, 60..61 bank (VGPR=0, SGPR=1).
import os, struct, sys
se, cu, simd, wave = (int(x) for x in sys.argv[1:5])
path = "/sys/kernel/debug/dri/0000:02:00.0/amdgpu_gpr"
fd = os.open(path, os.O_RDONLY)

def rd(bank, thread, nregs):
    pos = (se << 12) | (0 << 20) | (cu << 28) | (wave << 36) | (simd << 44) \
          | (thread << 52) | (bank << 60)
    try:
        b = os.pread(fd, nregs * 4, pos)
    except OSError as e:
        return None
    return struct.unpack("<%dI" % (len(b) // 4), b)

for thread in (0, 1, 32):
    v = rd(0, thread, 16)
    if not v:
        print("thread %d: VGPR read failed" % thread); continue
    print("thread %2d VGPR v0..v11: %s" % (thread, " ".join("%08x" % x for x in v[:12])))
    print("           flat addr v[2:3] = 0x%016x   store addr v[4:5] = 0x%016x"
          % ((v[3] << 32) | v[2], (v[5] << 32) | v[4]))
s = rd(1, 0, 32)
if s:
    print("SGPR s0..s25: %s" % " ".join("%08x" % x for x in s[:26]))
    print("  s[8:9]=0x%016x s[10:11]=0x%016x s[12:13]=0x%016x s[16:17]=0x%016x"
          % ((s[9] << 32) | s[8], (s[11] << 32) | s[10],
             (s[13] << 32) | s[12], (s[17] << 32) | s[16]))
os.close(fd)
