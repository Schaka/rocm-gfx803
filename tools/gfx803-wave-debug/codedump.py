# Dump the code object bytes around the stuck PC straight out of the hung
# process. Code objects are mapped in the process at the same VA the wave's
# PC reports, so /proc/<pid>/mem reaches them with no GPU involvement.
import os, sys
pid = int(sys.argv[1]); pc = int(open("/tmp/stuck_pc.txt").read().strip())
start = pc - 512
try:
    fd = os.open("/proc/%d/mem" % pid, os.O_RDONLY)
    data = os.pread(fd, 1280, start)
    os.close(fd)
except OSError as e:
    print("code read failed: %s" % e); sys.exit(1)
open("/tmp/stuck_code.bin", "wb").write(data)
print("dumped %d bytes from 0x%x (stuck PC 0x%x at offset %d)" % (len(data), start, pc, pc - start))
