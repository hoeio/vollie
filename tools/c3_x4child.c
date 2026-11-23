// Campaign 3 kit r35 (R75, patches/llama.cpp-c3/0007-x4-join-probe): the child process of the x4d and x4e probe blocks.
// The parent (llama_vollie_probe, block x4d / x4e) spawns this with VOLLIE_PROBE=1, VOLLIE_PROBE_TEE=1 and LD_LIBRARY_PATH = the
// app's native library dir, kills it at a timeout and copies the "c3-probe: x4 ..." lines it wrote to stdout. A crash or hang of the
// GPU driver (recordable queue, persistent kernels) then ends only this process. It holds no probe logic: it loads libllama.so
// and runs one block; the lines are written by the probe as they are printed (tee), so a killed child keeps its output so far.
// Shipped as libvollie_x4c.so so that Android extracts it with the native libraries (as campaign 1's libvollie_clc.so).
// Build (kit): aarch64-linux-android28-clang -O2 -s -Wl,-z,max-page-size=16384 tools/c3_x4child.c -ldl -o libvollie_x4c.so
// usage: libvollie_x4c.so <block> <dir>    exit 0 = the block ran, 1 = it returned no lines, 2 = setup
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    void * l = dlopen("libllama.so", RTLD_NOW | RTLD_LOCAL);
    if (!l) { printf("c3-probe: x4 child error=dlopen text=%s\n", dlerror()); return 2; }
    int (*f)(const char *, const char *, char *, size_t) = (int (*)(const char *, const char *, char *, size_t)) dlsym(l, "llama_vollie_probe");
    if (!f) { printf("c3-probe: x4 child error=dlsym\n"); return 2; }
    const size_t cap = 1 << 20;
    char * b = malloc(cap);
    if (!b) return 2;
    const int n = f(argv[1], argv[2], b, cap);
    fflush(stdout);
    return n > 0 ? 0 : 1;
}
