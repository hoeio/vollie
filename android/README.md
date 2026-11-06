# Using the patched CPU module in an Android app

llama.cpp's Android builds (with `GGML_BACKEND_DL`) load the CPU backend as a separate module,
picked at runtime by ISA. The patch only touches that module (`ggml/src/ggml-cpu/`), so an app
can keep its prebuilt `libllama.so` / `libggml-base.so` and replace one file:

1. Check out the llama.cpp commit in `patches/llama.cpp/BASE`. It must match the commit your app's
   `libggml-base.so` was built from: the module links against it.
2. `git apply` `patches/llama.cpp/0001-cpu-runtime.patch`, `0002-granite-rollback.patch` and `0003-sampler-bypass.patch`, in that order (0002 and 0003 only change `libllama.so`; see below), and run `android/build_cpu.sh <checkout> out/`.
3. Ship `out/libggml-cpu-android_armv8.2_2.so` in the app's `jniLibs/arm64-v8a/` and remove the
   bundled module of the same name (Vollie drops it from llamadart's `cpu_variants` list).
4. Turn the knobs on from the app (`runtime/dart`, or `setenv` from any language) and pick values
   per device with the procedure in `docs/tuning.md`.

Every library in the output is 16 KB page aligned (`ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES`), as
Google Play requires for apps targeting Android 15+.

## Rebuilding `libllama.so` (for `patches/llama.cpp/0002-granite-rollback.patch` and `0003-sampler-bypass.patch`)

The rollback-slot fix (0002, `src/llama-arch.cpp`) and the sampler fast path (0003, `src/llama-sampler.cpp`; enabled only by `VOLLIE_SAMPLER_BYPASS=1`, bit-identical otherwise) are in `libllama.so`, so it needs a rebuilt `libllama.so`, not just the
module; the rebuilt library carries 0002 and 0003. `android/build_cpu.sh` writes one next to the module (same commit as the bundle's
`libggml-base.so`, so the ABI matches; same SONAME/NEEDED; 16 KB aligned). With llamadart, deliver it
as a local bundle: copy the extracted upstream bundle (`.dart_tool/llamadart/native_bundles/<tag>/android-arm64/extracted/`)
to `third_party/bin/android-arm64/` inside the llamadart package, overwrite `libllama.so`, and apply
`patches/llamadart/0001-setthreads-extra-buffers.patch` (its hook change makes that directory take
precedence over the cached bundle). Clear `.dart_tool/hooks_runner/shared/llamadart` before the
next `flutter build`: the hook cache does not key on the bundle files. Verify with
`unzip -l app-release.apk | grep libllama` — the size must be the rebuilt file's, not the bundle's.

Phones with i8mm or SVE load other variants (`armv8.6_1`, `armv9.2_1`), which this patch also
compiles into but which Vollie still ships stock; only `armv8.2_2` was measured.
