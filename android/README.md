# Using the patched CPU module in an Android app

llama.cpp's Android builds (with `GGML_BACKEND_DL`) load the CPU backend as a separate module,
picked at runtime by ISA. The patch only touches that module (`ggml/src/ggml-cpu/`), so an app
can keep its prebuilt `libllama.so` / `libggml-base.so` and replace one file:

1. Check out the llama.cpp commit in `patches/llama.cpp/BASE`. It must match the commit your app's
   `libggml-base.so` was built from: the module links against it.
2. Apply `patches/llama.cpp/0001-cpu-runtime.patch` and run `android/build_cpu.sh <checkout> out/`.
3. Ship `out/libggml-cpu-android_armv8.2_2.so` in the app's `jniLibs/arm64-v8a/` and remove the
   bundled module of the same name (Vollie drops it from llamadart's `cpu_variants` list).
4. Turn the knobs on from the app (`runtime/dart`, or `setenv` from any language) and pick values
   per device with the procedure in `docs/tuning.md`.

Every library in the output is 16 KB page aligned (`ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES`), as
Google Play requires for apps targeting Android 15+.

Phones with i8mm or SVE load other variants (`armv8.6_1`, `armv9.2_1`), which this patch also
compiles into but which Vollie still ships stock; only `armv8.2_2` was measured.
