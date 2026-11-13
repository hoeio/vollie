# Campaign 2 app patches (research kit)

The app source is not in this repository. These patches apply to a private copy of the Vollie app at version
**2.3.0+45** (the production build of 2026-11-08; the source folder is not a git repository, so the version is
identified by file hashes):

| file | sha256 before the patch |
|---|---|
| `pubspec.yaml` | `0b876797a6c7efcf246ad42e7f8e9f35cfabab0b9323d5f752648fa01d02f258` |
| `lib/assistant.dart` | `fa51cc6fa2b6702f39000ffbab43890eec3bec85e50d944a3de83caadbc17dd6` |
| `lib/research.dart` | `b445efbc5dd45d9e91529103c031cf1e4a96aab6242de1eca407dbd34bb8eb07` |

## 0001-kit-r17.patch

The kit r17 app (`com.insaneutils.vollie.c2`, version 1.17.0+17). Design and report schema:
`campaigns/02-25toks-5pct/reports/memo-10-kit-r17.md`.

- Package `com.insaneutils.vollie.c2`, label "Vollie C2": the app installs **beside** the owner's Vollie and never
  over it.
- Release build signed with the Android debug key.
- Model: `olmoe-Mi.gguf` from `HoeioUser/vollie2` at commit `e893e8a6`. The download is resumable and checks the
  sha256 before use; the storage check asks for about 3.2 GB.
- No Granite defaults. No skip 117, no two-stage head, no sampler bypass. No saved or fallback `GGML_TINY_OP` /
  `GGML_EXPERT_IO` is set at start. Every known knob is unset at every load and before every kit run.
- Weight repack is set per kit run (`repack` in the run spec); a change reloads the model.
- `lib/research.dart` is rewritten as a table-driven kit. Edit `Research.runs` to change what runs.

## Rebuild

```sh
# a private copy of the app (2.3.0+45) without build outputs or signing secrets, e.g. lab/app-c2/
git -C lab/app-c2 apply ../../patches/app-c2/0001-kit-r17.patch   # or: (cd lab/app-c2 && patch -p1 < …)
# llama.cpp 620d2225c + patches/llama.cpp-c2/* in a scratch tree, then:
tools/c2_build_kit.sh <llama.cpp tree> lab/app-c2
```

`tools/c2_build_kit.sh` builds the CPU module and `libllama.so` with `android/build_cpu.sh`. It copies them into the
app copy (`jniLibs` and llamadart's local bundle), wipes the hook caches, and runs
`flutter clean && flutter build apk --release`. It then checks that both libraries in the APK are byte-identical to
the ones just built, and writes the APK and `BUILD-<kit>.txt` to `lab/dist-c2/`.

Do not put a keystore, `key.properties` or `local.properties` into the app copy or into this repository.
