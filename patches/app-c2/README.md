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

## 0002-kit-r18.patch (on top of 0001)

Kit r18, the lever kit (version 1.18.0+18). It keeps the package, the debug key and the model constants, so it installs as
an update over r17 and keeps the downloaded model. Design, run list and report schema:
`campaigns/02-25toks-5pct/reports/memo-12-kit-r18.md`.

- The run list of memo-12 (stock, stack, profiled, one in-run A/B per lever, skip, head2, package, repack off).
- The runtime's lines go to stderr (`VOLLIE_LINES=stderr`, always set). They are found anywhere in a log line, not only
  at its start, and are filed to the right run by their `run=` id (`VOLLIE_RUN` is set per run). A prompt's per-prompt
  lines arrive late (memo-11 §8); the last run's lines are flushed by freeing the context at the end of the kit.
- Per-token lines are kept as compact tuples (`tok_fields`). `profiled32` runs also keep the full lines.
- The env actually set is read back with `getenv`.
- The build record `assets/kit_build.json` (llama.cpp commit, sha256 of every series patch, app patch and built
  library) is written by `tools/c2_build_kit.sh` and goes into the report. The file is not in the patch.
- Reports over 300,000 characters are copied to the clipboard as `c2gz:` + base64(gzip(JSON)); the file is plain JSON.
- `test/kit_lines_test.dart`: line parsing (`flutter test test/kit_lines_test.dart`).

## Rebuild

```sh
# a private copy of the app (2.3.0+45) without build outputs or signing secrets, e.g. lab/app-c2/
for p in patches/app-c2/0*.patch; do git -C lab/app-c2 apply "$PWD/$p"; done
# llama.cpp 620d2225c + every patch of patches/llama.cpp-c2/ in order, in a scratch tree, then:
tools/c2_build_kit.sh <llama.cpp tree> lab/app-c2     # checks the tree against the series file by file
```

`tools/c2_build_kit.sh` builds the CPU module and `libllama.so` with `android/build_cpu.sh`. It copies them into the
app copy (`jniLibs` and llamadart's local bundle), wipes the hook caches, and runs
`flutter clean && flutter build apk --release`. It then checks that both libraries in the APK are byte-identical to
the ones just built, and writes the APK and `BUILD-<kit>.txt` to `lab/dist-c2/`.

Do not put a keystore, `key.properties` or `local.properties` into the app copy or into this repository.
