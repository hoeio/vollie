# R68 amendment 5: session R's stock runtime runs from a separate app (install refused on the phone)

Date: 2026-11-19. Author: clyde (lead). Written and pushed before any stock-runtime data exists.

## Why

The phone refused to install `vollie-r28-stock.apk` over the c1 APK with "package invalid". The owner had
downloaded it again and the size matched. The file itself checks clean (*host*):

- the sha256 equals R68 §0;
- the zip is intact and signed v2 with cert `c49dde85…`;
- the manifest is identical to the c1 APK's, which installed;
- only `libllama.so` and `libggml-cpu-android_armv8.2_2.so` differ from c1's.

The cause is not known. The owner declined a USB connection, so the exact installer error could not be read.
Raising the versionCode is not used, because R70 must later reinstall the unchanged c2 APK at versionCode 28.

## Change

The stock reference runs from a **separate app**:

| field | value |
|---|---|
| file | `vollie-c2-1.28.0+28-research-r28-stock-app.apk` (HF `r28/vollie-r28-stock-app.apk`) |
| size | 16,319,610 bytes |
| sha256 | `d689380779b05896d7e94c0f94c3547657a67647128865392729cac7ce776db5` |
| application id | `com.insaneutils.vollie.c2stock` |
| label | "Vollie stock r28" |
| versionCode | 28 |
| certificate | `c49dde85…`, as before |

- **Runtime unchanged.** All 13 native libraries, `libapp.so` included, are byte-identical to the APK in R68 §0
  (sha256 per file; `libllama.so` `de2ccec7…`, CPU module `604e0f69…`). They were copied from that APK, not
  rebuilt: a fresh build differs in its 20-byte GNU build-id.
- **What differs.** Three zip entries differ: `AndroidManifest.xml` and `resources.arsc` (the id and the label),
  and `kit_build.json`, which gains `app_id`. The kit's D3 library check is unchanged and applies as written.
- **Models.** The app has its own storage and downloads the two models again; the kit checks their sha256
  (D3).

Nothing else changes: no prediction, band, rule or reading. The session state is as in R68 §1 for session R.
