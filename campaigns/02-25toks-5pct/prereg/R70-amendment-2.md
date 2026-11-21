# R70 amendment 2: the claim session runs on kit r31's c2 runtime

Date: 2026-11-21. Author: clyde (lead). Written and pushed before any R70 data exists.

R70 amendment 1 named kit r30. The phone gets kit r31 (versionCode 31) for R71's S2 and S3 (R71 amendment 1), and
r30 cannot be installed over it. The build under test is therefore kit r31 in its **c2 runtime**:

| field | value |
|---|---|
| APK | `vollie-c2-1.31.0+31-research-r31.apk` (HF `r31/vollie-r31.apk`), 20,807,802 bytes |
| sha256 | `fd9fec6ad00dd34354b24ef28409e5ab065aa4c9593bab6583656441cff0ea61` |
| c2 libraries | `libllama.so` `126bfb4f…`, CPU module `6f1d859a…`, byte-identical to R70 §0 and amendment 1 |

What differs from r30 is the kit code only: app patch 0014. In a single-runtime run, 0014 only keeps the screen on
for the length of the run while a chain is active. A single run does not use the chain, so it behaves as r30.

Everything in amendment 1 still applies, with r31 in place of r30:

- the run is started with "Run research kit once, this runtime" in runtime c2;
- it is its own session, after R71;
- the void additions apply.
