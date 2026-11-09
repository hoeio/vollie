# Run it on your phone, and share what you get

The experiment in this repository is not only a log. The Vollie app contains the patched runtime and the
benchmark that the headline number comes from, so you can repeat the measurement on your own phone.

> **Derived page.** Written from [tools/research_kit.md](../tools/research_kit.md),
> [tuning.md](tuning.md), [RULES.md](../RULES.md) and the reports in [../reports/device/](../reports/device/).
> The app's source code is not in this repository, so button labels may differ slightly between app versions;
> follow what your app shows.

## What you need

- An Android phone with [Vollie from Google Play](https://play.google.com/store/apps/details?id=com.insaneutils.vollie).
  The store build carries the round-16 configuration (R49, see the end of [R49](../reports/03_final.md)).
- Room for the model: the frozen file is about 3.16 GB ([models/](../models/granite-4.0-h-tiny-d6-e6.md)).
  Every result in this repository was measured on an 8 GB phone; smaller phones have not been tested here.
- Nothing else. You do not need to close apps or free memory first: the project's rule is that the runtime
  must cope with the phone as it is ([RULES.md](../RULES.md) §3).

## Two ways to measure

| in the app | what it does | time |
|---|---|---|
| Settings → **Optimize for this phone** | the speed test: loads the model once, sweeps the runtime knobs and thread counts, picks the fastest, and profiles; **Copy speed test details** puts the result on the clipboard ([tuning.md](tuning.md)) | not stated here; see [tuning.md](tuning.md) |
| Settings → **Run research kit** | the measurement behind R49: a cold baseline, two story runs, and four groups of official 32-token runs (10 plain, 8 profiled); no settings to change ([research_kit.md](../tools/research_kit.md)) | about 9-35 min (*estimated* in the kit card) |

The kit waits for the phone's thermal headroom to drop before its timed runs (up to 90 s each), as the app's
speed test does, so it is slower on a warm phone. When it ends, **Copy research results** puts one JSON report
on the clipboard.

## Read your result

Paste the clipboard into a file, for example `my-phone.json`, and run the project's parser (Python 3, standard
library only):

```sh
python3 -I tools/research_parse.py my-phone.json --official
```

The first line of the report names the kit; `--official` is for kit `r16` reports. Line 1 of the output is the
median and the best printed tok/s of the 10 plain official runs, with the cold baseline excluded, and it says
"20 reached" only if the median is at least 20.0. That is the same rule as the headline result in
[R49](../reports/03_final.md).

To compare, the raw reports of the Galaxy Z Flip 3 are in [../reports/device/](../reports/device/); round 16
is `r16_2026-11-08.json`.

What to expect:

- The first run after memory pressure is slower. On the Z Flip 3 it printed 12.3-18.5 tok/s against a median
  of 20.51 (R49). That is expected, and worth reporting rather than hiding.
- Other phones will print different numbers. This project measured one chip (Snapdragon 888); how far the
  method carries is the open question ([README](../README.md#honest-limits)). A result below 20 on your phone
  is a data point, not an error.

## Share your result

**If you run the research kit on a different phone and email us the report, we would be very grateful.** All
the results here come from one phone and one model, and reports from other devices are the evidence we cannot
produce ourselves.

- **Where:** the support email shown on the app's
  [Google Play page](https://play.google.com/store/apps/details?id=com.insaneutils.vollie).
- **Attach** the report as a file. It is plain JSON of roughly 170-180 KB (*estimated*), too long to paste into
  a message.
- **Write in the email:** the phone's model name and Android version, because the report does not contain
  them; the app version; and, if you know it, whether the phone had just been restarted or had been busy
  before the run.

**What is in a report.** Judging from the reports in [../reports/device/](../reports/device/): the kit and app
version, start and end time, the chip name, RAM size, CPU core types and maximum clocks, CPU feature flags, the
GPU name, the settings in use (threads, context size), memory statistics (free, cached, swap), and per run the
timings, clock readings, page-fault counts and the model's replies to the benchmark's built-in prompts. None of
those 18 reports has an account, name, e-mail or serial-number field. It is plain text, so open it and read it
before you send it.
