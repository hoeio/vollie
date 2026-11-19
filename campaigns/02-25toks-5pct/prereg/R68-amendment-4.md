# R68 amendment 4: how a nice factor with no contrast is scored (before session A is seen)

Date: 2026-11-19. Author: clyde (lead). Written and pushed **after** session B was scored and **before** any data of
session A, session R or the cap probes has reached the repository or this session.

## Why

Session B (*device*, `reports/device/r28_2026-11-18.json`) showed every DartWorker and pool thread at nice −10 in
all 72 runs, with or without `VOLLIE_NICE`. Every `c2-nice` line shows zero sets. Tom's reading of the code and the
report is in `reports/memo-33-work/tom-nice-baseline.md`:

- the app's threads inherit −10 from the top app's main thread (an inference; the main thread's nice is not
  recorded);
- 0012 works as written: a target equal to the current value is not set, so there is nothing to restore.

So N and n ran at the same priority. Session A runs the same APK and is expected to show the same.

## Reading (binds sessions A and B)

1. **No contrast.** A session's nice factor is **"VOID: no contrast"** when, in that session, every N run and every
   n run reads the same nice on every pool thread and no `c2-nice` line shows a set (`s1`). It is then neither a
   pass nor a kill. B3's adopted −10 test gives the same outcome in this case, so nothing scored in session B changes.
2. **Permission.** "Permitted" (B1) is reported as written. A note is added that no set was needed.
3. **The claim configuration and the reference runtimes.** Every run of the c2 APK ran with its threads at −10.
   This is the app's state, not a lever. It is recorded as a condition of b1, b2 and b5. It is not credited to
   `VOLLIE_NICE`.
   - b5 is reported as "num + touch (threads at −10, app baseline)".
   - Session R's c1 and stock APKs run under the same app, so their threads are expected at −10 too. Their
     `workers_after` rows are read and reported.
4. **A later kit** that wants to price priority uses an explicit off arm (`VOLLIE_NICE=0`) against −10, and states
   the expected baseline in its pre-registration (tom-nice-baseline.md, "fix for a later kit").

Nothing here changes a prediction, a band, a threshold or the go rule.
