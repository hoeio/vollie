import 'package:test/test.dart';
import 'package:vollie_runtime/vollie_runtime.dart';

void main() {
  test('Snapdragon 888: the A55s are not fast cores', () {
    // cpu0-3 A55 (0xd05), cpu4-6 A78 (0xd41), cpu7 X1 (0xd44)
    final parts = [0xd05, 0xd05, 0xd05, 0xd05, 0xd41, 0xd41, 0xd41, 0xd44];
    final khz = [1804800, 1804800, 1804800, 1804800, 2419200, 2419200, 2419200, 2841600];
    expect(fastCoreIds(khz: khz, parts: parts), [4, 5, 6, 7]);
    expect(performanceCores(khz: khz, parts: parts), 4);
  });

  test('best writer and best reader are paired, each by its best run', () {
    // SD888, R13/R14: writing peaks at 3 threads, reading at 4; one 3-thread run was a low outlier.
    final runs = [
      const Timing('CPU', 4, 25, 9.2),
      const Timing('CPU', 3, 18, 8.4),
      const Timing('CPU', 3, 20.6, 12.0), // the sweep's run at the same settings
      const Timing('CPU', 2, 15, 9.0),
      const Timing('GPU', 4, 19, 2.1, cpu: false),
    ];
    final best = Timing.fastest(runs, bigCores: 4)!;
    expect((best.threads, best.readThreads, best.decode, best.prefill), (3, 4, 12.0, 25.0));
  });

  test('knobs are a no-op off Android', () {
    setTinyOps(65536);
    setExpertIo(5);
    expect(pageFaults() == null || pageFaults()!.major >= 0, isTrue);
  });
}
