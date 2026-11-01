/// App-side helpers for serving llama.cpp on Android phones, from Vollie.
///
/// Pure Dart (dart:ffi, dart:io), no Flutter: set the patched CPU module's
/// knobs, pin llama.cpp's threads to the big cores, read what the kernel says
/// about page faults, memory and clocks, and pick the fastest measured setting.
/// Off Android every call is a no-op or returns null/empty.
library;

import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;

// --- Knobs of the patched CPU module (patches/llama.cpp, docs/runtime-knobs.md) ---

/// Ops of at most [elements] elements run on one thread with no sync point
/// between consecutive ones (GGML_TINY_OP; ggml reads it before every graph).
/// 0 = stock. SD888, Granite-4.0-H-Tiny: 65536 doubled writing at 2 threads.
void setTinyOps(int elements) => setNativeEnv('GGML_TINY_OP', '$elements');

/// How the CPU module reads MoE experts back from the mmapped model
/// (GGML_EXPERT_IO, bits): 1 prefetch the routed experts of a layer together,
/// 2 no read-around on expert tensors (lost on an SD888), 4 also guess the
/// next layer's experts and prefetch them a layer early, 8 guess 2 more.
/// 0 = stock. Advice only: results are unchanged.
void setExpertIo(int mode) => setNativeEnv('GGML_EXPERT_IO', '$mode');

/// Per-op timing lines on stderr ("cpu-prof ..."), one per 32 decode tokens.
void setProfiling(bool on) => setNativeEnv('GGML_CPU_PROF', on ? '1' : '0');

/// Alternates a compute setting every decode token inside one generation
/// ("q3k", "tiny:<n>", or "" for off); with [setProfiling] the stats come out
/// split into "ab=A" and "ab=B" lines over the same window.
void setAb(String spec) => setNativeEnv('GGML_CPU_AB', spec);

/// Sets an environment variable for native code in this process.
// Leaks two short strings per call; meant for a handful of calls per session.
void setNativeEnv(String name, String value) {
  final setenv = _setenv;
  if (setenv == null) return;
  setenv(_cString(name), _cString(value), 1);
}

// --- Threads ---

/// Restricts llama.cpp's threads to [cpus] (all cores when null).
///
/// Dart FFI bindings such as llamadart run llama.cpp on Dart's "DartWorker"
/// pool threads, and ggml (built without OpenMP) starts its compute threads
/// from the calling thread, so they inherit its affinity. Without this,
/// Android can park a worker on a little core and every sync waits for it.
/// Re-apply before each generation: the pool can grow.
void pinInferenceThreads(List<int>? cpus) {
  final set = _setAffinity, mask = _mask;
  if (set == null || mask == null) return;
  mask.value = cpus == null ? -1 : cpus.where((c) => c < 64).fold<int>(0, (m, c) => m | (1 << c));
  try {
    for (final t in Directory('/proc/self/task').listSync()) {
      final comm = File('${t.path}/comm').readAsStringSync().trim();
      if (comm.startsWith('DartWorker')) set(int.parse(t.path.split('/').last), 8, mask);
    }
  } on FileSystemException {
    // A thread exited mid-scan; the next call re-applies.
  }
}

/// In-order "little" core designs (MIDR part numbers): Cortex-A53, A35, A55,
/// A510, A520 and Qualcomm's Kryo Silver variants of them.
const littleParts = {0xd03, 0xd04, 0xd05, 0xd46, 0xd80, 0x801, 0x803, 0x805};

/// Cores worth running llama.cpp on; empty when nothing is known. Every
/// thread waits for the slowest at each sync, so one little core holds all
/// the others back (SD888: 4 big-core threads read 23 tok/s, 6-8 threads with
/// the A55s 14.5). By core design ([parts], /proc/cpuinfo "CPU part"), else
/// scheduler capacity (/sys/.../cpu_capacity, littles < half the biggest),
/// else max frequency ([khz]; drop the lowest cluster when < 75% of the top).
List<int> fastCoreIds({required List<int> khz, List<int> capacity = const [], List<int> parts = const []}) {
  final n = khz.length;
  bool complete(List<int> v) => n > 0 && v.length == n && v.every((x) => x > 0);
  List<int> where(bool Function(int i) keep) => [for (var i = 0; i < n; i++) if (keep(i)) i];

  if (complete(parts)) {
    final big = where((i) => !littleParts.contains(parts[i]));
    return big.isEmpty ? where((_) => true) : big; // all-little budget chip
  }
  if (complete(capacity)) {
    final top = capacity.reduce(math.max);
    return where((i) => capacity[i] >= top * 0.5);
  }
  final known = khz.where((f) => f > 0).toList();
  if (known.isEmpty) return const [];
  final top = known.reduce(math.max);
  final low = known.reduce(math.min);
  return where((i) => khz[i] > 0 && (low >= top * 0.75 || khz[i] > low));
}

/// Default thread count: the [fastCoreIds], minus one when every core
/// qualifies, so Android's UI/system threads don't preempt a worker (an
/// 8-core device ran 5.2 tok/s with 8 threads vs 13.0 with 7).
int performanceCores({required List<int> khz, List<int> capacity = const [], List<int> parts = const []}) {
  final n = khz.length;
  var count = fastCoreIds(khz: khz, capacity: capacity, parts: parts).length;
  if (count == 0) return math.max(1, math.min(4, n ~/ 2));
  if (count == n && n > 1) count--;
  return count.clamp(1, 8);
}

// --- What the kernel says ---

/// This process's page faults so far (/proc/self/stat fields 10 and 12);
/// major ones read a page from storage. Take one before and after a
/// generation: on an 8 GB phone whose page cache is smaller than the model,
/// hundreds of major faults per token mean the weights are re-read from flash.
({int minor, int major})? pageFaults() {
  try {
    final stat = File('/proc/self/stat').readAsStringSync();
    final f = stat.substring(stat.lastIndexOf(')') + 2).split(' '); // f[0] = field 3
    return (minor: int.parse(f[7]), major: int.parse(f[9]));
  } catch (_) {
    return null;
  }
}

/// System MemAvailable/Cached/SwapFree and this process's RssAnon (app memory,
/// incl. llama.cpp's repacked copy), RssFile (mapped file pages resident,
/// mostly the model) and VmSwap, in MB, as one "key=value" string.
String memInfo() {
  Map<String, int> read(String path) => {
        for (final l in File(path).readAsLinesSync())
          l.split(':').first: int.tryParse(l.split(':').last.trim().split(' ').first) ?? 0,
      };
  try {
    final m = read('/proc/meminfo'), me = read('/proc/self/status');
    return 'avail=${(m['MemAvailable'] ?? 0) ~/ 1024}MB cached=${(m['Cached'] ?? 0) ~/ 1024}MB '
        'swapfree=${(m['SwapFree'] ?? 0) ~/ 1024}MB anon=${(me['RssAnon'] ?? 0) ~/ 1024}MB '
        'file=${(me['RssFile'] ?? 0) ~/ 1024}MB swap=${(me['VmSwap'] ?? 0) ~/ 1024}MB';
  } catch (_) {
    return '';
  }
}

/// Current clock of every core in MHz ("mhz=1804,1804,…"): shows throttling
/// next to a timing.
String cpuMhz() {
  final mhz = <int>[];
  for (var c = 0; c < 16; c++) {
    try {
      mhz.add(int.parse(File('/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq').readAsStringSync().trim()) ~/ 1000);
    } catch (_) {
      break;
    }
  }
  return mhz.isEmpty ? '' : 'mhz=${mhz.join(',')}';
}

/// Points this process's stderr at [path]. llama.cpp logs (and the profiler
/// lines) go to stderr, which Android discards.
void stderrTo(String path) {
  final libc = _libc;
  if (libc == null) return;
  final open = libc.lookupFunction<Int32 Function(Pointer<Uint8>, Int32, Int32), int Function(Pointer<Uint8>, int, int)>('open');
  final dup2 = libc.lookupFunction<Int32 Function(Int32, Int32), int Function(int, int)>('dup2');
  final fd = open(_cString(path), 0x241 /* O_WRONLY | O_CREAT | O_TRUNC */, 0x1A4 /* 0644 */);
  if (fd >= 0) dup2(fd, 2);
}

// --- Picking a setting from timings ---

/// One timed run: prompt reading ([prefill]) and reply writing ([decode]) in
/// tokens/s, null when it failed. [cpu] false for a GPU backend.
class Timing {
  final String label;
  final bool cpu;
  final int threads;
  final double? prefill, decode;
  final int? readThreads; // prompt threads, when not [threads]
  final double? headroom; // thermal headroom when the run started (1.0 = severe)
  const Timing(this.label, this.threads, this.prefill, this.decode, {this.cpu = true, this.readThreads, this.headroom});

  bool get ok => prefill != null && decode != null;

  /// Seconds for a typical chat turn: the prompt (history is re-read every
  /// turn) plus the reply. Fixed 500-in / 200-out mix.
  double get secondsPerTurn => 500 / prefill! + 200 / decode!;

  /// The fastest setting in [results]. llama.cpp reads prompts with one thread
  /// count and writes replies with another, and they peak apart (SD888 with
  /// the expert-guess helper on: writing best at 3, reading at 4), so the CPU
  /// counts as its best writer paired with its best reader. Each count is
  /// scored by its best run: a warm phone's single runs vary about ±20%.
  /// Readers over [bigCores] are skipped while the writer fits (one affinity
  /// mask for both).
  static Timing? fastest(List<Timing> results, {int? bigCores}) {
    final ok = results.where((r) => r.ok).toList();
    final cpu = ok.where((r) => r.cpu).toList();
    final candidates = [
      ...ok.where((r) => !r.cpu),
      if (cpu.isNotEmpty)
        () {
          final w = cpu.reduce((a, b) => b.decode! > a.decode! ? b : a);
          final r = cpu
              .where((r) => bigCores == null || r.threads <= math.max(bigCores, w.threads))
              .reduce((a, b) => b.prefill! > a.prefill! ? b : a);
          return Timing('CPU', w.threads, r.prefill, w.decode, readThreads: r.threads);
        }(),
    ]..sort((x, y) => x.secondsPerTurn.compareTo(y.secondsPerTurn));
    return candidates.isEmpty ? null : candidates.first;
  }
}

// --- libc ---

typedef _SetAffinityC = Int32 Function(Int32 tid, IntPtr size, Pointer<Uint64> mask);
typedef _SetAffinity = int Function(int tid, int size, Pointer<Uint64> mask);

final _libc = Platform.isAndroid ? DynamicLibrary.open('libc.so') : null;
final _setAffinity = _libc?.lookupFunction<_SetAffinityC, _SetAffinity>('sched_setaffinity');
final _malloc = _libc?.lookupFunction<Pointer<Uint8> Function(IntPtr), Pointer<Uint8> Function(int)>('malloc');
final _mask = _malloc == null ? null : _malloc!(8).cast<Uint64>();
final _setenv = _libc?.lookupFunction<Int32 Function(Pointer<Uint8>, Pointer<Uint8>, Int32),
    int Function(Pointer<Uint8>, Pointer<Uint8>, int)>('setenv');

Pointer<Uint8> _cString(String s) {
  final b = s.codeUnits, p = _malloc!(b.length + 1);
  for (var i = 0; i < b.length; i++) {
    p[i] = b[i];
  }
  p[b.length] = 0;
  return p;
}
