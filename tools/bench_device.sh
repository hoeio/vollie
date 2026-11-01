#!/bin/sh
# On-device llama.cpp benchmark for Snapdragon 888 (Galaxy Z Flip 3) class phones.
# Two ways to run (see tools/BENCH_DEVICE.md):
#   host with adb:  tools/bench_device.sh adb <model.gguf>         (pushes bins+model, runs, pulls results)
#   on the phone:   sh bench_device.sh run <model.gguf> [bindir]   (Termux or `adb shell`)
# Output: results_<device>_<time>.csv (format in BENCH_DEVICE.md) + raw logs.
set -eu
mode=${1:?adb|run}; model=${2:?model.gguf}
D=/data/local/tmp/vollie-bench

if [ "$mode" = adb ]; then
  here=$(cd "$(dirname "$0")" && pwd)
  adb shell mkdir -p $D
  adb push "$here/device-bin/llama-bench" "$here/device-bin/llama-perplexity" "$here/device-bin/COMMIT" "$0" $D/
  adb shell "[ -f $D/$(basename "$model") ]" || adb push "$model" $D/
  [ -n "${PPL_FILE:-}" ] && adb push "$PPL_FILE" $D/ppl.txt
  adb shell "cd $D && chmod +x llama-bench llama-perplexity && sh bench_device.sh run $(basename "$model") ."
  mkdir -p results && adb pull "$D/out" results/ && ls results/out; exit 0
fi

bin=${3:-.}; out=$PWD/out; mkdir -p "$out"
prop() { getprop "$1" 2>/dev/null || echo unknown; }
dev=$(prop ro.product.model | tr ' ,' '__'); soc=$(prop ro.soc.model); [ "$soc" = "" ] && soc=$(prop ro.board.platform)
stamp=$(date +%Y%m%d-%H%M%S); csv=$out/results_${dev}_$stamp.csv
commit=$(head -1 "$bin/COMMIT" 2>/dev/null || echo unknown)
sha=$(sha256sum "$model" 2>/dev/null | cut -c1-16 || echo unknown)
# hottest thermal zone in C (best effort; many zones are unreadable without root)
temp() { t=0; for z in /sys/class/thermal/thermal_zone*/temp; do v=$(cat "$z" 2>/dev/null || echo 0)
  [ "$v" -gt "$t" ] 2>/dev/null && t=$v; done; echo $((t / 1000)); }
cool() { n=0; while [ "$(temp)" -gt "${COOL_C:-42}" ] && [ $n -lt 60 ]; do sleep 5; n=$((n+1)); done; }
set +e  # device info is best effort
{ echo "# device=$dev soc=$soc android=$(prop ro.build.version.release) kernel=$(uname -r)"
  grep -m1 Features /proc/cpuinfo; grep 'CPU part' /proc/cpuinfo | tr '\n' ' '; echo
  for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(basename $c) max_khz=$(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"; done
  grep -E 'MemTotal|MemAvailable' /proc/meminfo; } > "$out/device_$stamp.txt" 2>/dev/null
set -e
echo "device,soc,build_commit,model,model_sha16,threads,cpu_mask,repack,fa,kv_type,test,n_prompt,n_gen,avg_tok_s,stddev_tok_s,temp_start_c,temp_end_c" > "$csv"

# SD888: cpu0-3 A55, cpu4-6 A78, cpu7 X1 -> 0x80 = X1, 0xF0 = all big cores. Override with MASKS="..."
# Vollie's runtime config: flash attention on, q8_0 KV cache, mmap, repack off on 8 GB phones.
for spec in ${MATRIX:-"1:0x80:0" "2:0xC0:0" "3:0xE0:0" "4:0xF0:0" "4:0x0:0" "4:0xF0:1"}; do
  t=${spec%%:*}; rest=${spec#*:}; mask=${rest%%:*}; rp=${rest#*:}
  cool; t0=$(temp)
  "$bin/llama-bench" -m "$model" -t "$t" -C "$mask" --cpu-strict "$([ "$mask" = 0x0 ] && echo 0 || echo 1)" \
    --repack "$rp" -fa on -ctk q8_0 -ctv q8_0 -p 64 -n 64 -r "${REPS:-3}" -o csv \
    > "$out/raw_t${t}_${mask}_r${rp}_$stamp.csv" 2> "$out/raw_t${t}_${mask}_r${rp}_$stamp.log" || { echo "run failed t=$t mask=$mask"; continue; }
  t1=$(temp)
  awk -F, -v pre="$dev,$soc,$commit,$(basename "$model"),$sha,$t,$mask,$rp,on,q8_0" -v t0="$t0" -v t1="$t1" '
    NR==1 { for (i=1;i<=NF;i++) { gsub(/"/,"",$i); c[$i]=i }; next }
    { gsub(/"/,""); np=$c["n_prompt"]; ng=$c["n_gen"]; test=(ng>0?"tg":"pp")
      print pre "," test "," np "," ng "," $c["avg_ts"] "," $c["stddev_ts"] "," t0 "," t1 }' \
    "$out/raw_t${t}_${mask}_r${rp}_$stamp.csv" >> "$csv"
  tail -n +2 "$csv" | tail -2
done
# Optional quality check (kernel equivalence with host numbers): first 20 wikitext-2 chunks, ctx 512.
if [ -f ppl.txt ]; then cool
  "$bin/llama-perplexity" -m "$model" -f ppl.txt -c 512 -b 512 --chunks 20 -t 4 > "$out/ppl_$stamp.log" 2>&1
  grep -E 'Final estimate' "$out/ppl_$stamp.log" || tail -3 "$out/ppl_$stamp.log"; fi
echo "wrote $csv"
