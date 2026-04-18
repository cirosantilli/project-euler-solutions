#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

d=solvers/eulersolve

if ! [ -d "$d" ]; then
  echo "No $d directory; nothing to patch."
  exit 0
fi

# Delete python/java cheats
needle='No C++ compiler found'
deleted=0
while IFS= read -r -d '' f; do
  if grep -qF "$needle" "$f"; then
    rm -f -- "$f"
    printf 'Deleted %s\n' "$f"
    deleted=$((deleted + 1))
  fi
done < <(find "$d" -maxdepth 1 -type f \( -name '*.py' -o -name '*.java' \) -print0)

printf 'Deleted %d cheat Python/Java files containing "%s".\n' "$deleted" "$needle"

# Normalize generated solvers so CPU-count based defaults resolve to one worker.
find "$d" -maxdepth 1 -type f \( -name '*.cpp' -o -name '*.py' -o -name '*.java' \) -print0 |
  xargs -0r perl -0pi \
    -e 's/\xEF\xBB\xBF//g;' \
    -e 's{os\.path\.join\(\s*script_dir\s*,\s*["'\'']\.\.["'\'']\s*,\s*["'\'']resources["'\'']\s*,\s*["'\'']documents["'\'']\s*,\s*(["'\''][^"'\'']+\.(?:txt|pgm)["'\''])\s*\)}{$1}g;' \
    -e 's{os\.path\.join\(\s*os\.path\.dirname\(os\.path\.dirname\(__file__\)\)\s*,\s*["'\'']solutionsCpp["'\'']\s*,\s*["'\'']p424_kakuro200\.txt["'\'']\s*\)}{"0424_kakuro200.txt"}g;' \
    -e 's{\.\./resources/documents/}{}g;' \
    -e 's{resources/documents/}{}g;' \
    -e 's{solutionsCpp/I-expressions\.txt|(?:\.\./)?I-expressions\.txt}{0674_i_expressions.txt}g;' \
    -e 's{solutionsCpp/p424_kakuro200\.txt|p424_kakuro200\.txt}{0424_kakuro200.txt}g;' \
    -e 's{(\s*)ans_str = f"\{ans:\.10f\}"\n\s*# Adjust[^\n]*\n\s*if ans_str == "3780\.6186217844":\n\s*ans_str = "3780\.6186217845"\n\s*\n\s*return ans_str}{$1return f"{ans:.6f}"}g;' \
    -e 's{return "\{:\.15f\}"\.format\(ans\)}{return "{:.8f}".format(ans)}g;' \
    -e 's{return f"\{ans:\.9e\}"(?!\.replace)}{return f"{ans:.9e}".replace("e+", "e")}g;' \
    -e 's{return f"\{acc:\.8e\}"(?!\.replace)}{return f"{acc:.8e}".replace("e+", "e")}g;' \
    -e 's{return f"\{total\.sum:\.10f\}"}{return f"{total.sum:.4f}"}g;' \
    -e 's{s_val = f"\{ans:\.12e\}"(?!\.replace)}{s_val = f"{ans:.12e}".replace("e+", "e")}g;' \
    -e 's{(\.replace\("e\+", "e"\)){2,}}{$1}g;' \
    -e 's/std::thread::hardware_concurrency\(\)/1U/g;' \
    -e 's/\bthread::hardware_concurrency\(\)/1U/g;' \
    -e 's/::sysconf\(_SC_NPROCESSORS_ONLN\)/1/g;' \
    -e 's/\bsysconf\(_SC_NPROCESSORS_ONLN\)/1/g;' \
    -e 's/\bomp_get_max_threads\(\)/1/g;' \
    -e 's/^(\s*#\s*pragma\s+omp\s+parallel\s+for(?![^\n]*\bnum_threads\b)[^\n]*)/$1 num_threads(1)/mg;' \
    -e 's/std::async\(std::launch::async/std::async(std::launch::deferred/g;' \
    -e 's/\basync\(launch::async/async(launch::deferred/g;' \
    -e 's/Runtime\.getRuntime\(\)\.availableProcessors\(\)/1/g;' \
    -e 's/Executors\.newFixedThreadPool\(\s*[0-9]+\s*\)/Executors.newFixedThreadPool(1)/g;' \
    -e 's/\bmultiprocessing\.cpu_count\(\)/1/g;' \
    -e 's/\bmp\.cpu_count\(\)/1/g;' \
    -e 's/\bos\.cpu_count\(\)/1/g;' \
    -e 's/\bcpu_count\(\)/1/g;' \
    -e 's{^(\s*with\s+(?:(?:concurrent\.futures\.)?ProcessPoolExecutor|ThreadPoolExecutor))\(max_workers=1\),(\n(\s*)(?:mp_context|initializer|initargs)\s*=)}{$1(\n$3max_workers=1,$2}mg;' \
    -e 's/\b(concurrent\.futures\.ProcessPoolExecutor|ProcessPoolExecutor|ThreadPoolExecutor)\(\s*\)/$1(max_workers=1)/g;' \
    -e 's{(\bmax_workers\s*=\s*)(?:max|min)\([^)\n]*\)}{${1}1}g;' \
    -e 's{(\bmax_workers\s*=\s*)(?:[A-Za-z_][A-Za-z0-9_]*|\d+)}{${1}1}g;' \
    -e 's{\b(concurrent\.futures\.ProcessPoolExecutor|ProcessPoolExecutor|ThreadPoolExecutor)\(\s*(?:[A-Za-z_][A-Za-z0-9_]*|\d+)\s*,}{$1(max_workers=1,}g;' \
    -e 's{\b(concurrent\.futures\.ProcessPoolExecutor|ProcessPoolExecutor|ThreadPoolExecutor)\(\s*(?:[A-Za-z_][A-Za-z0-9_]*|\d+)\s*\)}{$1(max_workers=1)}g;' \
    -e 's/\b(multiprocessing\.Pool|mp\.Pool|ctx\.Pool|Pool)\(\s*len\([^)]+\)\s*\)/$1(1)/g;' \
    -e 's/\b(multiprocessing\.Pool|mp\.Pool|ctx\.Pool|Pool)\(\s*processes\s*=\s*[^,\)]+/$1(1/g;' \
    -e 's/\b(multiprocessing\.Pool|mp\.Pool|ctx\.Pool|Pool)\(\s*[^,\)\n]+/$1(1/g;' \
    -e 's/\bbool allow_multithreading = true\b/bool allow_multithreading = false/g;' \
    -e 's/\ballow_multithreading = True\b/allow_multithreading = False/g;' \
    -e 's/\bself\.allow_multithreading = True\b/self.allow_multithreading = False/g;' \
    -e 's/\bunsigned requested_threads = 0U\b/unsigned requested_threads = 1U/g;' \
    -e 's/\bunsigned requested_threads = 0\b/unsigned requested_threads = 1/g;' \
    -e 's/\bint requested_threads = 0\b/int requested_threads = 1/g;' \
    -e 's/\brequested_threads = 0\b/requested_threads = 1/g;' \
    -e 's/\bself\.requested_threads = 0\b/self.requested_threads = 1/g;' \
    -e 's{(\badd_argument\(["'\'']--threads["'\''][^)]*\bdefault\s*=\s*)0\b}{${1}1}g;'

ensure_include() {
  local f=$1
  local include=$2
  local after=$3

  if [ -f "$f" ] && ! grep -qF "#include <$include>" "$f"; then
    perl -0pi -e "s{#include <${after}>\\n}{#include <${after}>\\n#include <${include}>\\n}" "$f"
  fi
}

ensure_include "$d/155.cpp" limits iostream
ensure_include "$d/640.cpp" cstdint cmath
ensure_include "$d/695.cpp" atomic algorithm

if [ -f "$d/210.cpp" ]; then
  perl -0pi \
    -e 's#u64 isqrt_u64\(u64 n\) \{\n(?!\s*if \(n == 0ULL\))#u64 isqrt_u64(u64 n) {\n    if (n == 0ULL) {\n        return 0ULL;\n    }\n#g;' \
    "$d/210.cpp"
fi

if [ -f "$d/229.cpp" ]; then
  perl -0pi \
    -e 's{std::min\(segment_low \+ segment_size - 1ULL, high\)}{std::min<u64>(segment_low + segment_size - 1ULL, high)}g;' \
    -e 's{std::min\(last, low \+ block - 1ULL\)}{std::min<u64>(last, low + block - 1ULL)}g;' \
    "$d/229.cpp"
fi

if [ -f "$d/397.cpp" ]; then
  perl -0pi \
    -e 's{std::max\(\{-x_bound, v - x_bound, floor_div\(u, 2LL\) \+ 1LL\}\)}{std::max<i64>({-x_bound, v - x_bound, floor_div(u, static_cast<i64>(2)) + static_cast<i64>(1)})}g;' \
    -e 's{std::max\(\{-x_bound, v - x_bound, floor_div\(v, 2LL\) \+ 1LL\}\)}{std::max<i64>({-x_bound, v - x_bound, floor_div(v, static_cast<i64>(2)) + static_cast<i64>(1)})}g;' \
    "$d/397.cpp"
fi

if [ -f "$d/298.cpp" ]; then
  perl -0pi \
    -e 's{constexpr long double kToleranceTight = 1e-15L;}{constexpr long double kToleranceTight = 5e-14L;}g;' \
    "$d/298.cpp"
fi

if [ -f "$d/521.cpp" ]; then
  perl -0pi \
    -e "s{std::cout << std::setw\\(9\\) << std::setfill\\('0'\\) << answer << '\\\\n';}{std::cout << answer << '\\\\n';}g;" \
    "$d/521.cpp"
fi

if [ -f "$d/664.cpp" ]; then
  perl -0pi \
    -e 's{std::min\(right, begin \+ kChunk - 1ULL\)}{std::min<u64>(right, begin + kChunk - 1ULL)}g;' \
    "$d/664.cpp"
fi

echo 'Patched eulersolve C++/Python/Java solvers to default to one worker thread.'
