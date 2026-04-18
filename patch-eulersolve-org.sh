#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

d=solvers/eulersolve
needle='No C++ compiler found'
deleted=0

if ! [ -d "$d" ]; then
  echo "No $d directory; nothing to patch."
  exit 0
fi

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
    -e 's{os\.path\.join\(\s*script_dir\s*,\s*["'\'']\.\.["'\'']\s*,\s*["'\'']resources["'\'']\s*,\s*["'\'']documents["'\'']\s*,\s*(["'\''][^"'\'']+\.(?:txt|pgm)["'\''])\s*\)}{$1}g;' \
    -e 's{os\.path\.join\(\s*os\.path\.dirname\(os\.path\.dirname\(__file__\)\)\s*,\s*["'\'']solutionsCpp["'\'']\s*,\s*["'\'']p424_kakuro200\.txt["'\'']\s*\)}{"0424_kakuro200.txt"}g;' \
    -e 's{\.\./resources/documents/}{}g;' \
    -e 's{resources/documents/}{}g;' \
    -e 's{solutionsCpp/I-expressions\.txt|(?:\.\./)?I-expressions\.txt}{0674_i_expressions.txt}g;' \
    -e 's{solutionsCpp/p424_kakuro200\.txt|p424_kakuro200\.txt}{0424_kakuro200.txt}g;' \
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
    -e 's/\b(concurrent\.futures\.ProcessPoolExecutor|ProcessPoolExecutor|ThreadPoolExecutor)\(\s*\)/$1(max_workers=1)/g;' \
    -e 's/\b(concurrent\.futures\.ProcessPoolExecutor|ProcessPoolExecutor|ThreadPoolExecutor)\(\s*max_workers\s*=\s*[^,\)]+/$1(max_workers=1/g;' \
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
    -e 's/(\badd_argument\(["'\'']--threads["'\''][^)]*\bdefault\s*=\s*)0\b/${1}1/g;'

echo 'Patched eulersolve C++/Python/Java solvers to default to one worker thread.'
