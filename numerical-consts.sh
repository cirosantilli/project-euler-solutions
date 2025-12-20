#!/usr/bin/env bash
set -eu
ls -1 solvers | grep '.py$' | while IFS= read -r file; do
  count=$(grep -Po '[0-9]{3}' "solvers/$file" | wc -l | tr -d ' ')
  printf '%8d %s\n' "$count" "$file"
done | sort -n | grep -Ev ' (8|13|185|345).py$'
